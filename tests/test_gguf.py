"""GGUF backend: the shared prefix expressed as llama.cpp sequences (prefix tokens in the sequences of both option
orders, all questions of a chunk in one llama_decode) gives the same letter probabilities as evaluating every option
order on its own (tiny random Llama with attention and MLP biases written as GGUF)."""
import json

import pytest

pytest.importorskip("llama_cpp")
gguf = pytest.importorskip("gguf")

import numpy as np

from basal.engine import GGUFBackend

V, H, F, L, NH, NKV = 64, 64, 96, 2, 4, 2


class Tok:  # one character = one token, ids 3..63
    def __call__(self, text, add_special_tokens=False):
        return type("E", (), {"input_ids": [3 + ord(c) % (V - 3) for c in text]})()

    def __len__(self):
        return V


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    rng = np.random.default_rng(0)
    path = tmp_path_factory.mktemp("gguf") / "tiny.gguf"
    model_dir = path.parent / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "llama", "num_hidden_layers": L, "hidden_size": H, "vocab_size": V,
    }))
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_context_length(256); w.add_embedding_length(H); w.add_block_count(L); w.add_feed_forward_length(F)
    w.add_head_count(NH); w.add_head_count_kv(NKV); w.add_rope_freq_base(1e6); w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_dimension_count(H // NH); w.add_vocab_size(V); w.add_file_type(gguf.LlamaFileType.ALL_F32)
    w.add_tokenizer_model("llama"); w.add_bos_token_id(1); w.add_eos_token_id(2)
    w.add_token_list(["<unk>", "<s>", "</s>"] + [f"t{i}" for i in range(3, V)])
    w.add_token_scores([0.0] * V)
    w.add_token_types([gguf.TokenType.UNKNOWN, gguf.TokenType.CONTROL, gguf.TokenType.CONTROL] + [gguf.TokenType.NORMAL] * (V - 3))

    def t(name, *shape):
        w.add_tensor(name, rng.normal(0, 0.3, shape).astype(np.float32))

    t("token_embd.weight", V, H); t("output.weight", V, H)
    w.add_tensor("output_norm.weight", np.ones(H, np.float32))
    kv = H // NH * NKV
    for i in range(L):
        w.add_tensor(f"blk.{i}.attn_norm.weight", np.ones(H, np.float32))
        w.add_tensor(f"blk.{i}.ffn_norm.weight", np.ones(H, np.float32))
        for n, o in (("attn_q", H), ("attn_k", kv), ("attn_v", kv), ("attn_output", H)):
            t(f"blk.{i}.{n}.weight", o, H); t(f"blk.{i}.{n}.bias", o)
        for n, o, d in (("ffn_gate", F, H), ("ffn_up", F, H), ("ffn_down", H, F)):
            t(f"blk.{i}.{n}.weight", o, d); t(f"blk.{i}.{n}.bias", o)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return path, model_dir


@pytest.fixture(scope="module")
def backend(model):
    mp = pytest.MonkeyPatch()  # stand-in tokenizer instead of a Hugging Face model directory
    gguf_path, model_dir = model
    mp.setattr("basal.engine.AutoTokenizer.from_pretrained", lambda _: Tok())
    yield GGUFBackend(model_dir, gguf_path, n_gpu_layers=0)
    mp.undo()


def test_mismatched_architecture_fails_before_context_creation(model, tmp_path, monkeypatch):
    import llama_cpp as C

    gguf_path, _ = model
    model_dir = tmp_path / "mismatched"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "llama", "num_hidden_layers": L + 1, "hidden_size": H, "vocab_size": V,
    }))
    monkeypatch.setattr("basal.engine.AutoTokenizer.from_pretrained", lambda _: Tok())
    monkeypatch.setattr(C, "llama_init_from_model",
                        lambda *args: pytest.fail("context creation must follow architecture validation"))

    with pytest.raises(ValueError, match="num_hidden_layers"):
        GGUFBackend(model_dir, gguf_path, n_gpu_layers=0)


IDS = [10, 20, 30]


def group(state, o1=" A. x B. y C. z", o2=" A. z B. y C. x"):
    return [state + o1, state + o2], [IDS, IDS]


def test_shared_prefix_equals_separate_orders(backend):
    for state in ("state one; question?", "a longer shared state for question two"):
        (p1, p2), _ = g = group(state)
        packed = backend.run_shared([g])[0]  # prefix once, in the sequences of both orders
        separate = [backend.run_shared([([p], [IDS])])[0][0] for p in (p1, p2)]
        assert np.allclose(packed, separate, atol=5e-3)  # llama.cpp kernel paths differ slightly (see below)


def test_questions_in_one_decode_are_isolated(backend):
    a = group("state one; question?")
    with_b = backend.run_shared([a, group("a different state!!!")])[0]
    with_c = backend.run_shared([a, group("another other state?")])[0]  # same length as b, other tokens
    alone = backend.run_shared([a])[0]
    assert np.array_equal(with_b, with_c)  # an answer never depends on the other questions of the chunk
    # llama.cpp computes equal-length sequences of one batch along another kernel path: small numeric differences only
    assert np.allclose(with_b, alone, atol=5e-3)
    assert not np.allclose(with_b, backend.run_shared([group("a different state!!!")])[0], atol=5e-2)
