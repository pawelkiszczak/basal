"""--adapter: the mlx-lm and PEFT formats read to the same deltas, the merge equals mlx-lm's LoRALinear.fuse on every
framework (torch, MLX bf16 and 8-bit, the TPU backend's numpy weights), basal-serve's calibration rules under an
adapter, and basal-fuse-adapter's merged checkpoint."""
import json
import math
import sys

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file, save_file
from starlette.testclient import TestClient
from transformers import AutoModelForCausalLM

import basal.server as server_module
from basal import adapter as adapter_module
from basal.server import Server, build_app, parser
from tests.test_calibrate import SHIPPED, StubBackend
from tests.test_mlx import mlx_stub, tiny_llama

Q, DOWN = "layers.1.self_attn.q_proj", "layers.1.mlp.down_proj"


def deltas(seed=0, rank=2):
    """Random LoRA factors for tiny_llama's q_proj [64, 64] and down_proj [64, 128]: (a [r, in], b [out, r])."""
    rng = np.random.default_rng(seed)
    f = lambda *s: rng.normal(0, 0.05, s).astype(np.float32)
    return {Q: (f(rank, 64), f(64, rank)), DOWN: (f(rank, 128), f(64, rank))}


def write_adapter(path, fmt, ds, scale=2.0, **cfg):
    path.mkdir(parents=True, exist_ok=True)
    rank = next(iter(ds.values()))[0].shape[0]
    if fmt == "mlx":
        conf = {"fine_tune_type": "lora", "num_layers": 2,
                "lora_parameters": {"rank": rank, "scale": scale, "dropout": 0.0,
                                    "keys": sorted({k.split(".", 2)[2] for k in ds})}}
        tensors = {f"model.{k}.lora_{s}": np.ascontiguousarray(x.T) for k, (a, b) in ds.items()
                   for s, x in (("a", a), ("b", b))}
        name = "adapters.safetensors"
    else:
        conf = {"peft_type": "LORA", "r": rank, "lora_alpha": scale * rank, "task_type": "CAUSAL_LM"}
        tensors = {f"base_model.model.model.{k}.lora_{s}.weight": x for k, (a, b) in ds.items()
                   for s, x in (("A", a), ("B", b))}
        name = "adapter_model.safetensors"
    (path / "adapter_config.json").write_text(json.dumps({**conf, **cfg}), encoding="utf-8")
    save_file(tensors, str(path / name))
    return path


def test_both_formats_read_to_the_same_deltas(tmp_path):
    ds = deltas()
    mlx = adapter_module.read(write_adapter(tmp_path / "m", "mlx", ds))
    peft = adapter_module.read(write_adapter(tmp_path / "p", "peft", ds))
    assert mlx.deltas.keys() == peft.deltas.keys() == {Q, DOWN}
    for k in ds:
        for x, y, z in zip(mlx.deltas[k], peft.deltas[k], ds[k]):
            assert np.array_equal(x, z) and np.array_equal(y, z)
    assert (mlx.scale, peft.scale, mlx.rank, mlx.fmt, peft.fmt) == (2.0, 2.0, 2, "mlx", "peft")
    rs = adapter_module.read(write_adapter(tmp_path / "rs", "peft", ds, use_rslora=True))
    assert rs.scale == pytest.approx(4 / math.sqrt(2))
    assert mlx.identity == adapter_module.identity(tmp_path / "m") and mlx.identity.startswith("m@")
    assert len(mlx.identity) == len("m@") + 12 and mlx.identity != adapter_module.identity(tmp_path / "p")


def test_framework_prefixes_are_canonical():
    for key in ("model.layers.16.self_attn.q_proj", "base_model.model.model.layers.16.self_attn.q_proj",
                "language_model.model.layers.16.self_attn.q_proj"):
        assert adapter_module.canonical(key) == "layers.16.self_attn.q_proj"
    with pytest.raises(SystemExit, match="lm_head"):
        adapter_module.canonical("lm_head")


def test_what_cannot_be_merged_is_refused_naming_the_directory(tmp_path):
    ds = deltas()
    with pytest.raises(SystemExit, match=r"--adapter .*dora: fine_tune_type dora"):
        adapter_module.read(write_adapter(tmp_path / "dora", "mlx", ds, fine_tune_type="dora"))
    with pytest.raises(SystemExit, match="DoRA"):
        adapter_module.read(write_adapter(tmp_path / "pdora", "peft", ds, use_dora=True))
    write_adapter(tmp_path / "gone", "mlx", ds)
    (tmp_path / "gone" / "adapters.safetensors").unlink()
    with pytest.raises(SystemExit, match=r"--adapter .*gone: no adapters.safetensors"):
        adapter_module.read(tmp_path / "gone")
    half = write_adapter(tmp_path / "half", "peft", ds)
    t = load_file(str(half / "adapter_model.safetensors"))
    del t[f"base_model.model.model.{Q}.lora_B.weight"]
    save_file(t, str(half / "adapter_model.safetensors"))
    with pytest.raises(SystemExit, match=f"{Q}: lora_b is missing"):
        adapter_module.read(half)


def test_torch_merge_adds_the_scaled_delta_in_the_weight_dtype(tmp_path):
    tiny_llama(tmp_path / "model")
    ad = adapter_module.read(write_adapter(tmp_path / "ad", "peft", deltas()))
    m = AutoModelForCausalLM.from_pretrained(tmp_path / "model", dtype=torch.bfloat16)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    assert adapter_module.merge_torch(m, ad) == 2
    after = m.state_dict()
    for key in (Q, DOWN):
        a, b = map(torch.from_numpy, ad.deltas[key])
        name = f"model.{key}.weight"
        assert torch.equal(after[name], before[name] + ((2.0 * b) @ a).to(torch.bfloat16))
    assert all(torch.equal(after[k], v) for k, v in before.items() if k not in (f"model.{Q}.weight",
                                                                               f"model.{DOWN}.weight"))


def test_shape_mismatch_and_unknown_modules_are_refused(tmp_path):
    tiny_llama(tmp_path / "model")
    m = AutoModelForCausalLM.from_pretrained(tmp_path / "model")
    rng = np.random.default_rng(1)
    bad = {Q: (rng.normal(size=(2, 32)).astype(np.float32), rng.normal(size=(64, 2)).astype(np.float32))}
    with pytest.raises(SystemExit, match=rf"{Q}: the model's weight is \(64, 64\), the adapter's delta \(64, 32\)"):
        adapter_module.merge_torch(m, adapter_module.read(write_adapter(tmp_path / "bad", "peft", bad)))
    nope = {"layers.1.self_attn.nope": deltas()[Q]}
    with pytest.raises(SystemExit, match="layers.1.self_attn.nope: no such module"):
        adapter_module.merge_torch(m, adapter_module.read(write_adapter(tmp_path / "nope", "peft", nope)))
    deep = {"layers.7.self_attn.q_proj": deltas()[Q]}  # the tiny model has two layers
    with pytest.raises(SystemExit, match="layers.7.self_attn.q_proj: no such module"):
        adapter_module.merge_torch(m, adapter_module.read(write_adapter(tmp_path / "deep", "peft", deep)))


def test_numpy_merge_equals_the_torch_merge(tmp_path):
    tiny_llama(tmp_path / "model")  # float32 weights
    ad = adapter_module.read(write_adapter(tmp_path / "ad", "mlx", deltas()))
    w = load_file(str(tmp_path / "model" / "model.safetensors"))
    assert adapter_module.merge_numpy(w, ad) == 2
    m = AutoModelForCausalLM.from_pretrained(tmp_path / "model", dtype=torch.float32)
    adapter_module.merge_torch(m, ad)
    for key in (Q, DOWN):
        np.testing.assert_allclose(w[f"model.{key}.weight"], m.state_dict()[f"model.{key}.weight"].numpy(), atol=1e-6)


def _mlx_fused(path, adapter_dir, quantize=False):
    """mlx-lm's own fuse (python -m mlx_lm fuse) of a fresh load."""
    import mlx.nn as nn
    from mlx.utils import tree_unflatten
    from mlx_lm.utils import load_adapters, load_model

    from basal.engine import mlx_config_overrides
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    model, _ = load_model(path, model_config=mlx_config_overrides(cfg))
    if quantize:
        nn.quantize(model, group_size=64, bits=8)
    load_adapters(model, str(adapter_dir))
    model.update_modules(tree_unflatten([(n, m.fuse()) for n, m in model.named_modules() if hasattr(m, "fuse")]))
    return model


@pytest.mark.parametrize("quantize", [False, True], ids=["bf16", "8bit"])
def test_mlx_merge_equals_mlx_lm_fuse_bit_for_bit(tmp_path, quantize):
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn
    tiny_llama(tmp_path / "model")
    ad_dir = write_adapter(tmp_path / "ad", "mlx", deltas())
    be = mlx_stub(tmp_path / "model")
    if quantize:  # an -MLX-8bit export: dequantised, merged, re-quantised
        nn.quantize(be.model, group_size=64, bits=8)
    assert adapter_module.merge_mlx(be.model, adapter_module.read(ad_dir)) == 2
    ref = _mlx_fused(tmp_path / "model", ad_dir, quantize)
    for key in (Q, DOWN):
        _, n, rest = key.split(".", 2)
        mine, theirs = be.model.model.layers[int(n)], ref.model.layers[int(n)]
        for attr in rest.split("."):
            mine, theirs = getattr(mine, attr), getattr(theirs, attr)
        assert type(mine) is type(theirs) is (nn.QuantizedLinear if quantize else nn.Linear)
        for p in ("weight", "scales", "biases") if quantize else ("weight",):
            assert mx.array_equal(getattr(mine, p), getattr(theirs, p))


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A model directory with the shipped CALIBRATION.json behind the stub engine (--mode eager), and an adapter."""
    md = tmp_path / "model"
    md.mkdir()
    (md / "CALIBRATION.json").write_text(json.dumps(SHIPPED), encoding="utf-8")
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: md)
    monkeypatch.setattr(server_module, "EagerBackend", StubBackend)
    return md, write_adapter(tmp_path / "edu", "mlx", deltas())


def serve(md, *args):
    a = parser().parse_args(["--mode", "eager", "--model", str(md), *args])
    return Server(a), a


def test_the_adapters_calibration_file_is_served_with_its_name(served):
    md, ad = served
    ident = adapter_module.identity(ad)
    own = dict(SHIPPED, source_tag="adapter", fitted_on={"model": "model", "adapter": ident})
    (ad / "CALIBRATION.json").write_text(json.dumps(own), encoding="utf-8")
    srv, a = serve(md, "--adapter", str(ad))
    assert srv.cal["source_tag"] == "adapter" and srv.cal_unvalidated is None and srv.cal_mismatch is None
    assert srv.name == "model+edu"
    entry = TestClient(build_app(srv, a)).get("/v1/models").json()["models"][0]
    assert entry["name"] == "model+edu" and entry["adapter"] == ident and entry["certified"]
    assert "adapter" not in TestClient(build_app(*serve(md))).get("/v1/models").json()["models"][0]


def test_the_base_models_file_is_unvalidated_under_an_adapter(served):
    md, ad = served
    srv, _ = serve(md, "--adapter", str(ad))
    assert srv.cal["temperature_per_prim"] == SHIPPED["temperature_per_prim"] and "thresholds" not in srv.cal
    assert srv.cal_unvalidated == "CALIBRATION.json belongs to the base model, served with --adapter edu"
    assert serve(md)[0].cal["thresholds"] == SHIPPED["thresholds"]  # without the adapter: the model's own


def test_a_file_fitted_with_another_adapter_warns_or_is_refused(served, tmp_path, capsys):
    md, ad = served
    f = tmp_path / "C.json"
    f.write_text(json.dumps(dict(SHIPPED, fitted_on={"model": "model", "adapter": "other@000000000000"})),
                 encoding="utf-8")
    srv, _ = serve(md, "--adapter", str(ad), "--calibration", str(f))
    want = f"it was fitted with adapter other@000000000000, the served adapter is {adapter_module.identity(ad)}"
    assert srv.cal_mismatch == want and "warning: C.json" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--strict-calibration"):
        serve(md, "--adapter", str(ad), "--calibration", str(f), "--strict-calibration")
    fitted = tmp_path / "F.json"  # fitted with the adapter, served without it
    fitted.write_text(json.dumps(dict(SHIPPED, fitted_on={"model": "model", "adapter": "edu@0123456789ab"})),
                      encoding="utf-8")
    assert serve(md, "--calibration", str(fitted))[0].cal_mismatch == ("it was fitted with adapter edu@0123456789ab, "
                                                                       "the served adapter is none")
    (ad / "CALIBRATION.json").write_text(fitted.read_text(encoding="utf-8"), encoding="utf-8")  # an older adapter's
    assert "edu@0123456789ab" in serve(md, "--adapter", str(ad))[0].cal_mismatch


@pytest.mark.parametrize("mode", ["gguf", "ollama", "llamacpp"])
def test_gguf_engines_refuse_an_adapter_before_loading(served, mode):
    md, ad = served
    with pytest.raises(SystemExit, match="basal-fuse-adapter"):
        serve(md, "--adapter", str(ad), "--mode", mode, "--gguf", "x.gguf")


def test_fuse_adapter_writes_a_merged_checkpoint(tmp_path, monkeypatch, capsys):
    md = tmp_path / "model"
    tiny_llama(md)
    m = AutoModelForCausalLM.from_pretrained(md, dtype=torch.bfloat16)
    (md / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (md / "CALIBRATION.json").write_text("{}", encoding="utf-8")  # the base model's: not the merged model's
    ad = write_adapter(tmp_path / "edu", "peft", deltas())
    (ad / "CALIBRATION.json").write_text('{"adapter": 1}', encoding="utf-8")
    out = tmp_path / "fused"
    monkeypatch.setattr(sys, "argv", ["basal-fuse-adapter", "--model", str(md), "--adapter", str(ad), "--out", str(out)])
    adapter_module.main()
    assert f"fused 2 layers of edu into {out}" in capsys.readouterr().out
    adapter_module.merge_torch(m, adapter_module.read(ad))
    fused = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16)
    for k, v in m.state_dict().items():
        assert torch.equal(fused.state_dict()[k], v), k
    assert (out / "tokenizer_config.json").exists()
    assert json.loads((out / "CALIBRATION.json").read_text(encoding="utf-8")) == {"adapter": 1}
    with pytest.raises(SystemExit, match="not empty"):
        adapter_module.main()
