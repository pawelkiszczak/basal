"""MLX backend: the packed shared-prefix forward on MLX equals separate PyTorch forwards of the same checkpoint (tiny
random Llama with the basal-1.0 architecture options: attention / MLP biases, GQA, rope_theta 1e6 in rope_parameters)."""
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from basal.engine import GraphBackend, MLXBackend, load_mlx


class Stub(MLXBackend):
    def __init__(self, model_dir, dtype, quant=None):  # no tokenizer
        self.mx = mx
        self.model, theta = load_mlx(model_dir, dtype, quant)
        d = self.model.model.layers[0].self_attn.head_dim
        self.inv_freq = 1.0 / theta ** (mx.arange(0, d, 2, dtype=mx.float32) / d)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, attention_bias=True, mlp_bias=True,
                      max_position_embeddings=128, rope_parameters={"rope_type": "default", "rope_theta": 1e6},
                      tie_word_embeddings=False)
    m = LlamaForCausalLM(cfg).eval()
    for p in m.parameters():  # random biases (initialised to zero), so that a missing bias would show up
        torch.nn.init.normal_(p, std=0.2)
    d = tmp_path_factory.mktemp("tiny")
    m.save_pretrained(d)
    return d, m


def test_shared_equals_separate_torch(ckpt):
    d, m = ckpt
    be = Stub(d, "float32")
    a, b = [1, 5, 9, 11, 3, 4, 6, 40, 41, 42], [1, 5, 9, 11, 3, 8, 2, 7, 50]
    ids, pos, seg, last = GraphBackend._pack([a, b])
    pad = 3
    h = be._forward_mlx(mx.array([ids + [0] * pad]), mx.array([pos + [0] * pad]), mx.array([seg + [-1] * pad]))
    with torch.no_grad():
        ha = m.model(input_ids=torch.tensor([a])).last_hidden_state[0, -1].numpy()
        hb = m.model(input_ids=torch.tensor([b])).last_hidden_state[0, -1].numpy()
    h = np.array(h[0])
    assert np.allclose(h[last[0]], ha, atol=1e-4) and np.allclose(h[last[1]], hb, atol=1e-4)
