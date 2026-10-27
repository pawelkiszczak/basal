from types import SimpleNamespace

import pytest
import torch

from basal import viz
from basal.engine import EagerBackend
from basal.gemma4 import Gemma4Forward
from test_gemma4 import tiny


class Tokenizer:
    def decode(self, ids):
        return str(ids[0])


def explainer_for(model):
    ex = viz.Explainer.__new__(viz.Explainer)
    ex.gemma4 = True
    ex.model = model
    ex.text_model = model.model
    ex.head = model.get_output_embeddings()
    ex.softcap = model.config.final_logit_softcapping
    ex.dev = "cpu"
    ex.tok = Tokenizer()
    return ex


@pytest.mark.parametrize("codes", [[1, 2], [1, 2, 3, 4], list(range(1, 6)), list(range(1, 13))],
                         ids=["noul", "choice", "score", "many-options"])
def test_gemma4_lens_matches_eager_served_probabilities_at_every_layer(codes):
    model = tiny("e4b")
    ex = explainer_for(model)
    ids = [4, 5, 8, 3, 12, 6]

    layers, _ = ex.lens(ids, codes)

    assert len(layers) == model.config.num_hidden_layers
    assert all(sum(p) == pytest.approx(1.0, abs=1e-6) for p in layers)
    eager = object.__new__(EagerBackend)
    eager.head = model.get_output_embeddings()
    eager.softcap = model.config.final_logit_softcapping
    hidden = Gemma4Forward(model).hidden([[ids]])[0][0]
    served = eager._codes(hidden[None].float(), [codes])[0]
    assert layers[-1] == pytest.approx(served, abs=1e-6)


def test_gemma4_lens_applies_final_logit_softcap():
    model = tiny("e4b")
    ex = explainer_for(model)
    ids, codes = [4, 5, 8, 3, 12, 6], list(range(1, 6))

    layers, _ = ex.lens(ids, codes)
    hidden = Gemma4Forward(model).hidden([[ids]])[0][0]
    raw = torch.softmax(hidden.float() @ model.get_output_embeddings().weight[codes].float().T, -1).tolist()

    assert max(abs(a - b) for a, b in zip(layers[-1], raw)) > 1e-3


def test_gemma4_explainer_accepts_text_model_and_refuses_other_architectures(monkeypatch):
    model = tiny("e4b")

    class Server:
        def __init__(self, _args):
            self.backend = SimpleNamespace(tok=Tokenizer(), model=model, dev="cpu")
            self.fmt, self.orders, self.cal, self.name = 1, 1, {}, "tiny"

    monkeypatch.setattr(viz, "Server", Server)
    args = SimpleNamespace(model="tiny", device="cpu", dtype="bfloat16", orders=None, calibration=None)
    assert viz.Explainer(args).info["layers"] == model.config.num_hidden_layers

    unsupported = SimpleNamespace(config=SimpleNamespace(model_type="qwen3"))
    def unsupported_server(self, _args):
        self.backend = SimpleNamespace(tok=Tokenizer(), model=unsupported, dev="cpu")
        self.fmt, self.orders, self.cal, self.name = 1, 1, {}, "tiny"

    monkeypatch.setattr(Server, "__init__", unsupported_server)
    with pytest.raises(SystemExit, match="Llama-architecture models"):
        viz.Explainer(args)


def test_gemma4_explainer_refuses_media_requests():
    ex = viz.Explainer.__new__(viz.Explainer)
    with pytest.raises(ValueError, match="basal-viz reads text only: remove the media"):
        ex.explain({"media": [{"type": "image"}]})
