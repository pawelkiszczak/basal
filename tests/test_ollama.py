"""Ollama must score every option letter, without treating a truncated top-k as zero."""
import json
import math

import httpx
import pytest

from basal import ollama


class Tokenizer:

    def convert_ids_to_tokens(self, i):
        return {1: "A", 2: "B", 3: "C"}[i]


def test_raw_prompt_letter_probabilities_and_missing_letter(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama", "num_hidden_layers": 2,
                                                      "hidden_size": 64, "vocab_size": 100}))
    real_client = httpx.Client
    monkeypatch.setattr(ollama.AutoTokenizer, "from_pretrained", lambda _: Tokenizer())
    missing = False

    def respond(request):
        nonlocal missing
        body = json.loads(request.content)
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"details": {"format": "safetensors"}, "model_info": {
                "general.architecture": "llama", "llama.block_count": 2,
                "llama.embedding_length": 64, "llama.vocab_size": 100}})
        top = [{"bytes": [66], "logprob": -0.1}, {"bytes": [32, 66], "logprob": -0.2},
               {"bytes": [65], "logprob": -1.1}]
        if not missing:
            top.append({"bytes": [67], "logprob": -2.1})
        return httpx.Response(200, json={"logprobs": [{"top_logprobs": top}]})

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(ollama.httpx, "Client", lambda **kwargs: real_client(transport=transport, **kwargs))
    backend = ollama.OllamaBackend(tmp_path, "basal-test")
    result = backend.run_shared([(["<s>first", "<s>second"], [[1, 2, 3], [1, 2, 3]])])[0]
    expected = [math.exp(-1), 1.0, math.exp(-2)]
    expected = [p / sum(expected) for p in expected]
    for actual in result:
        assert actual == pytest.approx(expected)
    missing = True
    with pytest.raises(ValueError, match="omitted an option letter"):
        backend.run(["<s>first"], [[1, 2, 3]])
