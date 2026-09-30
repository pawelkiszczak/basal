"""Ollama safetensors import: read next-token letter probabilities from /api/generate.

Ollama accepts prompt text rather than Hugging Face token ids. Import the original
safetensors checkpoint, not a GGUF/MLX conversion, and use raw prompts so Ollama
does not insert its own template. A truncated top-logprob list cannot represent
a typed decision; fail instead of assigning a missing letter probability zero.
"""
import json
import math
from pathlib import Path

import httpx
from transformers import AutoTokenizer

from .prompt import PREFILL


class OllamaBackend:
    def __init__(self, model_dir, ollama_model, url="http://127.0.0.1:11434"):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.prefill = PREFILL
        self.model = ollama_model
        self.http = httpx.Client(base_url=url.rstrip("/"), timeout=120)
        info = self.http.post("/api/show", json={"model": ollama_model})
        info.raise_for_status()
        details = info.json()
        if details.get("details", {}).get("format") != "safetensors":
            raise ValueError("Ollama mode needs an import of the original safetensors checkpoint, not a GGUF file")
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        got = details.get("model_info", {})
        expected = {"general.architecture": cfg["model_type"], "llama.block_count": cfg["num_hidden_layers"],
                    "llama.embedding_length": cfg["hidden_size"], "llama.vocab_size": cfg["vocab_size"]}
        if any(got.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Ollama model {ollama_model!r} does not match --model architecture/vocabulary")

    def run(self, prompts, ids_list, policy=None):
        out = []
        for prompt, ids in zip(prompts, ids_list):
            want = {self.tok.convert_ids_to_tokens(i).replace("▁", " ").encode(): j
                    for j, i in enumerate(ids)}
            if len(want) != len(ids):
                raise ValueError("option letters do not have distinct byte representations")
            response = self.http.post("/api/generate", json={
                "model": self.model, "prompt": prompt, "raw": True, "stream": False,
                "logprobs": True, "top_logprobs": 20, "options": {"num_predict": 1, "temperature": 0},
            })
            response.raise_for_status()
            entries = response.json()["logprobs"][0]["top_logprobs"]
            lp = [None] * len(ids)
            for entry in entries:
                j = want.get(bytes(entry["bytes"]))
                if j is not None and lp[j] is None:
                    lp[j] = entry["logprob"]
            if any(v is None for v in lp):
                raise ValueError("Ollama top-20 logprobs omitted an option letter; cannot return exact probabilities")
            m = max(lp)
            e = [math.exp(v - m) for v in lp]
            z = sum(e)
            out.append([v / z for v in e])
        return out

    def run_shared(self, groups, policy=None):
        return [self.run(prompts, ids_list) for prompts, ids_list in groups]
