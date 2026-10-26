"""LoRA adapters: read an mlx-lm or PEFT adapter directory, merge it into a model's weights at load (torch, MLX, the
TPU backend's numpy weights), write it as a PEFT directory (vLLM, SGLang), and fuse it into a checkpoint
(basal-fuse-adapter) for the engines that serve weights basal cannot touch (GGUF, Ollama, llama.cpp).

Formats, by the files in the directory:
  mlx-lm  adapter_config.json (fine_tune_type "lora" or absent, lora_parameters.scale) + adapters.safetensors,
          keys <path>.lora_a [in, r] / <path>.lora_b [r, out]
  PEFT    adapter_config.json (peft_type "LORA", r, lora_alpha, use_rslora) + adapter_model.safetensors,
          keys base_model.model.<path>.lora_A.weight [r, in] / .lora_B.weight [out, r]

The merge is mlx-lm's LoRALinear.fuse on every framework: delta = (scale * b) @ a in float32 ([out, in]), cast to the
weight's dtype, added in the weight's dtype."""
import argparse
import functools
import hashlib
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MLX_WEIGHTS, PEFT_WEIGHTS, CONFIG = "adapters.safetensors", "adapter_model.safetensors", "adapter_config.json"
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")
_MLX_KEY = re.compile(r"^(.+)\.lora_([ab])$")
_PEFT_KEY = re.compile(r"^(.+)\.lora_([AB])(?:\.[^.]+)?\.weight$")


@dataclass
class Adapter:
    path: Path
    name: str  # the directory's basename
    identity: str  # name@sha256(weights file)[:12]: what a calibration file is fitted with
    scale: float
    rank: int
    fmt: str  # "mlx" | "peft"
    deltas: dict = field(default_factory=dict)  # canonical key -> (a [r, in], b [out, r]), float32


def _fail(path, why):
    raise SystemExit(f"--adapter {path}: {why}")


def _config(path):
    """-> (fmt, weights file, config dict, scale); refuses what is not a plain LoRA adapter."""
    p = Path(path)
    if not (p / CONFIG).is_file():
        _fail(path, f"no {CONFIG} (an mlx-lm or PEFT LoRA adapter directory)")
    cfg = json.loads((p / CONFIG).read_text(encoding="utf-8"))
    if "peft_type" in cfg:
        if cfg["peft_type"] != "LORA":
            _fail(path, f"peft_type {cfg['peft_type']}: only LORA adapters can be merged")
        if cfg.get("use_dora"):
            _fail(path, "a DoRA adapter (use_dora): only LORA adapters can be merged")
        if cfg.get("fan_in_fan_out"):
            _fail(path, "fan_in_fan_out (Conv1D layers): not a layout the engines use")
        if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
            _fail(path, "rank_pattern / alpha_pattern: per-module scales are not supported")
        r, alpha = cfg.get("r"), cfg.get("lora_alpha")
        if not r or alpha is None:
            _fail(path, f"{CONFIG} lacks r / lora_alpha")
        fmt, weights, scale = "peft", p / PEFT_WEIGHTS, alpha / (math.sqrt(r) if cfg.get("use_rslora") else r)
    else:
        kind = cfg.get("fine_tune_type", "lora")
        if kind != "lora":
            _fail(path, f"fine_tune_type {kind}: only LoRA adapters can be merged")
        lp = cfg.get("lora_parameters") or {}
        if "scale" not in lp:
            _fail(path, f"{CONFIG} lacks lora_parameters.scale")
        fmt, weights, scale = "mlx", p / MLX_WEIGHTS, float(lp["scale"])
    if not weights.is_file():
        _fail(path, f"no {weights.name} next to its {fmt} {CONFIG}")
    return fmt, weights, cfg, float(scale)


def _sha12(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def identity(path):
    """The adapter's identity (name@sha256 of its weights file, 12 hex) without reading the tensors."""
    _, weights, _, _ = _config(path)
    return f"{Path(path).resolve().name}@{_sha12(weights)}"


def canonical(key):
    """'layers.N.<rest>' from a module path under any framework prefix (model., base_model.model.model.,
    language_model.model. ...)."""
    m = _LAYER.search(key)
    if not m:
        raise SystemExit(f"--adapter: {key}: not a module of a decoder layer (layers.N.<module>)")
    return f"layers.{m[1]}.{m[2]}"


def read(path):
    from safetensors.numpy import load_file
    fmt, weights, _, scale = _config(path)
    pat = _MLX_KEY if fmt == "mlx" else _PEFT_KEY
    parts = {}
    for k, v in load_file(str(weights)).items():
        m = pat.match(k)
        if not m:
            _fail(path, f"{k}: not a LoRA tensor (lora_a / lora_b, lora_A / lora_B)")
        key, side = canonical(m[1]), m[2].lower()
        if side in parts.setdefault(key, {}):
            _fail(path, f"{k}: a second {side} tensor for {key}")
        v = np.asarray(v, dtype=np.float32)
        parts[key][side] = v.T if fmt == "mlx" else v  # mlx-lm stores a [in, r], b [r, out]
    deltas = {}
    for key, ab in sorted(parts.items()):
        if len(ab) != 2:
            _fail(path, f"{key}: lora_{'b' if 'a' in ab else 'a'} is missing")
        a, b = np.ascontiguousarray(ab["a"]), np.ascontiguousarray(ab["b"])
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            _fail(path, f"{key}: lora shapes a {a.shape} / b {b.shape} do not chain")
        deltas[key] = (a, b)
    if not deltas:
        _fail(path, f"{weights.name} holds no LoRA tensors")
    p = Path(path).resolve()
    return Adapter(path=p, name=p.name, identity=f"{p.name}@{_sha12(weights)}", scale=scale,
                   rank=max(a.shape[0] for a, _ in deltas.values()), fmt=fmt, deltas=deltas)


def delta(ad, key, np_dtype=np.float32):
    a, b = ad.deltas[key]
    return ((np.float32(ad.scale) * b) @ a).astype(np_dtype)


def _target(ad, layers, key):
    _, n, rest = key.split(".", 2)
    try:
        layer = layers[int(n)]
        parent = functools.reduce(getattr, rest.split(".")[:-1], layer)
        return parent, rest.split(".")[-1], getattr(parent, rest.split(".")[-1])
    except (IndexError, AttributeError):
        raise SystemExit(f"--adapter {ad.name}: {key}: no such module in the model") from None


def _check_shape(ad, key, shape):
    a, b = ad.deltas[key]
    if tuple(shape) != (b.shape[0], a.shape[1]):
        raise SystemExit(f"--adapter {ad.name}: {key}: the model's weight is {tuple(shape)}, the adapter's delta "
                         f"{(b.shape[0], a.shape[1])}")


def merge_torch(model, ad):
    """Merge into a transformers model's decoder linears in place -> the number of merged modules."""
    import torch
    root = getattr(model, "model", model)
    root = getattr(root, "language_model", root)
    layers = root.layers
    with torch.no_grad():
        for key, (a, b) in ad.deltas.items():
            _, _, mod = _target(ad, layers, key)
            if not isinstance(mod, torch.nn.Linear):
                raise SystemExit(f"--adapter {ad.name}: {key}: {type(mod).__name__} is not a linear layer the "
                                 "adapter can be merged into")
            w = mod.weight
            if w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                raise SystemExit(f"--adapter {ad.name}: {key}: FP8 weights cannot take a merge")
            _check_shape(ad, key, w.shape)
            d = (ad.scale * torch.from_numpy(b).to(w.device)) @ torch.from_numpy(a).to(w.device)
            w.add_(d.to(w.dtype))
    return len(ad.deltas)


def merge_mlx(model, ad):
    """Merge into an mlx-lm model's decoder linears in place (QuantizedLinear: dequantised, merged and re-quantised
    with its group size and bits, as LoRALinear.fuse) -> the number of merged modules."""
    import mlx.core as mx
    import mlx.nn as nn
    lm = getattr(model, "language_model", model)
    layers = lm.model.layers
    for key, (a, b) in ad.deltas.items():
        parent, attr, mod = _target(ad, layers, key)
        quantized = isinstance(mod, nn.QuantizedLinear)
        if not (quantized or isinstance(mod, nn.Linear)):
            raise SystemExit(f"--adapter {ad.name}: {key}: {type(mod).__name__} is not a linear layer the adapter "
                             "can be merged into")
        w = mod.weight
        if quantized:
            w = mx.dequantize(w, mod.scales, mod.biases, group_size=mod.group_size, bits=mod.bits, mode=mod.mode)
        _check_shape(ad, key, w.shape)
        # mlx-lm's layout and expression (lora_a [in, r], lora_b [r, out]), so the result is bit-identical to its fuse
        la, lb = mx.array(np.ascontiguousarray(a.T)), mx.array(np.ascontiguousarray(b.T))
        w = w + ((ad.scale * lb.T) @ la.T).astype(w.dtype)
        if quantized:
            out, inp = w.shape
            fused = nn.Linear(inp, out, bias="bias" in mod)
            fused.weight = w
            if "bias" in mod:
                fused.bias = mod.bias
            fused = nn.QuantizedLinear.from_linear(fused, mod.group_size, mod.bits, mode=mod.mode)
            mx.eval(fused.parameters())
            setattr(parent, attr, fused)
        else:
            mx.eval(w)
            mod.weight = w
    return len(ad.deltas)


def merge_numpy(weights, ad):
    """Merge into Hugging Face-named numpy weights ("model.layers.N.<rest>.weight"; bf16 through ml_dtypes); the
    entries are replaced (the arrays may be read-only memory maps) -> the number of merged modules."""
    for key in ad.deltas:
        name = f"model.{key}.weight"
        if name not in weights:
            raise SystemExit(f"--adapter {ad.name}: {key}: no such module in the model")
        w = np.asarray(weights[name])
        _check_shape(ad, key, w.shape)
        d = delta(ad, key, w.dtype)
        weights[name] = (w.astype(np.float32) + d.astype(np.float32)).astype(w.dtype)
    return len(ad.deltas)


def peft_dir(ad):
    """A PEFT directory of the adapter (vLLM, SGLang): its own for a PEFT adapter, else a converted copy in a temporary
    directory."""
    if ad.fmt == "peft":
        return ad.path
    from safetensors.numpy import save_file
    out = Path(tempfile.mkdtemp("basal-adapter-"))
    cfg = {"peft_type": "LORA", "r": ad.rank, "lora_alpha": ad.scale * ad.rank,
           "target_modules": sorted({k.rsplit(".", 1)[-1] for k in ad.deltas}), "bias": "none",
           "task_type": "CAUSAL_LM", "fan_in_fan_out": False, "use_rslora": False}
    (out / CONFIG).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    tensors = {}
    for key, (a, b) in ad.deltas.items():
        if a.shape[0] != ad.rank:
            raise SystemExit(f"--adapter {ad.name}: {key}: rank {a.shape[0]} differs from the adapter's {ad.rank}; "
                             "a PEFT directory has one rank")
        tensors[f"base_model.model.model.{key}.lora_A.weight"] = a
        tensors[f"base_model.model.model.{key}.lora_B.weight"] = b
    save_file(tensors, str(out / PEFT_WEIGHTS))
    return out


def main():
    ap = argparse.ArgumentParser(description="Merge a LoRA adapter into a model and write the merged checkpoint (for "
                                 "the engines that cannot take --adapter: GGUF via basal-export-gguf, Ollama, "
                                 "llama.cpp, FP8 ports).")
    ap.add_argument("--model", required=True, help="model directory or Hugging Face repo id")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", required=True, help="LoRA adapter directory (mlx-lm or PEFT)")
    ap.add_argument("--out", required=True, help="output directory (new or empty)")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"--out {out}: exists and is not empty")
    import torch
    from transformers import AutoModelForCausalLM

    from .engine import resolve
    ad = read(a.adapter)
    md = resolve(a.model, a.revision)
    model = AutoModelForCausalLM.from_pretrained(md, dtype=getattr(torch, a.dtype))
    n = merge_torch(model, ad)
    model.save_pretrained(out)
    skip = ("*.safetensors*", "CALIBRATION*", "SHA256SUMS")
    for f in sorted(Path(md).iterdir()):  # tokenizer, chat template, basal.json; save_pretrained wrote config.json
        if f.is_file() and not any(f.match(s) for s in skip) and not (out / f.name).exists():
            shutil.copy2(f, out / f.name)
    for f in sorted(ad.path.glob("CALIBRATION*.json")):  # fitted with this adapter
        shutil.copy2(f, out / f.name)
    print(f"fused {n} layers of {ad.name} into {out}")


if __name__ == "__main__":
    main()
