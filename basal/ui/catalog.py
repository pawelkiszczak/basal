"""Hugging Face catalog of the basal models and hardware-aware model selection.

Each repository of Remek/basal-* (and each .gguf file of a -GGUF repository) is one Variant. assess() says whether
this machine can serve it with basal-serve, in which --mode, and whether it fits the memory budget; recommend()
picks a model the way docs/HARDWARE.md (format per GPU class) and the README (start with 4.5B) advise.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

from .hardware import Hardware

_REPO_RE = re.compile(r"^basal-(omni-)?(\d+\.\d+)-(mini|max|ultra|\d+(?:\.\d+)?B)(?:-(FP8|NVFP4|GGUF|MLX(?:-8bit|-fp4)?))?$")
TIER_ORDER = {"mini": 0, "1.5B": 1, "4.5B": 2, "max": 3, "ultra": 4}
CURRENT_TEXT_FAMILY = "basal-2.0"
# docs/HARDWARE.md, Recommendation: bf16 on data-centre GPUs (launch-bound at batch 1) and on the RTX 4090 (FP8
# compilation stalled there); FP8 on other GPUs with compute capability 8.9+
_BF16_GPUS = ("A100", "H100", "H200", "H800", "B100", "B200", "B300", "GH200", "GB200", "4090")
GIB = 1024**3


@dataclass(frozen=True)
class Variant:
    id: str
    repo: str
    family: str
    tier: str
    port: str
    label: str
    gguf_file: str | None
    weights_bytes: int | None
    experimental: bool


@dataclass(frozen=True)
class Assessment:
    variant: Variant
    runnable: bool
    reason: str | None
    mode: str | None
    est_memory_bytes: int | None
    fit: str


def parse_repo(repo_id: str) -> tuple[str, str, str] | None:
    """(family, tier, port) of an official repository name, None for anything else."""
    match = _REPO_RE.fullmatch(repo_id.rsplit("/", 1)[-1])
    if not match:
        return None
    omni, version, tier, port = match.groups()
    return f"basal-{'omni-' if omni else ''}{version}", tier, port or "bf16"


def _label(port: str, filename: str | None = None) -> str:
    if port == "GGUF" and filename:
        quant = re.search(r"-(Q\w+|F16|BF16)\.gguf$", filename, re.I)
        return f"GGUF {quant.group(1)}" if quant else "GGUF"
    return {"MLX-8bit": "MLX 8-bit", "MLX-fp4": "MLX fp4", "MLX": "MLX 8-bit"}.get(port, port)


def _variants(repo: str, files: dict[str, int | None]) -> list[Variant]:
    """The variants of one repository from its file names and sizes."""
    family, tier, port = parse_repo(repo)
    def make(filename, size):
        return Variant(f"{repo}:{filename}" if filename else repo, repo, family, tier, port, _label(port, filename),
                       filename, size, family.startswith("basal-omni-"))
    if port == "GGUF":
        return [make(name, size) for name, size in files.items() if name.lower().endswith(".gguf")]
    sizes = [size for name, size in files.items() if name.endswith(".safetensors") and not _optional(name)]
    return [make(None, sum(sizes) if sizes and None not in sizes else None)]


@lru_cache(maxsize=1)
def _hub_variants() -> tuple[Variant, ...]:
    from huggingface_hub import HfApi
    api = HfApi()
    repos = [m.id for m in api.list_models(author="Remek", search="basal") if parse_repo(m.id)]
    def get(repo):
        info = api.model_info(repo, files_metadata=True)
        return _variants(repo, {f.rfilename: f.size for f in info.siblings or []})
    with ThreadPoolExecutor(max_workers=8) as pool:
        return tuple(v for group in pool.map(get, repos) for v in group)


def _cache_variants() -> list[Variant]:
    from huggingface_hub import scan_cache_dir
    out = []
    for repo in scan_cache_dir().repos:
        if repo.repo_type == "model" and parse_repo(repo.repo_id):
            out += _variants(repo.repo_id, {f.file_name: f.size_on_disk for rev in repo.revisions for f in rev.files})
    return out


def list_variants(refresh: bool = False) -> tuple[list[Variant], str]:
    """All variants from the Hub (cached in this process); offline, the ones in the local Hugging Face cache."""
    if refresh:
        _hub_variants.cache_clear()
    try:
        return list(_hub_variants()), "hub"
    except Exception:  # offline, rate-limited, Hub down: what is already downloaded can still be served
        return _cache_variants(), "cache"


def _mode(v: Variant, hw: Hardware) -> tuple[str | None, str | None]:
    """(basal-serve --mode, None) or (None, why this machine cannot serve the variant)."""
    cap = hw.cuda_capability or (0, 0)
    if v.port == "FP8":
        if hw.platform != "cuda" or cap < (8, 9):
            return None, "FP8 checkpoints need an NVIDIA GPU with compute capability 8.9+ (Ada, Hopper, Blackwell)."
        if v.family == CURRENT_TEXT_FAMILY:  # loaded as stored by the basal engine
            return "fast", None
        if "vllm" not in hw.engines:  # the older ModelOpt FP8 ports and basal-omni's are checked with vLLM
            return None, "This FP8 checkpoint is served with vLLM: install basal[vllm] (own environment)."
        return "vllm", None
    if v.port == "NVFP4":
        if hw.platform != "cuda" or cap < (10, 0):
            return None, "NVFP4 checkpoints need a Blackwell GPU (compute capability 10.0+)."
        if "vllm" not in hw.engines:
            return None, "NVFP4 checkpoints are served with vLLM: install basal[vllm] (own environment)."
        return "vllm", None
    if v.port.startswith("MLX"):
        if hw.platform != "apple":
            return None, "MLX ports run on Apple Silicon only."
        if "mlx" not in hw.engines:
            return None, "MLX ports need mlx and mlx-lm: install basal[mlx]."
        return "mlx", None
    if v.port == "GGUF":
        if "gguf" not in hw.engines:
            return None, "GGUF files need llama-cpp-python: install basal[gguf]."
        return "gguf", None
    # bf16 weights
    if hw.platform == "cuda":
        return ("eager" if v.experimental else "fast"), None  # basal-omni: eager runs on every GPU (docs/OMNI.md)
    if hw.platform == "apple":
        if "mlx" in hw.engines:
            return "mlx", None
        if v.experimental:
            return None, "basal-omni on Apple Silicon needs mlx and mlx-lm: install basal[mlx]."
        return "mps", None
    if v.experimental:
        return None, "basal-omni needs an NVIDIA GPU or Apple Silicon."
    return "eager", None


def assess(v: Variant, hw: Hardware) -> Assessment:
    mode, reason = _mode(v, hw)
    est = int(v.weights_bytes * 1.2) + GIB if v.weights_bytes is not None else None  # weights + runtime + KV cache
    if est is None or not hw.memory_bytes:
        fit = "unknown"
    elif est <= 0.85 * hw.memory_bytes:
        fit = "fits"
    elif est <= hw.memory_bytes:
        fit = "tight"
    else:
        fit = "too_large"
    return Assessment(v, mode is not None, reason, mode, est, fit)


def preferred_format(hw: Hardware) -> str:
    """The format label to recommend on this machine."""
    if hw.platform == "cuda":
        name = hw.device_name.upper()
        if (hw.cuda_capability or (0, 0)) >= (8, 9) and not any(gpu in name for gpu in _BF16_GPUS):
            return "FP8"
        return "bf16"
    if hw.platform == "apple" and "mlx" in hw.engines:
        return "MLX 8-bit"
    if "gguf" in hw.engines:
        return "GGUF Q8_0"
    return "bf16"


def recommend(items: list[Assessment], hw: Hardware) -> dict:
    """{"recommended": id, "best_quality": id}: in the current text family (else the newest non-omni family) and the
    preferred format, 4.5B if it fits, else mini; best quality = the largest tier that fits. A tight fit is used only
    when nothing fits comfortably."""
    usable = [a for a in items if a.runnable and a.fit != "too_large"]
    families = sorted({a.variant.family for a in usable if not a.variant.experimental}, reverse=True)
    if not families:
        return {"recommended": None, "best_quality": None}
    family = CURRENT_TEXT_FAMILY if CURRENT_TEXT_FAMILY in families else families[0]
    pool = [a for a in usable if a.variant.family == family]
    fmt = preferred_format(hw)
    pool = [a for a in pool if a.variant.label == fmt] or pool
    pool = [a for a in pool if a.fit == "fits"] or pool
    by_tier = sorted(pool, key=lambda a: TIER_ORDER.get(a.variant.tier, -1))
    recommended = next((a for a in by_tier if a.variant.tier == "4.5B"), None) or by_tier[0]
    return {"recommended": recommended.variant.id, "best_quality": by_tier[-1].variant.id}


def _optional(name: str) -> bool:
    """Files the default serving modes never read: the 4.5B's thinking-mode model (think/, a second full copy)."""
    return name.startswith("think/")


def download_kwargs(v: Variant) -> dict:
    """snapshot_download filters: for GGUF the chosen file plus tokenizer, prompt format and calibration (a -GGUF
    repository holds several quantisations); otherwise the repository without the optional files."""
    if v.port == "GGUF":
        return {"allow_patterns": [v.gguf_file, "*.json", "*.jinja", "*.model", "*.txt"]}
    return {"ignore_patterns": ["think/*"]}


def is_downloaded(v: Variant) -> Path | None:
    """The local snapshot directory if the weights are in the Hugging Face cache."""
    from huggingface_hub import snapshot_download
    try:
        path = Path(snapshot_download(v.repo, local_files_only=True, **download_kwargs(v)))
    except Exception:  # not cached (or only partly)
        return None
    weights = [path / v.gguf_file] if v.gguf_file else list(path.glob("*.safetensors"))
    return path if weights and all(w.exists() for w in weights) else None


def serve_args(v: Variant, local_dir: Path, mode: str) -> list[str]:
    args = ["--model", str(local_dir), "--mode", mode]
    if v.port == "GGUF":
        args += ["--gguf", str(local_dir / v.gguf_file)]
    if mode == "eager" and v.port == "bf16" and not v.experimental:
        args += ["--device", "cpu"]  # eager is chosen for text models only on machines without a GPU
    return args
