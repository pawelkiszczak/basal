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
# The basal-omni checkpoints that carry image / audio encoders (docs/OMNI.md); basal-omni-1.1-max is Gemma 4 Unified
# without them, and the quantised ports drop them.
MEDIA_KINDS = {
    ("basal-omni-1.0", "mini"): ("image", "audio"),
    ("basal-omni-1.0", "max"): ("image", "audio"),
    ("basal-omni-1.0", "ultra"): ("image",),
}


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
    media: tuple[str, ...] = ()


@dataclass(frozen=True)
class Assessment:
    variant: Variant
    runnable: bool
    reason: str | None
    mode: str | None
    est_memory_bytes: int | None
    fit: str
    media_mode: str | None
    media_reason: str | None


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
                       filename, size, family.startswith("basal-omni-"),
                       MEDIA_KINDS.get((family, tier), ()) if port == "bf16" else ())
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


def _cache_variants(cache_dir: Path) -> list[Variant]:
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound
    try:
        cache = scan_cache_dir(cache_dir)
    except (CacheNotFound, OSError, ValueError):
        return []
    out = []
    for repo in cache.repos:
        if repo.repo_type == "model" and parse_repo(repo.repo_id):
            out += _variants(repo.repo_id, {f.file_name: f.size_on_disk for rev in repo.revisions for f in rev.files})
    return out


def list_variants(refresh: bool = False, cache_dir: Path | None = None) -> tuple[list[Variant], str]:
    """All variants from the Hub (cached in this process); offline, the ones in the selected cache."""
    if refresh:
        _hub_variants.cache_clear()
    try:
        return list(_hub_variants()), "hub"
    except Exception:  # offline, rate-limited, Hub down: what is already downloaded can still be served
        if cache_dir is None:
            from huggingface_hub.constants import HF_HUB_CACHE
            cache_dir = Path(HF_HUB_CACHE)
        return _cache_variants(cache_dir), "cache"



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


def _media_mode(v: Variant, hw: Hardware) -> tuple[str | None, str | None]:
    """("eager", None) when this machine can serve the variant with --media on, else (None, why not)."""
    if not v.media:
        return None, "this checkpoint has no image or audio encoder"
    if "media" not in hw.engines:
        return None, "images and audio need pillow, soundfile, scipy and torchvision: install basal[media]"
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
    media_mode, media_reason = _media_mode(v, hw)
    return Assessment(v, mode is not None, reason, mode, est, fit, media_mode, media_reason)


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


def is_downloaded(v: Variant, cache_dir: Path) -> Path | None:
    """The local snapshot directory if the weights are in the selected cache."""
    from huggingface_hub import snapshot_download
    try:
        path = Path(snapshot_download(v.repo, cache_dir=cache_dir, local_files_only=True, **download_kwargs(v)))
    except Exception:  # not cached (or only partly)
        return None
    weights = [path / v.gguf_file] if v.gguf_file else list(path.glob("*.safetensors"))
    return path if weights and all(w.exists() for w in weights) else None


def downloaded_bytes(v: Variant, cache_dir: Path) -> int | None:
    """Bytes of the variant's weights in the selected cache, None if not downloaded."""
    path = is_downloaded(v, cache_dir)
    if path is None:
        return None
    if v.gguf_file:
        return (path / v.gguf_file).stat().st_size
    return sum(file.stat().st_size for file in path.glob("*.safetensors"))


def delete(v: Variant, cache_dir: Path) -> int:
    """Remove the variant from the selected cache and return the bytes freed. GGUF repositories hold several
    variants, so only the variant's .gguf file goes, and the whole repository once no .gguf file is left."""
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound
    try:
        cache = scan_cache_dir(cache_dir)
    except (CacheNotFound, OSError, ValueError):
        return 0
    repo = next((r for r in cache.repos if r.repo_id == v.repo and r.repo_type == "model"), None)
    if repo is None:
        return 0
    freed = 0
    if v.gguf_file:
        for revision in repo.revisions:
            for file in revision.files:
                if file.file_name == v.gguf_file:
                    file.file_path.unlink(missing_ok=True)
                    try:
                        freed += file.blob_path.stat().st_size
                        file.blob_path.unlink()
                    except FileNotFoundError:
                        pass
        cache = scan_cache_dir(cache_dir)
        repo = next((r for r in cache.repos if r.repo_id == v.repo and r.repo_type == "model"), None)
        if repo is None or any(f.file_name.lower().endswith(".gguf") for rev in repo.revisions for f in rev.files):
            return freed
    freed += repo.size_on_disk
    cache.delete_revisions(*[r.commit_hash for r in repo.revisions]).execute()
    return freed


def serve_args(v: Variant, local_dir: Path, mode: str, media_platform: str | None = None) -> list[str]:
    """basal-serve arguments; media_platform (the Hardware.platform) serves with images / audio: --mode eager
    --media on, on the Apple GPU or the CPU where there is no CUDA."""
    if media_platform:
        args = ["--model", str(local_dir), "--mode", "eager", "--media", "on"]
        return args + {"apple": ["--device", "mps"], "cpu": ["--device", "cpu"]}.get(media_platform, [])
    args = ["--model", str(local_dir), "--mode", mode]
    if v.port == "GGUF":
        args += ["--gguf", str(local_dir / v.gguf_file)]
    if mode == "eager" and v.port == "bf16" and not v.experimental:
        args += ["--device", "cpu"]  # eager is chosen for text models only on machines without a GPU
    return args
