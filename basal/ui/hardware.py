"""Local hardware detection for the UI."""
from dataclasses import asdict, dataclass
import importlib.util
import os
import platform as host_platform
import sys


@dataclass(frozen=True)
class Hardware:
    platform: str
    device_name: str
    memory_bytes: int
    system_ram_bytes: int | None
    cuda_capability: tuple[int, int] | None
    engines: frozenset[str]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["cuda_capability"] = list(self.cuda_capability) if self.cuda_capability else None
        data["engines"] = sorted(self.engines)
        return data


def _ram() -> int | None:
    try:
        if sys.platform == "darwin":
            import subprocess
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, AttributeError):
        return None


def detect() -> Hardware:
    ram = _ram()
    engines: set[str] = set()
    if all(importlib.util.find_spec(name) is not None for name in ("PIL", "soundfile", "scipy", "torchvision")):
        engines.add("media")
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            capability = torch.cuda.get_device_capability(0)
            engines.add("cuda")
            if importlib.util.find_spec("vllm"):
                engines.add("vllm")
            if importlib.util.find_spec("llama_cpp"):
                engines.add("gguf")
            return Hardware("cuda", props.name, props.total_memory, ram, capability, frozenset(engines))
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            engines.add("mps")
    except (ImportError, RuntimeError):
        pass
    if sys.platform == "darwin" and host_platform.machine() == "arm64":
        try:
            import mlx.core as mx
            info = mx.device_info()
            name = info.get("device_name", "Apple Silicon")
            budget = int(info.get("max_recommended_working_set_size", 0))
        except Exception:
            name, budget = "Apple Silicon", 0
        if importlib.util.find_spec("mlx") and importlib.util.find_spec("mlx_lm"):
            engines.add("mlx")
        if importlib.util.find_spec("llama_cpp"):
            engines.add("gguf")
        return Hardware("apple", name, budget or int((ram or 0) * .7), ram, None, frozenset(engines))
    if importlib.util.find_spec("llama_cpp"):
        engines.add("gguf")
    return Hardware("cpu", host_platform.processor() or host_platform.machine(), int((ram or 0) * .6), ram, None,
                    frozenset(engines))
