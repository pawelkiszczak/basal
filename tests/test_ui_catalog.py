import math
from pathlib import Path

from basal.ui.catalog import Variant, _label, _media_mode, _variants, assess, delete, download_kwargs, parse_repo, recommend, serve_args
from basal.ui.hardware import Hardware


def hw(platform="cuda", name="H100", cap=(9, 0), engines=frozenset({"cuda"}), memory=80 * 1024**3):
    return Hardware(platform, name, memory, memory, cap, engines)

def variant(repo="Remek/basal-2.0-4.5B", port=None, size=None, gguf=None):
    family, tier, parsed_port = parse_repo(repo)
    port = port or parsed_port
    size = size if size is not None else int({"mini": 2, "4.5B": 8, "max": 12, "ultra": 16}.get(tier, 8) * 1024**3)
    return Variant(f"{repo}:{gguf}" if gguf else repo, repo, family, tier, port, _label(port, gguf), gguf, size,
                   family.startswith("basal-omni-"))



def test_hardware_detects_media_engine(monkeypatch):
    from basal.ui import hardware

    find_spec = hardware.importlib.util.find_spec
    packages = {"PIL", "soundfile", "scipy", "torchvision"}
    monkeypatch.setattr(hardware.importlib.util, "find_spec",
                        lambda name: object() if name in packages else find_spec(name))
    assert "media" in hardware.detect().engines
    monkeypatch.setattr(hardware.importlib.util, "find_spec",
                        lambda name: None if name == "torchvision" else find_spec(name))
    assert "media" not in hardware.detect().engines


def test_media_kinds_and_mode_eligibility():
    for tier, kinds in (("mini", ("image", "audio")), ("max", ("image", "audio")), ("ultra", ("image",))):
        (v,) = _variants(f"Remek/basal-omni-1.0-{tier}", {})
        assert v.media == kinds
        (mlx,) = _variants(f"Remek/basal-omni-1.0-{tier}-MLX-8bit", {})
        assert mlx.media == ()
    for repo in ("Remek/basal-omni-1.1-max", "Remek/basal-omni-1.0-mini-MLX"):
        (v,) = _variants(repo, {})
        assert v.media == ()

    v = _variants("Remek/basal-omni-1.0-mini", {})[0]
    for platform in ("cuda", "apple", "cpu"):
        assert _media_mode(v, hw(platform, engines=frozenset({"media"}))) == ("eager", None)
        mode, reason = _media_mode(v, hw(platform, engines=frozenset()))
        assert mode is None and "install basal[media]" in reason
    assert _media_mode(_variants("Remek/basal-2.0-mini", {})[0], hw())[1] == \
        "this checkpoint has no image or audio encoder"


def test_media_serve_args_devices():
    v = _variants("Remek/basal-omni-1.0-mini", {})[0]
    for platform, device in (("cuda", []), ("apple", ["--device", "mps"]), ("cpu", ["--device", "cpu"])):
        args = serve_args(v, Path("/model"), "mlx", media_platform=platform)
        assert args == ["--model", "/model", "--mode", "eager", "--media", "on", *device]


def test_parse_repo_contract_names():
    assert parse_repo("Remek/basal-omni-1.0-max-GGUF") == ("basal-omni-1.0", "max", "GGUF")
    assert parse_repo("Remek/basal-2.0-4.5B-MLX-8bit") == ("basal-2.0", "4.5B", "MLX-8bit")
    assert parse_repo("Remek/basalt-1.0-4.5B") is None
    assert parse_repo("Remek/basal-1.0-4.5B-GGUF-extra") is None


def test_assessment_runtime_engines_and_fit_boundaries():
    fp8 = variant("Remek/basal-2.0-4.5B-FP8", "FP8")
    assert assess(fp8, hw(cap=(8, 9))).runnable
    assert not assess(fp8, hw(cap=(8, 6))).runnable
    mlx = variant("Remek/basal-2.0-4.5B-MLX-8bit", "MLX-8bit")
    assert not assess(mlx, hw(platform="cuda", engines=frozenset({"cuda"}))).runnable
    assert assess(mlx, hw(platform="apple", cap=None, engines=frozenset({"mlx"}))).runnable
    gguf = variant("Remek/basal-2.0-4.5B-GGUF", "GGUF", gguf="model-Q8_0.gguf")
    a = assess(gguf, hw(engines=frozenset()))
    assert not a.runnable and "basal[gguf]" in a.reason
    assert not assess(variant("Remek/basal-omni-1.0-mini", "bf16"), hw(platform="cpu", cap=None, engines=frozenset())).runnable
    size = 10 * 1024**3
    v = variant(size=size)
    assert assess(v, hw(memory=math.ceil((size * 1.2 + 1024**3) / .85))).fit == "fits"
    budget = size * 1.2 + 1024**3
    assert assess(v, hw(memory=int(budget))).fit == "tight"
    assert assess(Variant("unknown", "Remek/basal-2.0-4.5B", "basal-2.0", "4.5B", "bf16", "bf16", None, None, False), hw()).fit == "unknown"


def test_recommendations_by_hardware():
    def choices(hw_, repos):
        return recommend([assess(variant(repo, "MLX-8bit" if "MLX" in repo else "FP8" if "FP8" in repo else "bf16"), hw_) for repo in repos], hw_)
    apple = hw("apple", "M4 Max", None, frozenset({"mlx"}), 24 * 1024**3)
    assert choices(apple, ["Remek/basal-2.0-4.5B-MLX-8bit", "Remek/basal-2.0-max-MLX-8bit", "Remek/basal-2.0-mini-MLX-8bit"]) == {
        "recommended": "Remek/basal-2.0-4.5B-MLX-8bit", "best_quality": "Remek/basal-2.0-max-MLX-8bit"}
    small = hw("apple", "M4", None, frozenset({"mlx"}), 8 * 1024**3)
    assert choices(small, ["Remek/basal-2.0-mini-MLX-8bit", "Remek/basal-2.0-4.5B-MLX-8bit"]) ["recommended"] == "Remek/basal-2.0-mini-MLX-8bit"
    dc = hw("cuda", "H100", (9, 0), frozenset({"cuda"}), 200 * 1024**3)
    assert choices(dc, ["Remek/basal-2.0-4.5B", "Remek/basal-2.0-4.5B-FP8"])["recommended"] == "Remek/basal-2.0-4.5B"
    for name, cap, expected in (("RTX 5090", (12, 0), "FP8"), ("RTX 4090", (8, 9), "bf16")):
        gpu = hw("cuda", name, cap, frozenset({"cuda"}), 100 * 1024**3)
        result = choices(gpu, ["Remek/basal-2.0-4.5B", "Remek/basal-2.0-4.5B-FP8"])["recommended"]
        assert result.endswith("FP8") if expected == "FP8" else result == "Remek/basal-2.0-4.5B"
    cpu = hw("cpu", "CPU", None, frozenset(), 128 * 1024**3)
    result = choices(cpu, ["Remek/basal-2.0-4.5B"])
    assert result["recommended"] == "Remek/basal-2.0-4.5B"
    assert "--device" in serve_args(variant(), Path("/model"), "eager")


def test_gguf_launch_and_download_patterns():
    v = variant("Remek/basal-2.0-4.5B-GGUF", "GGUF", gguf="basal-2.0-4.5B-Q8_0.gguf")
    assert serve_args(v, Path("/models/repo"), "gguf")[-2:] == ["--gguf", "/models/repo/basal-2.0-4.5B-Q8_0.gguf"]
    assert download_kwargs(v)["allow_patterns"][0] == v.gguf_file


def test_thinking_model_is_neither_counted_nor_downloaded():
    (v,) = _variants("Remek/basal-2.0-4.5B", {"model.safetensors": 9, "think/model.safetensors": 9, "basal.json": 1})
    assert v.weights_bytes == 9
    assert download_kwargs(v) == {"ignore_patterns": ["think/*"]}


def test_serving_mode_per_family_and_port():
    apple = hw("apple", "M4 Max", None, frozenset({"mlx"}), 64 * 1024**3)
    assert assess(variant("Remek/basal-2.0-4.5B-MLX-8bit", "MLX-8bit"), apple).mode == "mlx"  # the port as stored
    h100 = hw()
    assert assess(variant("Remek/basal-omni-1.0-max", "bf16"), h100).mode == "eager"
    older_fp8 = assess(variant("Remek/basal-1.5-4.5B-FP8", "FP8"), h100)
    assert not older_fp8.runnable and "basal[vllm]" in older_fp8.reason
    assert assess(variant("Remek/basal-1.5-4.5B-FP8", "FP8"), hw(engines=frozenset({"cuda", "vllm"}))).mode == "vllm"
    assert "--device" not in serve_args(variant("Remek/basal-omni-1.0-max", "bf16"), Path("/m"), "eager")


def test_cpu_with_llama_cpp_prefers_gguf_q8():
    cpu = hw("cpu", "CPU", None, frozenset({"gguf"}), 64 * 1024**3)
    repo = "Remek/basal-2.0-4.5B-GGUF"
    items = [assess(Variant(f"{repo}:{f}", repo, "basal-2.0", "4.5B", "GGUF", label, f, 5 * 1024**3, False), cpu)
             for f, label in (("basal-2.0-4.5B-Q5_K_M.gguf", "GGUF Q5_K_M"), ("basal-2.0-4.5B-Q8_0.gguf", "GGUF Q8_0"))]
    items.append(assess(variant(), cpu))
    assert recommend(items, cpu)["recommended"] == f"{repo}:basal-2.0-4.5B-Q8_0.gguf"


def test_delete_gguf_variants_removes_only_selected_file_then_repo(tmp_path):
    repo_id = "Remek/basal-2.0-mini-GGUF"
    repo = tmp_path / "models--Remek--basal-2.0-mini-GGUF"
    blobs, snapshot = repo / "blobs", repo / "snapshots" / ("a" * 40)
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("a" * 40)
    names = ("basal-2.0-mini-Q4_K_M.gguf", "basal-2.0-mini-Q8_0.gguf")
    for index, name in enumerate(names, 1):
        blob = blobs / (str(index) * 64)
        blob.write_bytes(bytes([index]) * index)
        (snapshot / name).symlink_to(blob)
    first, second = _variants(repo_id, {name: index for index, name in enumerate(names, 1)})

    assert delete(first, tmp_path) == 1
    assert not (snapshot / names[0]).exists()
    assert (snapshot / names[1]).exists()
    assert repo.is_dir()
    assert delete(second, tmp_path) == 2
    assert not repo.exists()
