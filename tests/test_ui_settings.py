from pathlib import Path

import pytest
from huggingface_hub.constants import HF_HUB_CACHE

from basal.ui import settings


def test_set_and_reset_models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    target = tmp_path / "models"

    assert settings.set_models_dir(str(target)) == target
    assert target.is_dir()
    assert settings.models_dir() == target
    assert settings.set_models_dir(None) == Path(HF_HUB_CACHE)
    assert settings.models_dir() == Path(HF_HUB_CACHE)


def test_set_models_dir_rejects_unwritable_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    target = tmp_path / "readonly"
    target.mkdir()
    target.chmod(0o500)
    try:
        with pytest.raises(ValueError, match="is not writable"):
            settings.set_models_dir(str(target))
    finally:
        target.chmod(0o700)


def test_invalid_json_shape_uses_default_and_relative_path_is_absolute(tmp_path, monkeypatch):
    config = tmp_path / "settings.json"
    monkeypatch.setenv("BASAL_UI_CONFIG", str(config))
    monkeypatch.chdir(tmp_path)
    config.write_text("[]")
    assert settings.models_dir() == Path(HF_HUB_CACHE)
    stored = settings.set_models_dir("relative-models-dir")
    assert stored == (tmp_path / "relative-models-dir").resolve()
    assert settings.load()["models_dir"] == str(stored)


def test_serve_settings_validation_and_api_key_file(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    assert settings.serve_settings()["enabled"] is False
    for patch, message in (({"port": 80}, "port"), ({"host": "example.com"}, "host"),
                           ({"max_len": 256}, "max_len"), ({"cors_origins": ["not-an-origin"]}, "cors_origins")):
        with pytest.raises(ValueError, match=message):
            settings.update_serve(patch)
    settings.update_serve({"enabled": True, "cors_origins": ["https://example.com"]})
    record, secret = settings.create_key("laptop")
    assert secret.startswith("bsl_") and record["prefix"] == secret[:8]
    path = settings.keys_file("a" * 64)
    content = __import__("json").loads(path.read_text())
    assert content["keys"] == [
        {"name": "laptop", "sha256": __import__("hashlib").sha256(secret.encode()).hexdigest(), "admin": False},
        {"name": "basal-ui", "sha256": "a" * 64, "admin": True},
    ]
    assert path.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "settings.json").stat().st_mode & 0o777 == 0o600
    settings.delete_key("laptop")
    assert settings.keys() == []


def test_serve_settings_reject_invalid_fields_and_normalize_log_path(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    invalid = (
        ({"enabled": "yes"}, "enabled"),
        ({"require_key": 1}, "require_key"),
        ({"mcp": 0}, "mcp"),
        ({"host": "10.0.0.1"}, "host"),
        ({"port": 80}, "port"),
        ({"port": 70000}, "port"),
        ({"max_len": 100}, "max_len"),
        ({"max_len": 131073}, "max_len"),
        ({"cors_origins": ["not-an-origin"]}, "cors_origins"),
        ({"cors_origins": ["https://example.com/path"]}, "cors_origins"),
    )
    for patch, field in invalid:
        with pytest.raises(ValueError, match=field):
            settings.update_serve(patch)
    with pytest.raises(ValueError, match="decision log directory"):
        settings.update_serve({"log_decisions": str(tmp_path / "missing" / "decisions.jsonl")})
    log_path = tmp_path / "decisions.jsonl"
    settings.update_serve({"log_decisions": str(log_path)})
    assert settings.serve_settings()["log_decisions"] == str(log_path.resolve())


def test_mcp_setting_requires_available_extra(tmp_path, monkeypatch):
    from basal import mcp_server

    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    monkeypatch.setattr(mcp_server, "available", lambda: "install basal[mcp]")
    with pytest.raises(ValueError, match="install basal"):
        settings.update_serve({"mcp": True})


def test_api_key_names_are_unique_validated_and_revocable(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    with pytest.raises(ValueError, match="name must"):
        settings.create_key("")
    with pytest.raises(ValueError, match="name must"):
        settings.create_key("a" * 65)
    record, _secret = settings.create_key("laptop")
    with pytest.raises(ValueError, match="already exists"):
        settings.create_key("laptop")
    assert settings.keys() == [{**record, "sha256": settings.load()["api_keys"][0]["sha256"]}]
    settings.keys_file("a" * 64)
    first = (tmp_path / "api-keys.json").read_text()
    settings.delete_key("laptop")
    second = (tmp_path / "api-keys.json").read_text()
    assert "laptop" in first and "laptop" not in second
    with pytest.raises(ValueError, match="unknown API key"):
        settings.delete_key("laptop")


def test_auto_load_setting_requires_boolean(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    assert settings.set_auto_load(True) is True
    assert settings.load()["auto_load"] is True
    with pytest.raises(ValueError, match="auto_load must be a boolean"):
        settings.set_auto_load("yes")
