from fastapi.testclient import TestClient

from basal.ui.app import create_app
from basal.ui.catalog import Variant
from basal.ui.hardware import Hardware


class FakeRunner:
    def __init__(self, ready=False):
        self.ready = ready
        self.model = None
    def status(self): return {"state": "ready" if self.ready else "idle", "model": self.model}
    def load(self, assessment):
        self.ready = True
        self.model = assessment.variant.id
        return self.status()
    def unload(self): self.ready = False; self.model = None; return self.status()
    def decide(self, body):
        if not self.ready: raise RuntimeError("No model is ready")


def test_model_api_sorting_load_validation_and_unready_decide():
    items = [Variant("mini", "Remek/basal-2.0-mini", "basal-2.0", "mini", "bf16", "bf16", None, 100, False),
             Variant("large", "Remek/basal-2.0-4.5B-FP8", "basal-2.0", "4.5B", "FP8", "FP8", None, 100, False)]
    hw = Hardware("cpu", "cpu", 1024**4, None, None, frozenset())
    runner = FakeRunner()
    app = create_app(catalog_fn=lambda **kwargs: (items, "hub"), hardware_fn=lambda: hw, runner=runner)
    with TestClient(app) as client:
        models = client.get("/api/models").json()
        assert models["source"] == "hub"
        assert [m["id"] for m in models["models"]] == ["mini", "large"]
        assert set(("recommended", "best_quality", "models")) <= models.keys()
        assert client.post("/api/load", json={"id": "large"}).status_code == 400
        assert client.post("/api/decide", json={"state": "x", "questions": {}}).status_code == 409


def test_directory_change_rejected_while_runner_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    item = Variant("mini", "Remek/basal-2.0-mini", "basal-2.0", "mini", "bf16", "bf16", None, 100, False)
    runner = FakeRunner(ready=True)
    runner.model = item.id
    app = create_app(catalog_fn=lambda **kwargs: ([item], "hub"), runner=runner)
    with TestClient(app) as client:
        response = client.put("/api/settings", json={"models_dir": str(tmp_path / "models")})
    assert response.status_code == 409
    assert response.json()["detail"] == "unload the model before changing the models directory"


def test_delete_rejects_loaded_and_unknown_models(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    item = Variant("mini", "Remek/basal-2.0-mini", "basal-2.0", "mini", "bf16", "bf16", None, 100, False)
    runner = FakeRunner(ready=True)
    runner.model = item.id
    app = create_app(catalog_fn=lambda **kwargs: ([item], "hub"), runner=runner)
    with TestClient(app) as client:
        loaded = client.post("/api/models/delete", json={"id": item.id})
        unknown = client.post("/api/models/delete", json={"id": "missing"})
    assert loaded.status_code == 409
    assert loaded.json()["detail"] == "unload the model first"
    assert unknown.status_code == 400
