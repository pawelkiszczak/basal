"""basal-ui: a local web app that detects the hardware, suggests a basal model (size and format), downloads it,
starts basal-serve for it and runs decisions in a playground.

  basal-ui --port 7860      then open http://127.0.0.1:7860

The model runs in a basal-serve subprocess on a free local port; this app proxies /api/decide to its
/v1/systemone, so a crashed or switched model never takes the UI down.
"""
import argparse
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import httpx
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import catalog, hardware, settings

STATIC = Path(__file__).parent / "static"
FAMILY_ORDER = {f"basal-{n}": i for i, n in enumerate(("2.0", "omni-1.1", "omni-1.0", "1.5", "1.0"))}
START_TIMEOUT_S = 1800  # --mode fast compiles for 4-17 minutes on the first start (README, Serving modes)


class Runner:
    """One model at a time: idle -> downloading -> starting -> ready, or error. Every load or stop starts a new
    generation; a background load that finds its generation outdated stops instead of touching the state."""

    def __init__(self, models_dir_fn=settings.models_dir, platform_fn=lambda: hardware.detect().platform):
        self.models_dir_fn = models_dir_fn
        self.platform_fn = platform_fn
        self.lock = threading.RLock()
        self.gen = 0
        self.process = None
        self.logs = deque(maxlen=200)
        self._reset()

    def _reset(self):
        self.state, self.model, self.mode = "idle", None, None
        self.command, self.progress, self.error, self.base_url = None, None, None, None
        self.media_requested, self.media = False, []

    def status(self):
        with self.lock:
            if self.state == "ready" and self.process and self.process.poll() is not None:
                self.state = "error"
                self.error = f"basal-serve exited ({self.process.returncode}): {self.logs[-1] if self.logs else ''}"
            return {"state": self.state, "model": self.model, "mode": self.mode, "command": self.command,
                    "progress": self.progress, "error": self.error, "logs": list(self.logs),
                    "media_requested": self.media_requested, "media": self.media}

    def load(self, assessment, media=False):
        with self.lock:
            if self.state in ("downloading", "starting"):
                raise RuntimeError("a model is already loading; unload it first")
            self._stop_locked()
            self.state, self.model = "downloading", assessment.variant.id
            self.mode = "eager" if media else assessment.mode
            self.media_requested, self.media = media, []
            self.logs.clear()
            gen = self.gen
        threading.Thread(target=self._start, args=(assessment, gen, media), daemon=True).start()
        return self.status()

    def _update(self, gen, **fields):
        """Set fields if gen is still the current load; False when it was unloaded or replaced meanwhile."""
        with self.lock:
            if gen != self.gen:
                return False
            for k, v in fields.items():
                setattr(self, k, v)
            return True

    def _download(self, v, gen):
        from huggingface_hub import snapshot_download
        bar = _byte_counter()
        result = {}
        worker = threading.Thread(target=lambda: result.update(path=_capture(
            lambda: snapshot_download(v.repo, cache_dir=self.models_dir_fn(), tqdm_class=bar,
                                     **catalog.download_kwargs(v)))), daemon=True)
        worker.start()
        while worker.is_alive():
            done = bar.done
            if not self._update(gen, progress={"done_bytes": done, "total_bytes": max(v.weights_bytes or 0, done)}):
                return None  # unloaded while downloading: the download finishes in the background, into the cache
            worker.join(0.5)
        path = result["path"]
        if isinstance(path, Exception):
            raise path
        return Path(path)

    def _start(self, assessment, gen, media=False):
        v = assessment.variant
        try:
            path = catalog.is_downloaded(v, self.models_dir_fn()) or self._download(v, gen)
            if path is None:
                return
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            command = [sys.executable, "-m", "basal.server",
                       *catalog.serve_args(v, path, assessment.mode, self.platform_fn() if media else None),
                       "--host", "127.0.0.1", "--port", str(port)]
            with self.lock:
                if gen != self.gen:
                    return
                self.state, self.progress, self.command = "starting", None, command
                self.base_url = f"http://127.0.0.1:{port}"
                self.process = proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                       text=True, bufsize=1)
            threading.Thread(target=self._read_logs, args=(proc,), daemon=True).start()
            deadline = time.monotonic() + START_TIMEOUT_S
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    time.sleep(0.2)  # let the log reader drain the last lines
                    raise RuntimeError(f"basal-serve exited ({proc.returncode}): "
                                       f"{self.logs[-1] if self.logs else 'no output'}")
                try:
                    if httpx.get(self.base_url + "/health", timeout=1).status_code == 200:
                        media_kinds = []
                        if media:
                            try:
                                response = httpx.get(self.base_url + "/v1/models", timeout=1)
                                response.raise_for_status()
                                models = response.json().get("models", [])
                                media_kinds = models[0].get("media", []) if models else []
                            except Exception:
                                pass
                        self._update(gen, state="ready", media=media_kinds)
                        return
                except httpx.HTTPError:
                    pass
                if gen != self.gen:
                    return
                time.sleep(0.5)
            raise RuntimeError(f"basal-serve did not become healthy within {START_TIMEOUT_S} s")
        except Exception as exc:  # any failure of this load is shown in the UI
            self._update(gen, state="error", error=str(exc))

    def _read_logs(self, proc):
        for line in proc.stdout:
            with self.lock:
                if proc is self.process:
                    self.logs.append(line.rstrip())

    def unload(self):
        with self.lock:
            self._stop_locked()
            return self.status()

    def _stop_locked(self):
        self.gen += 1
        proc, self.process = self.process, None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self._reset()

    def decide(self, body):
        with self.lock:
            if self.state != "ready":
                raise RuntimeError("no model is ready; load one first")
            url = self.base_url + "/v1/systemone"
        return httpx.post(url, json=body, timeout=120)


def _capture(fn):
    try:
        return fn()
    except Exception as exc:  # re-raised in the waiting thread
        return exc


class _Sink:
    def write(self, _text):
        pass

    def flush(self):
        pass


def _byte_counter():
    """A tqdm class for snapshot_download that prints nothing and keeps, as the class attribute `done`, the larger of
    its two aggregated byte bars: bytes received ("Downloading bytes", which Xet reports while chunks arrive) and
    bytes written ("Reconstructing", fed by every file, Xet or plain HTTP)."""
    from tqdm.auto import tqdm

    class Counter(tqdm):
        done = 0

        def __init__(self, *args, **kwargs):
            kwargs["file"] = _Sink()
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            out = super().update(n)
            if self.unit == "B" and str(self.desc).startswith(("Reconstructing", "Downloading bytes")):
                Counter.done = max(Counter.done, int(self.n))
            return out

    return Counter


def create_app(catalog_fn=catalog.list_variants, hardware_fn=hardware.detect, runner=None,
              settings_mod=settings) -> FastAPI:
    runner = runner or Runner(models_dir_fn=settings_mod.models_dir, platform_fn=lambda: hw().platform)
    hw_cache = []

    def hw():
        if not hw_cache:
            hw_cache.append(hardware_fn())
        return hw_cache[0]

    @asynccontextmanager
    async def lifespan(_app):
        yield
        runner.unload()

    app = FastAPI(title="basal-ui", lifespan=lifespan)
    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/hardware")
    def get_hardware():
        return hw().to_dict()

    def settings_response():
        from huggingface_hub.constants import HF_HUB_CACHE
        models = Path(settings_mod.models_dir()).resolve()
        default = Path(HF_HUB_CACHE).resolve()
        return {"models_dir": str(models), "default_models_dir": str(default), "is_default": models == default}

    @app.get("/api/settings")
    def get_settings():
        return settings_response()

    @app.put("/api/settings")
    def put_settings(body: dict = Body(...)):
        if runner.status()["state"] != "idle":
            raise HTTPException(409, "unload the model before changing the models directory")
        try:
            settings_mod.set_models_dir(body.get("models_dir"))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return settings_response()

    @app.get("/api/models")
    def get_models(refresh: bool = False):
        cache_dir = settings_mod.models_dir()
        variants, source = catalog_fn(refresh=refresh, cache_dir=cache_dir)
        assessed = [catalog.assess(v, hw()) for v in variants]
        assessed.sort(key=lambda a: (FAMILY_ORDER.get(a.variant.family, 99),
                                     catalog.TIER_ORDER.get(a.variant.tier, 99), a.variant.port, a.variant.label))
        fields = ("id", "repo", "family", "tier", "port", "label", "gguf_file", "weights_bytes", "experimental")
        on_disk = {a.variant.id: catalog.downloaded_bytes(a.variant, cache_dir) for a in assessed}
        return {"source": source, "suggestions": catalog.suggest(assessed, hw()),
                "models_dir": str(Path(cache_dir).resolve()), "models": [
            {**{k: getattr(a.variant, k) for k in fields}, "runnable": a.runnable, "reason": a.reason,
             "mode": a.mode, "est_memory_bytes": a.est_memory_bytes, "fit": a.fit, "fit_now": a.fit_now,
             "media": list(a.variant.media), "media_mode": a.media_mode, "media_reason": a.media_reason,
             "downloaded": on_disk[a.variant.id] is not None, "downloaded_bytes": on_disk[a.variant.id]}
            for a in assessed]}

    @app.post("/api/models/delete")
    def delete_model(body: dict = Body(...)):
        model_id = body.get("id")
        variants, _ = catalog_fn(cache_dir=settings_mod.models_dir())
        variant = next((v for v in variants if v.id == model_id), None)
        if variant is None:
            raise HTTPException(400, f"unknown model {model_id!r}")
        status = runner.status()
        if status["state"] != "idle" and status["model"] == model_id:
            raise HTTPException(409, "unload the model first")
        return {"freed_bytes": catalog.delete(variant, settings_mod.models_dir())}


    @app.post("/api/load", status_code=202)
    def load(body: dict = Body(...)):
        variants, _ = catalog_fn(cache_dir=settings_mod.models_dir())
        variant = next((v for v in variants if v.id == body.get("id")), None)
        if variant is None:
            raise HTTPException(400, f"unknown model {body.get('id')!r}")
        media = body.get("media", False)
        if not isinstance(media, bool):
            raise HTTPException(400, "media must be a boolean")
        assessment = catalog.assess(variant, hw())
        if not assessment.runnable and not media:
            raise HTTPException(400, assessment.reason)
        if media and assessment.media_mode is None:
            raise HTTPException(400, assessment.media_reason)
        try:
            return runner.load(assessment, media=media)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/unload")
    def unload():
        return runner.unload()

    @app.get("/api/status")
    def status():
        return runner.status()

    @app.get("/api/examples")
    def get_examples():
        from basal.viz import examples
        return examples()

    @app.post("/api/decide")
    def decide(body: dict = Body(...)):  # sync: runs in the threadpool, a slow decision does not block the UI
        try:
            r = runner.decide(body)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"basal-serve: {exc}") from exc
        return Response(r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))

    return app


def main():
    ap = argparse.ArgumentParser(description="basal browser UI: hardware check, model choice, playground")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    import uvicorn
    print(f"basal-ui -> http://{a.host}:{a.port}", flush=True)
    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
