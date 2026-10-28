"""basal-loadtest --determinism: the same requests alone and under load -> identical answers, max |dp| and decision
flips (answers whose top option changed)."""
import argparse
import asyncio
import json
import sys

import httpx
import pytest

from basal import loadtest


def test_flips_and_differences(monkeypatch, tmp_path):
    calls = {"n": 0}

    def handler(request):
        body = json.loads(request.content)
        calls["n"] += 1
        p = 0.6 if calls["n"] <= 2 else (0.4 if body["state"] == "a" else 0.6)  # request "a" flips under load
        return httpx.Response(200, json={"answers": {"q": {"probabilities": {"x": p, "y": 1 - p}}}})
    real = httpx.AsyncClient
    monkeypatch.setattr(loadtest.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    bodies = [{"state": s, "questions": {}} for s in ("a", "b", "c", "d")]
    a = argparse.Namespace(determinism=2, url="http://test/v1/systemone", concurrency=2, out=str(tmp_path / "d.json"))
    asyncio.run(loadtest.determinism(a, bodies))
    d = json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))["determinism"]
    assert d["requests"] == 2 and d["identical"] == 1 and d["decision_flips"] == 1
    assert abs(d["max_abs_diff"] - 0.2) < 1e-9


def test_seed_gives_other_synthetic_states(monkeypatch):
    """--seed: a second load test with another seed asks states a prefix-caching server has not seen yet."""
    seen = []
    monkeypatch.setattr(loadtest, "run", lambda a: seen.append(a.seed))
    monkeypatch.setattr(loadtest.asyncio, "run", lambda x: None)
    for argv in (["--synthetic", "200"], ["--synthetic", "200", "--seed", "2"]):
        monkeypatch.setattr(sys, "argv", ["basal-loadtest"] + argv)
        loadtest.main()
    assert seen == [0, 2]
    from basal import synth
    assert synth.items(3, 200, 1)[0]["state"] != synth.items(3, 200, 2)[0]["state"]


@pytest.mark.parametrize("determinism", [False, True])
def test_results_do_not_log_url_credentials(monkeypatch, tmp_path, capsys, determinism):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"answers": {"q": {"probabilities": {"a": 0.6, "b": 0.4}}}})

    real = httpx.AsyncClient
    monkeypatch.setattr(loadtest.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    url = "http://user:secret@test/v1/systemone?token=private&signature=signed#fragment-secret"
    a = argparse.Namespace(url=url, out=str(tmp_path / "results.json"), concurrency=1, determinism=1,
                           n_seq=1, n_conc=1, synthetic=100, seed=0, long_frac=0,
                           early_exit=None, questions_per_request=1)
    if determinism:
        asyncio.run(loadtest.determinism(a, [{"state": "a"}, {"state": "b"}]))
    else:
        asyncio.run(loadtest.run(a))
    assert seen and all(r.url.params["token"] == "private" for r in seen)
    assert all(r.headers.get("authorization", "").startswith("Basic ") for r in seen)
    output = capsys.readouterr().out + (tmp_path / "results.json").read_text()
    assert json.loads((tmp_path / "results.json").read_text())["url"] == "http://test/v1/systemone"
    assert all(secret not in output for secret in ("user", "secret", "private", "signed", "fragment-secret"))
