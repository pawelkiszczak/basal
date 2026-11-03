"""Managing the state cache while the server runs: clear a tier (memory, this model's files, every model's), change a
tier's budget (least recently used out beyond it; the disk's "auto": half of the free space plus its own files), reset
the counters, prune the directories of models not used for days, and requests that read cached states but store none
("state_cache": "no-store"); the management routes of basal-serve (/v1/state-cache) and their flags."""
import os
import threading
import time

import pytest
import torch

from basal.engine import DiskTier, StateCache
from tests.test_state_cache_disk import Maker, files, kv3, stored


@pytest.fixture
def mk(tmp_path):
    m = Maker(tmp_path)
    yield m
    for t in m.made:
        t.close()


# -- the tiers ---------------------------------------------------------------------------------------------------------

def test_clear_memory_keeps_disk(mk):
    c, k, kv, _ = stored(mk)
    assert c.clear() == 1
    st = c.stats()
    assert (st["entries"], st["used_bytes"], st["disk_entries"]) == (0, 0, 1)
    got = c.get(k)
    assert got is not None and all(torch.equal(x, y) for p, q in zip(got, kv) for x, y in zip(p, q))
    assert c.stats()["disk_loads"] == 1


def test_clear_disk_model_scope(mk):
    """A model's clear removes every engine's files of it, also those another tier indexed; that tier's next load is a
    miss (counted as an error), never a wrong state."""
    c, k, kv, toks = stored(mk)
    mlx = mk.tier("mlx")
    mlx.save(k, kv, toks, c._blocks(toks))
    mlx.flush()
    assert len(files(c.disk)) == 2 and mlx.has(k)
    assert c.disk.clear("model") == {"files": 2, "directories": 0}
    assert files(c.disk) == [] and not c.disk.has(k)
    assert mlx.load(k) is None and not mlx.has(k) and mlx.stats()["disk_errors"] == 1
    assert c.disk.stats()["disk_bytes"] == 0
    with pytest.raises(ValueError):
        c.disk.clear("everything")


def test_clear_disk_all_models(mk):
    c, _, _, _ = stored(mk)
    other = mk.tier(signature=mk.sig.replace("tiny", "other"))
    toks = list(range(300))
    other.save(b"x" * 16, kv3(300), toks, StateCache(1, salt=b"s")._blocks(toks))
    other.flush()
    (mk.dir / "notes").mkdir()  # (not a signature directory: kept)
    assert c.disk.clear("all-models") == {"files": 1, "directories": 1}
    assert sorted(p.name for p in mk.dir.iterdir()) == sorted([c.disk.root.name, "notes"])
    assert sorted(p.name for p in c.disk.root.iterdir()) == sorted(DiskTier.SUBDIRS)


def test_clear_waits_for_the_writer(mk):
    c = mk.cache()
    t = c.disk
    entered, release, write = threading.Event(), threading.Event(), DiskTier._write

    def slow(path, items, meta):
        entered.set()
        release.wait(10)
        return write(path, items, meta)
    t._write = slow
    toks = list(range(300))
    c.put(c.key(toks), kv3(300), tokens=toks)
    entered.wait(10)
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("r", t.clear()))
    th.start()
    time.sleep(0.2)
    assert th.is_alive()  # waiting for the queued write
    release.set()
    th.join(10)
    assert out["r"]["files"] == 1 and files(t) == [] and t.stats()["disk_writes"] == 1


def test_set_budget_evicts(mk):
    c = mk.cache()
    for s in range(3):
        toks = list(range(s, s + 300))
        c.put(c.key(toks), kv3(300, s), tokens=toks)
    c.disk.flush()
    one, disk_one = c.stats()["used_bytes"] // 3, c.stats()["disk_bytes"] // 3
    c.set_budget(int(1.5 * one))
    c.disk.set_budget(int(1.5 * disk_one))
    st = c.stats()
    assert (st["entries"], st["evictions"], st["budget_bytes"]) == (1, 2, int(1.5 * one))
    assert (st["disk_entries"], st["disk_evictions"], len(files(c.disk))) == (1, 2, 1)
    assert c.get(c.key(list(range(2, 302)))) is not None  # the most recent stays


def test_auto_disk_budget(mk):
    t = mk.tier(budget="auto")
    st = os.statvfs(t.root)
    assert abs(t.budget - int(0.5 * (st.f_bavail * st.f_frsize + t.used))) < 1 << 20
    assert t.stats()["disk_budget_auto"] is True
    t.budget = 5
    assert t.budget == 5 and t.stats()["disk_budget_auto"] is False
    t.set_budget("auto")
    assert t.budget > 5 and t._fixed is None


def test_prune_by_age(mk, capsys):
    old = time.time() - 3 * 86400
    mk.dir.mkdir(parents=True)
    a, b, young = (mk.dir / n for n in ("aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb", "cccccccccccccccc"))
    for d in (a, b, young):
        (d / "0").mkdir(parents=True)
    (a / "0" / "x.safetensors").write_bytes(b"x")
    os.utime(a / "0" / "x.safetensors", (old, old))
    os.utime(b, (old, old))
    (young / "0" / "y.safetensors").write_bytes(b"y")
    t1 = time.time() - 86400
    os.utime(young / "0" / "y.safetensors", (t1, t1))
    t = mk.tier(prune_days=2)
    assert sorted(p.name for p in mk.dir.iterdir()) == sorted([t.root.name, young.name])
    assert "removed 2 model directories" in capsys.readouterr().out
    assert t.prune(2) == 0


def test_reset_counters(mk):
    c, k, _, _ = stored(mk)
    c.clear()
    c.get(k)
    c.get(b"y" * 16)
    before = c.stats()
    assert before["hits"] and before["misses"] and before["disk_loads"] and before["disk_writes"]
    c.reset_counters()
    st = c.stats()
    for name in ("hits", "misses", "evictions", "prefix_hits", "disk_loads", "disk_prefix_loads", "disk_writes",
                 "disk_write_drops", "disk_errors", "disk_evictions"):
        assert st[name] == 0, name
    for name in ("entries", "used_bytes", "disk_entries", "disk_bytes"):
        assert st[name] == before[name], name


def test_mlx_budget_rewires(tmp_path, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from tests.test_mlx import mlx_stub, tiny_llama
    tiny_llama(tmp_path)
    be = mlx_stub(tmp_path)
    be.state_cache = StateCache(1 << 30)
    be._wired_rec, be._weights_bytes = 10 << 30, 1 << 20
    calls = []
    monkeypatch.setattr(mx, "set_wired_limit", lambda n: calls.append(n))
    be.set_state_cache_budget(2 << 20)
    assert be.state_cache.budget == 2 << 20
    assert calls == [min(10 << 30, (1 << 20) + be.CACHE_BYTES + (2 << 20))]
    be._wired_rec = None
    be.set_state_cache_budget(3 << 20)
    assert be.state_cache.budget == 3 << 20 and len(calls) == 1


# -- "state_cache": "no-store" -----------------------------------------------------------------------------------------

def gemma_stub():
    from tests.test_gemma4_graph import Stub, tiny
    m = tiny("e4b", torch.float32)
    be = Stub(m)
    be.state_cache = StateCache(1 << 30)
    return m, be


STATE = [1, 5, 9, 11, 3, 4, 6, 12, 13, 14, 15, 16]
G_LIDS = [[7, 8], [7, 8]]


def test_no_store_graph():
    _, be = gemma_stub()
    toks = [STATE + [20, 21, 30 + o, 31 - o] for o in range(2)]
    a = be.run_shared([(toks, G_LIDS, len(STATE), False)])[0]
    b = be.run_shared([(toks, G_LIDS, len(STATE), False)])[0]
    st = be.state_cache.stats()
    assert (st["entries"], st["misses"], st["hits"]) == (0, 2, 0)
    c = be.run_shared([(toks, G_LIDS, len(STATE))])[0]
    assert be.state_cache.stats()["entries"] == 1
    d = be.run_shared([(toks, G_LIDS, len(STATE), False)])[0]  # a no-store request reads the cached state
    st = be.state_cache.stats()
    assert (st["entries"], st["hits"]) == (1, 1)
    assert a == b == c and all(torch.allclose(torch.tensor(x), torch.tensor(y), atol=1e-6) for x, y in zip(c, d))


def test_no_store_graph_prefix_reused_not_stored():
    from tests.test_gemma4_graph import separate
    m, be = gemma_stub()
    be.CHUNK, be.gemma.MARGIN = 5, 0
    be.state_cache.BLOCK = 4
    state = STATE + [17, 18]
    be.run_shared([([state + [20, 21, 30 + o, 31 - o] for o in range(2)], G_LIDS, len(state))])
    grown = state + [19, 20, 21, 22, 23, 24, 25]
    toks = [grown + [20, 21, 30 + o, 31 - o] for o in range(2)]
    res = be.run_shared([(toks, G_LIDS, len(grown), False)])[0]
    st = be.state_cache.stats()
    c = be.state_cache
    assert st["prefix_hits"] == 1 and st["entries"] == 1
    assert c.has(c.key(state)) and not c.has(c.key(grown))  # the old entry, not replaced
    for r, e in zip(res, separate(m, toks, G_LIDS)):
        assert torch.allclose(torch.tensor(r), e, atol=1e-6)


def test_no_store_mlx(tmp_path):
    pytest.importorskip("mlx.core")
    from tests.test_mlx import LIDS, STATE as M_STATE, TOKS, tiny_llama
    from tests.test_mlx_state_cache import cached_stub
    tiny_llama(tmp_path)
    be = cached_stub(tmp_path)
    cold = be.run_shared([(TOKS, LIDS, len(M_STATE), False)])[0]
    st = be.state_cache.stats()
    assert (st["entries"], st["misses"]) == (0, 1)
    stored_run = be.run_shared([(TOKS, LIDS, len(M_STATE))])[0]
    assert be.state_cache.stats()["entries"] == 1 and cold == stored_run


def test_state_cache_body_field():
    import asyncio

    from basal.server import RequestError, Server
    from tests.test_server import FakeServer
    body = {"state": "Paczka dotarła uszkodzona.", "questions": {"q": {"type": "noul", "instructions": "Uszkodzona?"}}}
    with pytest.raises(RequestError, match='state_cache must be "store" or "no-store"'):
        asyncio.run(FakeServer([0.6, 0.4])._run(dict(body, state_cache="maybe")))
    for value, want in (("no-store", False), ("store", True), (None, True)):
        srv, seen = FakeServer([0.6, 0.4]), []

        async def ask(branches, pol, cut=None, lane="main", store=True, srv=srv, seen=seen):
            seen.append(store)
            return await Server.ask(srv, branches, pol, cut, lane, store)
        srv.ask = ask
        asyncio.run(srv._run(body if value is None else dict(body, state_cache=value)))
        assert seen and set(seen) == {want}


def test_no_store_gdn(monkeypatch, tmp_path):
    """Qwen3.5 (GDNGraphBackend): a no-store request neither stores its computed checkpoint nor the continuation of a
    cached one; it still reads both."""
    from tests.test_recurrent import ROOT, gdn_backend, max_dp, tiny_qwen35
    tiny_qwen35(tmp_path / "tiny")
    plain = gdn_backend(monkeypatch, tmp_path / "tiny")
    be = gdn_backend(monkeypatch, tmp_path / "tiny", state_cache_gb=0.01)
    be.state_cache.BLOCK = 16
    c = be.state_cache

    def qs(root, *store):
        return ([root + [20 + q, 30, 31] for q in range(3)], [[40, 41, 42]] * 3, len(root), *store)
    a = be.run_shared([qs(ROOT, False)])
    assert c.stats()["entries"] == 0
    assert max_dp(a, be.run_shared([qs(ROOT)])) < 1e-6 and c.stats()["entries"] == 1
    longer = ROOT + [14, 15, 16]
    got = be.run_shared([qs(longer, False)])
    assert max_dp(got, plain.run_shared([qs(longer)[:2]])) < 1e-5
    assert c.stats()["prefix_hits"] == 1 and c.has(c.key(ROOT)) and not c.has(c.key(longer))
