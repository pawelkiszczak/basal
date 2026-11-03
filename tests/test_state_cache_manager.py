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
