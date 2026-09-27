"""HTTP load test of a running basal server: sequential latency and throughput with concurrent clients.

  basal-loadtest --url http://127.0.0.1:8000/v1/systemone --questions basal/examples/questions.jsonl
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

from .bench import DEFAULT_QUESTIONS, load_questions


def body(q, early_exit=None):
    b = {"state": q["state"], "questions": {"q": {"type": "choice", "instructions": q["question"],
                                                   "criteria": {f"option_{k + 1}": o for k, o in enumerate(q["options"])}}}}
    if early_exit:
        b["early_exit"] = early_exit
    return b


async def run(a):
    qs = load_questions(a.questions, max(a.n_seq, a.n_conc) + 20)
    bodies = [body(q, a.early_exit) for q in qs]
    while len(bodies) < max(a.n_seq, a.n_conc) + 20:  # small question files are reused
        bodies += bodies
    async with httpx.AsyncClient(timeout=300) as c:
        for b in bodies[:10]:  # warm-up
            (await c.post(a.url, json=b)).raise_for_status()
        lat = []
        for b in bodies[10: 10 + a.n_seq]:
            t0 = time.perf_counter()
            r = await c.post(a.url, json=b)
            lat.append(time.perf_counter() - t0)
            r.raise_for_status()
        lat.sort()
        sem = asyncio.Semaphore(a.concurrency)

        async def one(b):
            async with sem:
                t0 = time.perf_counter()
                r = await c.post(a.url, json=b)
                return time.perf_counter() - t0, r.status_code

        t0 = time.perf_counter()
        res = await asyncio.gather(*[one(b) for b in bodies[: a.n_conc]])
        wall = time.perf_counter() - t0
    ok = sorted(x for x, s in res if s == 200)
    out = dict(url=a.url, p50_ms=statistics.median(lat) * 1000, p95_ms=lat[int(0.95 * len(lat))] * 1000,
               concurrency=a.concurrency, decisions_per_s=len(ok) / wall, errors=len(res) - len(ok))
    print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in out.items()}))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/systemone")
    ap.add_argument("--questions", nargs="+", default=DEFAULT_QUESTIONS, help="JSONL file(s) with simple items")
    ap.add_argument("--n-seq", dest="n_seq", type=int, default=200)
    ap.add_argument("--n-conc", dest="n_conc", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--early-exit", dest="early_exit", default=None)
    ap.add_argument("--out", default=None)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
