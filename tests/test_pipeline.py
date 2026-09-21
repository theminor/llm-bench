"""End-to-end: run a small sweep against the fake engine and validate storage,
aggregation, and exports. Works on a GPU-less machine by design."""
import asyncio
import sys
import time
from pathlib import Path

import pytest

from llmbench.core.db import Database
from llmbench.core.report import aggregate, to_csv, to_markdown, to_sql
from llmbench.core.runner import Runner
from llmbench.core.system import snapshot

HERE = Path(__file__).parent
FAKE = str(HERE / "fake_engine.py")


@pytest.fixture()
def env(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    db.upsert_engine({
        "name": "fake",
        "executable": sys.executable,
        "args": [FAKE, "--port", "{port}"],
        "ready_path": "/health",
        "ready_timeout_s": 30,
        "model_required": False,
        "timing": "llamacpp",
    })
    return db, Runner(db, tmp_path)


def _spec(reps=2):
    return {
        "name": "e2e",
        "engine": "fake",
        "model": "",
        "base_args": "",
        "workloads": [
            {"kind": "pp", "n_prompt": 48, "n_gen": 0},
            {"kind": "tg", "n_prompt": 8, "n_gen": 24},
        ],
        "dimensions": [
            {"name": "tg", "args": "--tg-tps {v}", "values": "40,80"},
        ],
        "repetitions": reps,
        "warmup": True,
        "cooldown_s": 0,
        "startup_timeout_s": 30,
    }


def test_end_to_end(env):
    db, runner = env
    sid = db.create_sweep("e2e", _spec(), snapshot())

    async def go():
        await runner.run_sweep(sid)

    asyncio.run(go())

    sweep = db.get_sweep(sid)
    assert sweep["status"] == "done", sweep

    variants = db.list_variants(sid)
    assert len(variants) == 2
    assert all(v["status"] == "done" for v in variants)

    samples = db.list_samples([sid])
    # 2 variants x 2 workloads x 2 reps
    assert len(samples) == 8
    assert all(s["ok"] for s in samples)

    # client-side numbers are in a sane range for the fake engine
    pp = [s["pp_tps_client"] for s in samples if s["workload"] == "pp48"]
    assert pp and all(v > 100 for v in pp)

    # server-reported timings were parsed from the log (llama.cpp style)
    srv = [s["pp_tps_server"] for s in samples if s["workload"] == "pp48"]
    assert all(v and 500 < v < 1200 for v in srv), srv
    tg_srv = [s["tg_tps_server"] for s in samples if s["workload"] == "tg24"]
    assert all(v and v > 10 for v in tg_srv), tg_srv

    # tg variants differ measurably
    tg = {}
    for v in variants:
        rows = [s for s in samples if s["variant_id"] == v["id"] and s["workload"] == "tg24"]
        tg[v["label"]] = sum(s["tg_tps_server"] for s in rows) / len(rows)
    fast = [k for k in tg if "80" in k][0]
    slow = [k for k in tg if "40" in k][0]
    assert tg[fast] > tg[slow] * 1.5

    agg = aggregate(db, [sid])
    assert len(agg) == 4  # variant x workload
    md = to_markdown(db, [sid], "test summary")
    assert "# test summary" in md
    assert "Decode t/s" in md
    assert "tg=80" in md
    assert "Prefill t/s" in md
    assert "Best per workload" in md
    csv = to_csv(agg)
    assert csv.startswith("sweep_id,variant_id,label,workload,n")
    sql = to_sql(agg)
    assert "INSERT INTO llm_bench" in sql


def test_startup_failure_recorded(env, tmp_path):
    db, runner = env
    bad = _spec()
    db.upsert_engine({
        "name": "broken",
        "executable": sys.executable,
        "args": ["-c", "import sys; print('boom to stderr', file=sys.stderr); sys.exit(3)"],
        "ready_timeout_s": 5,
        "model_required": False,
        "timing": "none",
    })
    bad["engine"] = "broken"
    bad["name"] = "broken"
    sid = db.create_sweep("broken", bad, snapshot())
    asyncio.run(runner.run_sweep(sid))
    sweep = db.get_sweep(sid)
    assert sweep["status"] == "failed"
    variants = db.list_variants(sid)
    assert all(v["status"] == "startup_error" for v in variants)
    assert "boom" not in variants[0]["error"]  # error is our own message
    log_file = Path(variants[0]["log_path"])
    assert log_file.exists() and "boom" in log_file.read_text()
