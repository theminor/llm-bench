"""FastAPI application: REST API + static WebUI."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .core import report
from .core.db import Database
from .core.engines import EngineProfile, builtin_presets
from .core.runner import Runner
from .core.spec import SweepSpec, plan_variants
from .core.system import snapshot

BASE_DIR = Path(__file__).resolve().parent.parent


def create_app(data_dir: str | Path | None = None) -> tuple[FastAPI, Database, Runner]:
    data_dir = Path(data_dir or BASE_DIR / "data")
    (data_dir / "logs").mkdir(parents=True, exist_ok=True)
    db = Database(data_dir / "llm_bench.sqlite3")
    runner = Runner(db, data_dir)

    app = FastAPI(title="llm-bench", version="0.1.0")

    async def json_body(request: Request) -> dict:
        """Content-type-agnostic JSON object body (tolerates text/plain etc.)."""
        raw = await request.body()
        if not raw:
            raise HTTPException(400, "request body is required")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            raise HTTPException(400, "request body must be valid JSON")
        if not isinstance(data, dict):
            raise HTTPException(400, "request body must be a JSON object")
        return data

    @app.on_event("startup")
    def seed() -> None:
        if not db.list_engines():
            for preset in builtin_presets():
                db.upsert_engine(preset)
        for s in db.list_sweeps():
            if s["status"] == "running":
                db.set_sweep_status(s["id"], "interrupted")

    # ---- engines -----------------------------------------------------
    @app.get("/api/engines")
    def engines_list():
        return db.list_engines()

    @app.get("/api/engine-presets")
    def engine_presets():
        return builtin_presets()

    @app.post("/api/engines")
    def engines_upsert(eng: dict = Depends(json_body)):
        try:
            profile = EngineProfile.from_dict(eng)
        except (KeyError, ValueError) as e:
            raise HTTPException(400, f"invalid engine definition: {e}")
        db.upsert_engine({**eng, "name": profile.name})
        return {"ok": True}

    @app.delete("/api/engines/{name}")
    def engines_delete(name: str):
        db.delete_engine(name)
        return {"ok": True}

    # ---- sweeps ------------------------------------------------------
    @app.get("/api/sweeps")
    def sweeps_list():
        sweeps = db.list_sweeps()
        for s in sweeps:
            if s["status"] == "running":
                s["progress"] = runner.status(s["id"])
        return sweeps

    @app.post("/api/sweeps")
    def sweep_create(spec: dict = Depends(json_body)):
        try:
            parsed = SweepSpec.from_dict(spec)
        except (KeyError, ValueError, TypeError) as e:
            raise HTTPException(400, f"invalid sweep spec: {e}")
        if db.get_engine(parsed.engine) is None:
            raise HTTPException(400, f"unknown engine {parsed.engine!r}")
        if not parsed.workloads:
            raise HTTPException(400, "sweep needs at least one workload")
        eng = db.get_engine(parsed.engine)
        if eng.get("model_required", True) and not (parsed.models or [parsed.model]):
            raise HTTPException(400, f"engine {parsed.engine!r} requires at least one model")
        sid = db.create_sweep(parsed.name, spec, snapshot())
        return {"id": sid}

    @app.post("/api/sweeps/plan")
    def sweep_plan(spec: dict = Depends(json_body)):
        try:
            parsed = SweepSpec.from_dict(spec)
            variants = plan_variants(parsed)
        except (KeyError, ValueError, TypeError) as e:
            raise HTTPException(400, f"invalid sweep spec: {e}")
        # Render the full command for the first variant so the user can see
        # exactly what will be executed (catches missing/duplicated flags).
        example_command = None
        if variants:
            eng = db.get_engine(parsed.engine)
            if eng:
                prof = EngineProfile.from_dict(eng)
                example_command = " ".join(
                    [prof.executable]
                    + prof.render_args(variants[0].model, 0)
                    + variants[0].args
                )
        return {
            "n_variants": len(variants),
            "n_requests_total": len(variants) * len(parsed.workloads) * parsed.repetitions,
            "variants": [v.label for v in variants[:200]],
            "example_command": example_command,
        }

    @app.post("/api/sweeps/{sweep_id}/run")
    async def sweep_run(sweep_id: int):
        sweep = db.get_sweep(sweep_id)
        if sweep is None:
            raise HTTPException(404, "sweep not found")
        if sweep["status"] == "running":
            raise HTTPException(409, "already running")
        task = asyncio.create_task(runner.run_sweep(sweep_id))
        runner.tasks[sweep_id] = task
        return {"ok": True}

    @app.post("/api/sweeps/{sweep_id}/cancel")
    def sweep_cancel(sweep_id: int):
        runner.cancel(sweep_id)
        return {"ok": True}

    @app.get("/api/sweeps/{sweep_id}")
    def sweep_get(sweep_id: int):
        sweep = db.get_sweep(sweep_id)
        if sweep is None:
            raise HTTPException(404, "sweep not found")
        sweep["variants"] = db.list_variants(sweep_id)
        sweep["progress"] = runner.status(sweep_id)
        return sweep

    @app.delete("/api/sweeps/{sweep_id}")
    def sweep_delete(sweep_id: int):
        if db.get_sweep(sweep_id) is None:
            raise HTTPException(404, "sweep not found")
        db.delete_sweep(sweep_id)
        return {"ok": True}

    @app.get("/api/variants/{variant_id}/log")
    def variant_log(variant_id: int):
        v = db.get_variant(variant_id)
        if v is None:
            raise HTTPException(404, "variant not found")
        path = v.get("log_path") or ""
        if not path or not Path(path).exists():
            return PlainTextResponse("(no log captured)")
        return PlainTextResponse(Path(path).read_text(errors="replace")[-200_000:])

    @app.get("/api/logs")
    def logs_list():
        """All variant server logs, newest first — one place to inspect runs."""
        rows = db.list_variants_all()
        for r in rows:
            p = r.get("log_path") or ""
            size = Path(p).stat().st_size if p and Path(p).exists() else 0
            r["log_bytes"] = size
        return rows

    # ---- results & exports -------------------------------------------
    def _parse_ids(sweep_ids: str) -> list[int]:
        try:
            ids = [int(x) for x in sweep_ids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(400, "sweep_ids must be comma-separated integers")
        if not ids:
            raise HTTPException(400, "no sweep_ids given")
        return ids

    @app.get("/api/results")
    def results(sweep_ids: str):
        return report.aggregate(db, _parse_ids(sweep_ids))

    @app.get("/api/export/csv")
    def export_csv(sweep_ids: str):
        text = report.to_csv(report.aggregate(db, _parse_ids(sweep_ids)))
        return PlainTextResponse(text, media_type="text/csv",
                                 headers={"Content-Disposition": 'attachment; filename="llm-bench.csv"'})

    @app.get("/api/export/json")
    def export_json(sweep_ids: str):
        text = report.to_json(report.aggregate(db, _parse_ids(sweep_ids)))
        return PlainTextResponse(text, media_type="application/json",
                                 headers={"Content-Disposition": 'attachment; filename="llm-bench.json"'})

    @app.get("/api/export/sql")
    def export_sql(sweep_ids: str):
        text = report.to_sql(report.aggregate(db, _parse_ids(sweep_ids)))
        return PlainTextResponse(text, media_type="text/plain",
                                 headers={"Content-Disposition": 'attachment; filename="llm-bench.sql"'})

    @app.get("/api/export/markdown")
    def export_markdown(sweep_ids: str, title: str = ""):
        text = report.to_markdown(db, _parse_ids(sweep_ids), title)
        return PlainTextResponse(text, media_type="text/markdown",
                                 headers={"Content-Disposition": 'attachment; filename="llm-bench-summary.md"'})

    @app.get("/api/system/env")
    def system_env():
        return snapshot()

    @app.get("/api/system/check-engine/{name}")
    async def check_engine(name: str):
        eng = db.get_engine(name)
        if eng is None:
            raise HTTPException(404, "engine not found")
        import shutil
        exe = shutil.which(eng.get("executable", "")) or ""
        found = bool(exe)
        version = ""
        if found:
            try:
                proc = await asyncio.create_subprocess_exec(
                    eng["executable"], "--version",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                version = out.decode(errors="replace").strip()[:400]
            except Exception as e:
                version = f"(could not read version: {e})"
        return {"found": found, "path": exe, "version": version}

    # ---- static UI ---------------------------------------------------
    web_dir = BASE_DIR / "web"
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

    return app, db, runner


app, DB, RUNNER = create_app()
