"""Sweep execution: for each variant, spawn engine -> measure -> tear down."""
from __future__ import annotations

import asyncio
import json
import socket
import time
from pathlib import Path
from typing import Any

import httpx

from .db import Database
from .engines import EngineProfile, ServerProcess
from .probe import build_prompt, measure_request
from .spec import SweepSpec, plan_variants


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Runner:
    def __init__(self, db: Database, data_dir: str | Path) -> None:
        self.db = db
        self.data_dir = Path(data_dir)
        self.lock = asyncio.Lock()
        self.progress: dict[int, dict[str, Any]] = {}
        self.tasks: dict[int, asyncio.Task] = {}
        self._cancel: set[int] = set()

    def status(self, sweep_id: int) -> dict[str, Any]:
        return self.progress.get(sweep_id, {"state": "unknown"})

    def cancel(self, sweep_id: int) -> None:
        self._cancel.add(sweep_id)

    def _update(self, sweep_id: int, **kw: Any) -> None:
        p = self.progress.setdefault(sweep_id, {})
        p.update(kw)
        p["updated_at"] = time.time()

    async def run_sweep(self, sweep_id: int) -> None:
        async with self.lock:  # one sweep at a time: GPU exclusivity
            sweep = self.db.get_sweep(sweep_id)
            if sweep is None:
                return
            self._cancel.discard(sweep_id)
            spec = SweepSpec.from_dict(sweep["spec"])
            eng_dict = self.db.get_engine(spec.engine)
            if eng_dict is None:
                self.db.set_sweep_status(sweep_id, "failed")
                self._update(sweep_id, state="failed", message=f"engine {spec.engine!r} not found")
                return
            profile = EngineProfile.from_dict(eng_dict)
            if profile.model_required and not (spec.models or [spec.model]):
                self.db.set_sweep_status(sweep_id, "failed")
                self._update(sweep_id, state="failed", message="this engine requires a model path")
                return

            variants = plan_variants(spec)
            previews = [
                {
                    "label": v.label,
                    "labels": v.labels,
                    "args": v.args,
                    "command_preview": " ".join(
                        [profile.executable] + profile.render_args(v.model, spec.port or 0) + v.args
                    ),
                }
                for v in variants
            ]
            variant_ids = self.db.add_variants(sweep_id, previews)
            total = len(variants)
            self.db.set_sweep_status(sweep_id, "running")
            self._update(sweep_id, state="running", done=0, total=total, message="")

            failures = 0
            for i, (variant, variant_id) in enumerate(zip(variants, variant_ids)):
                if sweep_id in self._cancel:
                    break
                self._update(
                    sweep_id,
                    done=i,
                    total=total,
                    variant_label=variant.label,
                    workload="",
                    rep=0,
                )
                ok = await self._run_variant(sweep_id, spec, profile, variant, variant_id)
                if not ok:
                    failures += 1
                if i < total - 1 and sweep_id not in self._cancel:
                    await asyncio.sleep(spec.cooldown_s)

            if sweep_id in self._cancel:
                self.db.set_sweep_status(sweep_id, "cancelled")
                self._update(sweep_id, state="cancelled", message="cancelled")
            elif failures == total and total > 0:
                self.db.set_sweep_status(sweep_id, "failed")
                self._update(sweep_id, state="failed", message="all variants failed")
            else:
                self.db.set_sweep_status(sweep_id, "done")
                self._update(sweep_id, state="done", message="")

    async def _run_variant(
        self,
        sweep_id: int,
        spec: SweepSpec,
        profile: EngineProfile,
        variant,
        variant_id: int,
    ) -> bool:
        port = spec.port or _free_port()
        log_path = str(self.data_dir / "logs" / f"sweep{sweep_id}_variant{variant_id}.log")
        model = variant.model or "default"
        # Sweep-level env: fixed base_env plus this variant's env-dimension overrides.
        extra_env = {**(spec.base_env or {}), **(variant.env or {})}
        server = ServerProcess(
            profile, variant.args, variant.model, port, log_path,
            ready_timeout_s=spec.startup_timeout_s, extra_env=extra_env,
        )
        self.db.update_variant(
            variant_id, status="starting", started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            log_path=log_path, command=" ".join(server.command),
        )
        try:
            await server.start()
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            self.db.update_variant(variant_id, status="startup_error", error=err,
                                   finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            await server.stop()
            self._update(sweep_id, message=f"variant {variant.label}: startup failed")
            await asyncio.sleep(spec.cooldown_s)
            return False

        self.db.update_variant(variant_id, status="measuring")
        any_ok = False
        sample_errors: list[str] = []
        headers = profile.render_headers(port)
        async with httpx.AsyncClient(base_url=server.base_url) as client:
            for wl in spec.workloads:
                prompt = build_prompt(wl.n_prompt or 8, seed=hash((variant.key, wl.label)) & 0x7FFFFFFF)
                if sweep_id in self._cancel:
                    break
                samples = []
                if spec.warmup:
                    self._update(sweep_id, workload=wl.label, rep=-1, message="warmup")
                    warm = await measure_request(
                        client, server.base_url, model,
                        prompt + " [warmup]", wl.max_tokens, wl.label, -1, headers,
                    )
                    self.db.add_sample(variant_id, sweep_id, warm.__dict__ | {"workload": wl.label, "rep": -1}, is_warmup=True)
                    samples.append(warm)
                mark = len(server.lines)
                for rep in range(spec.repetitions):
                    if sweep_id in self._cancel:
                        break
                    self._update(sweep_id, workload=wl.label, rep=rep)
                    # unique suffix per rep defeats prefix caching
                    s = await measure_request(
                        client, server.base_url, model,
                        f"{prompt} [rep{rep}]", wl.max_tokens, wl.label, rep, headers,
                    )
                    if profile.timing == "llamacpp" and s.ok:
                        want_tg = wl.max_tokens > 1  # eval timing is meaningless for 1-token pp runs
                        pp, tg, mark = await _collect_log_timings(server, mark, want_tg=want_tg)
                        if want_tg:
                            s.tg_tps_server = tg[-1] if tg else s.tg_tps_server
                        else:
                            s.tg_tps_server = None
                        s.pp_tps_server = pp[-1] if pp else s.pp_tps_server
                    self.db.add_sample(variant_id, sweep_id, s.__dict__ | {"workload": wl.label, "rep": rep})
                    samples.append(s)
                    if not s.ok:
                        sample_errors.append(f"[{wl.label} rep{rep}] {s.error}")
                if any(s.ok for s in samples):
                    any_ok = True
        await server.stop()
        status = "done" if any_ok else "error"
        self.db.update_variant(
            variant_id, status=status,
            error="" if any_ok else "; ".join(sample_errors)[:2000],
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        return any_ok


async def _collect_log_timings(
    server: ServerProcess, mark: int, want_tg: bool, timeout_s: float = 2.0
) -> tuple[list[float], list[float], int]:
    """Wait briefly for llama-server timing lines emitted after `mark`."""
    deadline = time.monotonic() + timeout_s
    pp: list[float] = []
    tg: list[float] = []
    while True:
        pp, tg = server.llamacpp_timings_since(mark)
        if pp and (tg or not want_tg):
            break
        if time.monotonic() >= deadline:
            break
        await asyncio.sleep(0.1)
    return pp, tg, len(server.lines)
