"""Result aggregation and exports: Markdown, CSV, JSON, SQL.

Aggregation matches llama-bench: per (variant, workload, metric) the mean and
sample standard deviation across repetitions.
"""
from __future__ import annotations

import csv
import io
import json
import time
from typing import Any

from .db import Database
from .metrics import summarize

# metric key -> (column in samples, nicer header)
METRICS: list[tuple[str, str]] = [
    ("pp_tps", "Prefill t/s"),
    ("tg_tps", "Decode t/s"),
    ("ttft_ms", "TTFT ms"),
    ("e2e_ms", "E2E ms"),
    ("itl_p50_ms", "ITL p50 ms"),
    ("itl_p90_ms", "ITL p90 ms"),
    ("itl_p99_ms", "ITL p99 ms"),
]

SERVER_METRICS = {"pp_tps": "pp_tps_server", "tg_tps": "tg_tps_server"}
CLIENT_METRICS = {"pp_tps": "pp_tps_client", "tg_tps": "tg_tps_client"}


def aggregate(db: Database, sweep_ids: list[int]) -> list[dict[str, Any]]:
    """Rows of {sweep_id, variant_id, label, workload, <metric>_mean/std/n}."""
    samples = db.list_samples(sweep_ids)
    variants: dict[int, dict] = {}
    for sid in sweep_ids:
        for v in db.list_variants(sid):
            variants[v["id"]] = v
    groups: dict[tuple[int, str], list[dict]] = {}
    for s in samples:
        groups.setdefault((s["variant_id"], s["workload"]), []).append(s)

    rows: list[dict[str, Any]] = []
    for (vid, workload), ss in sorted(groups.items()):
        v = variants.get(vid, {})
        row: dict[str, Any] = {
            "sweep_id": ss[0]["sweep_id"],
            "variant_id": vid,
            "label": v.get("label", str(vid)),
            "workload": workload,
            "n": sum(1 for s in ss if s["ok"]),
            "errors": "; ".join(dict.fromkeys(s["error"] for s in ss if not s["ok"]))[:500],
        }
        for key, _hdr in METRICS:
            if key in ("pp_tps", "tg_tps"):
                vals_client = [s[CLIENT_METRICS[key]] for s in ss if s.get(CLIENT_METRICS[key])]
                vals_server = [s[SERVER_METRICS[key]] for s in ss if s.get(SERVER_METRICS[key])]
                c, sv = summarize(vals_client), summarize(vals_server)
                row[f"{key}_mean"] = c.mean if c.n else (sv.mean if sv.n else None)
                row[f"{key}_std"] = c.std if c.n else (sv.std if sv.n else None)
                row[f"{key}_source"] = "client" if c.n else ("server" if sv.n else "")
                row[f"{key}_client_mean"] = c.mean if c.n else None
                row[f"{key}_server_mean"] = sv.mean if sv.n else None
            else:
                vals = [s[key] for s in ss if s.get(key) is not None]
                su = summarize(vals)
                row[f"{key}_mean"] = su.mean if su.n else None
                row[f"{key}_std"] = su.std if su.n else None
        rows.append(row)
    return rows


def _fmt(value: float | None, std: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f} ± {std:.{digits}f}" if std else f"{value:.{digits}f}"


def to_markdown(db: Database, sweep_ids: list[int], title: str = "") -> str:
    """A human-readable summary of one or more sweeps."""
    sweeps = [db.get_sweep(sid) for sid in sweep_ids]
    sweeps = [s for s in sweeps if s]
    agg = aggregate(db, [s["id"] for s in sweeps])
    out: list[str] = []
    out.append(f"# {title or 'llm-bench results summary'}")
    out.append("")
    out.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} by llm-bench_")
    out.append("")
    for sweep in sweeps:
        spec = sweep["spec"]
        env = sweep.get("env", {})
        out.append(f"## Sweep {sweep['id']}: {sweep['name']}")
        out.append("")
        out.append(f"- **Status:** {sweep['status']}  ")
        out.append(f"- **Engine:** `{spec.get('engine', '')}`  ")
        # Model: prefer the (possibly multi) models list, fall back to legacy field.
        model_list = spec.get("models") or ([spec["model"]] if spec.get("model") else [])
        model_short = [m.rstrip("/").rsplit("/", 1)[-1] for m in model_list]
        out.append(f"- **Model:** `{', '.join(model_short) or '(none)'}`  ")
        if len(model_list) > 1:
            out.append(f"  comparing {len(model_list)} models: " + ", ".join(f"`{m}`" for m in model_list) + "  ")
        dims = spec.get("dimensions", [])
        if dims:
            out.append(f"- **Dimensions:** " + ", ".join(f"`{d['name']}`" for d in dims) + "  ")
        wl_labels = []
        for w in spec.get("workloads", []):
            kind, np_, ng = w.get("kind"), int(w.get("n_prompt", 0)), int(w.get("n_gen", 0))
            wl_labels.append("pp%d" % np_ if kind == "pp" else "tg%d" % ng if kind == "tg" else f"pg{np_},{ng}")
        out.append(
            f"- **Repetitions:** {spec.get('repetitions', 3)} | "
            f"**Workloads:** {', '.join(wl_labels) or '(none)'}"
        )
        out.append(
            f"- **Host:** {env.get('hostname', '?')} ({env.get('os', '?')}) | "
            f"CPU: {env.get('cpu', '?')} x{env.get('cpu_cores', '?')} | "
            f"RAM: {env.get('mem', '?')} | GPU: {env.get('gpus', 'none detected')}"
        )
        out.append("")
        rows = [r for r in agg if r["sweep_id"] == sweep["id"]]
        if not rows:
            out.append("_No samples._")
            out.append("")
            continue
        header = "| Variant | Test | n |"
        sep = "|---|---|---|"
        for _key, hdr in METRICS:
            header += f" {hdr} |"
            sep += "---|"
        out.append(header)
        out.append(sep)
        for r in rows:
            line = f"| {r['label']} | {r['workload']} | {r['n']} |"
            for key, _hdr in METRICS:
                line += f" {_fmt(r.get(f'{key}_mean'), r.get(f'{key}_std'))} |"
            out.append(line)
        note = _metric_source_note(rows)
        if note:
            out.append("")
            out.append(note)
        winners = _winners(rows)
        if winners:
            out.append("")
            out.append("**Best per workload** (by decode t/s, then prefill t/s):")
            for wl, best in winners.items():
                out.append(f"- {wl}: **{best}**")
        out.append("")
    return "\n".join(out)


def _metric_source_note(rows: list[dict]) -> str:
    srcs = {r.get("pp_tps_source", "") for r in rows} | {r.get("tg_tps_source", "") for r in rows}
    if "server" in srcs and "client" not in srcs:
        return "_Prefill/decode t/s are server-reported._"
    if "server" in srcs:
        return "_Prefill/decode t/s are client-measured; server-reported values available in JSON/CSV export._"
    return "_Prefill/decode t/s are client-measured (HTTP-level)._".rstrip("_") + "_"


def _winners(rows: list[dict]) -> dict[str, str]:
    best: dict[str, tuple[float, float, str]] = {}
    for r in rows:
        wl = r["workload"]
        score = (
            r.get("tg_tps_mean") or 0.0,
            r.get("pp_tps_mean") or 0.0,
            r["label"],
        )
        if wl not in best or (score[0], score[1]) > (best[wl][0], best[wl][1]):
            best[wl] = score
    return {wl: b[2] for wl, b in best.items() if b[0] or b[1]}


def to_csv(agg: list[dict]) -> str:
    buf = io.StringIO()
    if agg:
        cols = ["sweep_id", "variant_id", "label", "workload", "n"]
        for key, _ in METRICS:
            cols += [f"{key}_mean", f"{key}_std", f"{key}_source"]
        cols += ["errors"]
        w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(agg)
    return buf.getvalue()


def to_json(agg: list[dict]) -> str:
    return json.dumps(agg, indent=2)


SQL_SCHEMA = """CREATE TABLE IF NOT EXISTS llm_bench (
    sweep_id INTEGER, variant_id INTEGER, variant TEXT, workload TEXT, n INTEGER,
    pp_tps REAL, pp_tps_std REAL, tg_tps REAL, tg_tps_std REAL,
    ttft_ms REAL, ttft_ms_std REAL, e2e_ms REAL, e2e_ms_std REAL,
    itl_p50_ms REAL, itl_p90_ms REAL, itl_p99_ms REAL, errors TEXT
);"""


def to_sql(agg: list[dict]) -> str:
    out = [SQL_SCHEMA]
    for r in agg:
        vals = {
            "sweep_id": r["sweep_id"],
            "variant_id": r["variant_id"],
            "variant": r["label"],
            "workload": r["workload"],
            "n": r["n"],
            "pp_tps": r.get("pp_tps_mean"),
            "pp_tps_std": r.get("pp_tps_std"),
            "tg_tps": r.get("tg_tps_mean"),
            "tg_tps_std": r.get("tg_tps_std"),
            "ttft_ms": r.get("ttft_ms_mean"),
            "ttft_ms_std": r.get("ttft_ms_std"),
            "e2e_ms": r.get("e2e_ms_mean"),
            "e2e_ms_std": r.get("e2e_ms_std"),
            "itl_p50_ms": r.get("itl_p50_ms_mean"),
            "itl_p90_ms": r.get("itl_p90_ms_mean"),
            "itl_p99_ms": r.get("itl_p99_ms_mean"),
            "errors": r.get("errors", ""),
        }
        items = ", ".join(
            "NULL" if v is None else ("'" + str(v).replace("'", "''") + "'" if isinstance(v, str) else str(v))
            for v in vals.values()
        )
        out.append(f"INSERT INTO llm_bench ({', '.join(vals)}) VALUES ({items});")
    return "\n".join(out)
