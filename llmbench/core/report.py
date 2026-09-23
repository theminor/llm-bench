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

# Human-readable explanation of each column, in the order they appear.
GLOSSARY: list[tuple[str, str]] = [
    ("Prefill t/s", "How fast the engine reads/encodes the input prompt. Higher is better. "
                    "This is what makes the first token wait on a long chat."),
    ("Decode t/s", "How fast tokens are produced once generation starts — the \"typing speed\" "
                   "you actually watch. Higher is better."),
    ("TTFT ms", "Time to first token. Lower is better. Grows with prompt length / slow prefill."),
    ("E2E ms", "Total time for the whole request (prefill + every generated token). Lower is "
               "better. The best single number for \"how long did this request take\"."),
    ("ITL p50 ms", "Median gap between consecutive tokens (inter-token latency). Lower is better. "
                   "This is the typical \"time between letters\"."),
    ("ITL p90/p99 ms", "The 90th/99th-percentile token gap — i.e. the worst cases. Lower is better. "
                       "A big jump from p50 to p99 (or a large ±) means spiky output: the occasional "
                       "stutter you feel while reading."),
    ("± (std dev)", "Run-to-run spread across the repetitions. Small ± = stable, trust the number. "
                    "Large ± = noisy; the mean is less reliable and more reps would help."),
]


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
        eng = db.get_engine(spec.get("engine", ""))
        if eng and eng.get("docs"):
            out.append(f"- **Engine argument docs:** <{eng['docs']}>  ")
        # Model: prefer the (possibly multi) models list, fall back to legacy field.
        model_list = spec.get("models") or ([spec["model"]] if spec.get("model") else [])
        model_short = [m.rstrip("/").rsplit("/", 1)[-1] for m in model_list]
        out.append(f"- **Model:** `{', '.join(model_short) or '(none)'}`  ")
        if len(model_list) > 1:
            out.append(f"  comparing {len(model_list)} models: " + ", ".join(f"`{m}`" for m in model_list) + "  ")
        dims = spec.get("dimensions", [])
        if dims:
            dim_str = ", ".join(
                f"`{d['name']}`" + (" (env)" if d.get("type") == "env" else "") for d in dims
            )
            out.append(f"- **Dimensions:** {dim_str}  ")
        base_env = spec.get("base_env") or {}
        if base_env:
            out.append(f"- **Base env:** " + ", ".join(f"`{k}={v}`" for k, v in base_env.items()) + "  ")
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
        recs = recommendations(agg)
        if recs:
            out.append("")
            out.extend(recommendation_markdown(recs))
        out.append("")
    out.extend(glossary_markdown())
    return "\n".join(out)


def _metric_source_note(rows: list[dict]) -> str:
    srcs = {r.get("pp_tps_source", "") for r in rows} | {r.get("tg_tps_source", "") for r in rows}
    if "server" in srcs and "client" not in srcs:
        return "_Prefill/decode t/s are server-reported._"
    if "server" in srcs:
        return "_Prefill/decode t/s are client-measured; server-reported values available in JSON/CSV export._"
    return "_Prefill/decode t/s are client-measured (HTTP-level)._".rstrip("_") + "_"


def _best_by(rows: list[dict], key: str, mode: str) -> dict | None:
    """Row with the best (max/min) mean for a metric; None if no data."""
    cands = [r for r in rows if r.get(f"{key}_mean") is not None]
    if not cands:
        return None
    return (max(cands, key=lambda r: r[f"{key}_mean"]) if mode == "max"
            else min(cands, key=lambda r: r[f"{key}_mean"]))


def _cv(r: dict, key: str) -> float | None:
    """Coefficient of variation (std/mean) — a spread that's comparable across metrics."""
    m, s = r.get(f"{key}_mean"), r.get(f"{key}_std")
    if not m or s is None:
        return None
    return s / m


def _fnum(v: float | None, d: int = 1) -> str:
    return f"{v:.{d}f}" if v is not None else "?"


def recommend_for_workload(rows: list[dict], wl: str) -> dict | None:
    """Per-workload 'what's best and what to watch out for'."""
    wr = [r for r in rows if r["workload"] == wl]
    if not wr:
        return None
    dec = _best_by(wr, "tg_tps", "max")
    pre = _best_by(wr, "pp_tps", "max")
    e2e = _best_by(wr, "e2e_ms", "min")
    tail = _best_by(wr, "itl_p99_ms", "min")
    main = e2e or dec
    if main is None:
        return None
    caveats: list[str] = []
    # Tail-latency caveat: the fastest-on-average isn't the most consistent.
    if e2e and tail and e2e["label"] != tail["label"] and tail.get("itl_p99_ms_mean"):
        m_p99 = e2e.get("itl_p99_ms_mean")
        t_p99 = tail.get("itl_p99_ms_mean")
        m_cv = _cv(e2e, "itl_p99_ms") or 0
        if m_p99 and (m_p99 >= 1.4 * t_p99 or m_cv > 0.15):
            m_std = e2e.get("itl_p99_ms_std")
            caveats.append(
                f"but **{e2e['label']}** has much worse tail latency (ITL p99 {_fnum(m_p99)} ms"
                + (f" ± {_fnum(m_std)}" if m_std else "")
                + f") than **{tail['label']}** ({_fnum(t_p99)} ms). If you need steady per-token "
                  f"latency, prefer **{tail['label']}**; pick **{e2e['label']}** only for maximum "
                  f"average speed when occasional lag spikes are acceptable."
            )
    # Prefill vs decode trade-off.
    if pre and dec and pre["label"] != dec["label"] and pre.get("pp_tps_mean") and dec.get("tg_tps_mean"):
        caveats.append(
            f"trade-off: **{dec['label']}** decodes fastest ({_fnum(dec['tg_tps_mean'])} t/s) while "
            f"**{pre['label']}** prefills fastest ({_fnum(pre['pp_tps_mean'])} t/s) — choose "
            f"**{pre['label']}** for long prompts, **{dec['label']}** for long generations."
        )
    unstable = sorted({r["label"] for r in wr
                       if (_cv(r, "e2e_ms") or 0) > 0.15 or (_cv(r, "itl_p99_ms") or 0) > 0.15})
    return {
        "workload": wl,
        "main": main["label"],
        "main_by": "total time (E2E)" if e2e else "decode t/s",
        "main_value": (e2e.get("e2e_ms_mean") if e2e else dec.get("tg_tps_mean")),
        "main_unit": "ms" if e2e else "t/s",
        "winners": {
            "tg_tps": dec["label"] if dec else None,
            "pp_tps": pre["label"] if pre else None,
            "e2e_ms": e2e["label"] if e2e else None,
            "itl_p99_ms": tail["label"] if tail else None,
        },
        "caveats": caveats,
        "unstable": unstable,
    }


def recommendations(agg: list[dict]) -> list[dict]:
    wls = list(dict.fromkeys(r["workload"] for r in agg))
    return [x for x in (recommend_for_workload(agg, w) for w in wls) if x]


def recommend(db: Database, sweep_ids: list[int]) -> list[dict]:
    return recommendations(aggregate(db, sweep_ids))


def recommendation_markdown(recs: list[dict]) -> list[str]:
    out: list[str] = []
    out.append("**Best per workload** (fastest by total request time):")
    for rec in recs:
        line = f"- {rec['workload']}: **{rec['main']}**"
        if rec.get("main_value") is not None:
            line += f" — {_fnum(rec['main_value'])} {rec['main_unit']}"
        out.append(line)
        for c in rec["caveats"]:
            out.append(f"  - {c}")
        if rec["unstable"]:
            out.append(f"  - _high run-to-run variance in: {', '.join(rec['unstable'])} — "
                       f"more repetitions would firm this up._")
    return out


def glossary_markdown() -> list[str]:
    out = ["", "_**Reading the numbers:**_", ""]
    for name, desc in GLOSSARY:
        out.append(f"- **{name}** — {desc}")
    return out


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
