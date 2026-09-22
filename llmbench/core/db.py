"""SQLite persistence: engines, models, sweeps, variants, samples."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS engines (
    name TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sweeps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    spec TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    env TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS variants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sweep_id INTEGER NOT NULL REFERENCES sweeps(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    label TEXT NOT NULL,
    labels TEXT NOT NULL DEFAULT '{}',
    args TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT NOT NULL DEFAULT '',
    log_path TEXT NOT NULL DEFAULT '',
    command TEXT NOT NULL DEFAULT '',
    started_at TEXT,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    variant_id INTEGER NOT NULL REFERENCES variants(id) ON DELETE CASCADE,
    sweep_id INTEGER NOT NULL,
    workload TEXT NOT NULL,
    rep INTEGER NOT NULL,
    is_warmup INTEGER NOT NULL DEFAULT 0,
    ok INTEGER NOT NULL DEFAULT 1,
    error TEXT NOT NULL DEFAULT '',
    ttft_ms REAL, e2e_ms REAL,
    pp_tps_client REAL, tg_tps_client REAL,
    pp_tps_server REAL, tg_tps_server REAL,
    itl_mean_ms REAL, itl_p50_ms REAL, itl_p90_ms REAL, itl_p99_ms REAL,
    prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0,
    raw TEXT NOT NULL DEFAULT '{}'
);
"""


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock, self._conn:
            return self._conn.execute(sql, params)

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ---- engines -----------------------------------------------------
    def list_engines(self) -> list[dict]:
        return [json.loads(r["data"]) for r in self._query("SELECT data FROM engines ORDER BY name")]

    def get_engine(self, name: str) -> dict | None:
        rows = self._query("SELECT data FROM engines WHERE name = ?", (name,))
        return json.loads(rows[0]["data"]) if rows else None

    def upsert_engine(self, eng: dict) -> None:
        self._exec(
            "INSERT INTO engines(name, data) VALUES(?, ?) "
            "ON CONFLICT(name) DO UPDATE SET data = excluded.data",
            (eng["name"], json.dumps(eng)),
        )

    def delete_engine(self, name: str) -> None:
        self._exec("DELETE FROM engines WHERE name = ?", (name,))

    # ---- sweeps ------------------------------------------------------
    def create_sweep(self, name: str, spec: dict, env: dict) -> int:
        cur = self._exec(
            "INSERT INTO sweeps(name, spec, status, env, created_at) VALUES(?, ?, 'pending', ?, ?)",
            (name, json.dumps(spec), json.dumps(env), _now()),
        )
        return int(cur.lastrowid)

    def set_sweep_status(self, sweep_id: int, status: str) -> None:
        now = _now()
        if status == "running":
            self._exec(
                "UPDATE sweeps SET status = ?, started_at = ? WHERE id = ?",
                (status, now, sweep_id),
            )
        elif status in ("done", "failed", "cancelled", "interrupted"):
            self._exec(
                "UPDATE sweeps SET status = ?, finished_at = ? WHERE id = ?",
                (status, now, sweep_id),
            )
        else:
            self._exec("UPDATE sweeps SET status = ? WHERE id = ?", (status, sweep_id))

    def list_sweeps(self) -> list[dict]:
        rows = self._query("SELECT * FROM sweeps ORDER BY id DESC")
        for r in rows:
            r["spec"] = json.loads(r["spec"])
            r["env"] = json.loads(r["env"])
        return rows

    def get_sweep(self, sweep_id: int) -> dict | None:
        rows = self._query("SELECT * FROM sweeps WHERE id = ?", (sweep_id,))
        if not rows:
            return None
        r = rows[0]
        r["spec"] = json.loads(r["spec"])
        r["env"] = json.loads(r["env"])
        return r

    def delete_sweep(self, sweep_id: int) -> None:
        self._exec("DELETE FROM samples WHERE sweep_id = ?", (sweep_id,))
        self._exec("DELETE FROM variants WHERE sweep_id = ?", (sweep_id,))
        self._exec("DELETE FROM sweeps WHERE id = ?", (sweep_id,))

    # ---- variants ----------------------------------------------------
    def add_variants(self, sweep_id: int, variants: list[dict]) -> list[int]:
        ids = []
        with self._lock, self._conn:
            for i, v in enumerate(variants):
                cur = self._conn.execute(
                    "INSERT INTO variants(sweep_id, idx, label, labels, args, command) "
                    "VALUES(?, ?, ?, ?, ?, ?)",
                    (
                        sweep_id,
                        i,
                        v["label"],
                        json.dumps(v.get("labels", {})),
                        " ".join(v.get("args", [])),
                        " ".join(v.get("command_preview", v.get("args", []))),
                    ),
                )
                ids.append(int(cur.lastrowid))
        return ids

    def update_variant(self, variant_id: int, **fields: Any) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        self._exec(f"UPDATE variants SET {sets} WHERE id = ?", tuple(fields.values()) + (variant_id,))

    def list_variants(self, sweep_id: int) -> list[dict]:
        rows = self._query(
            "SELECT v.*, "
            "(SELECT COUNT(*) FROM samples s WHERE s.variant_id = v.id AND s.is_warmup = 0) AS n_samples, "
            "(SELECT MIN(s.tg_tps_client) FROM samples s WHERE s.variant_id = v.id AND s.is_warmup = 0 AND s.tg_tps_client IS NOT NULL) AS min_tg "
            "FROM variants v WHERE sweep_id = ? ORDER BY idx",
            (sweep_id,),
        )
        return rows

    def get_variant(self, variant_id: int) -> dict | None:
        rows = self._query("SELECT * FROM variants WHERE id = ?", (variant_id,))
        return rows[0] if rows else None

    def list_variants_all(self) -> list[dict]:
        """Variants across all sweeps, newest first, with sweep name."""
        return self._query(
            "SELECT v.id, v.idx, v.label, v.status, v.error, v.log_path, v.started_at, v.finished_at, "
            "v.sweep_id, s.name AS sweep_name, s.status AS sweep_status "
            "FROM variants v JOIN sweeps s ON s.id = v.sweep_id "
            "ORDER BY v.id DESC",
        )

    # ---- samples -----------------------------------------------------
    def add_sample(self, variant_id: int, sweep_id: int, sample: dict, is_warmup: bool = False) -> None:
        self._exec(
            "INSERT INTO samples(variant_id, sweep_id, workload, rep, is_warmup, ok, error, "
            "ttft_ms, e2e_ms, pp_tps_client, tg_tps_client, pp_tps_server, tg_tps_server, "
            "itl_mean_ms, itl_p50_ms, itl_p90_ms, itl_p99_ms, prompt_tokens, completion_tokens, raw) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                variant_id,
                sweep_id,
                sample.get("workload", ""),
                int(sample.get("rep", 0)),
                1 if is_warmup else 0,
                1 if sample.get("ok", True) else 0,
                str(sample.get("error", "")),
                sample.get("ttft_ms"),
                sample.get("e2e_ms"),
                sample.get("pp_tps_client"),
                sample.get("tg_tps_client"),
                sample.get("pp_tps_server"),
                sample.get("tg_tps_server"),
                sample.get("itl_mean_ms"),
                sample.get("itl_p50_ms"),
                sample.get("itl_p90_ms"),
                sample.get("itl_p99_ms"),
                int(sample.get("prompt_tokens", 0)),
                int(sample.get("completion_tokens", 0)),
                json.dumps(sample.get("raw", {})),
            ),
        )

    def list_samples(self, sweep_ids: list[int], include_warmup: bool = False) -> list[dict]:
        if not sweep_ids:
            return []
        qmarks = ",".join("?" * len(sweep_ids))
        where = f"sweep_id IN ({qmarks})" + ("" if include_warmup else " AND is_warmup = 0")
        return self._query(f"SELECT * FROM samples WHERE {where} ORDER BY id", tuple(sweep_ids))


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
