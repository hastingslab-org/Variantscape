"""SQLite-backed state of the pipeline.

Every paper is fetched once and moves through the stages; each stage records
its raw results so that later runs only process new papers, and so that
reference-dependent steps (cancer mapping, variant normalization, consensus,
graph) can be recomputed cheaply from stored results without new LLM calls.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id      TEXT PRIMARY KEY,
    title         TEXT,
    abstract      TEXT,
    authors       TEXT,
    pub_date      TEXT,
    pub_year      INTEGER,
    source        TEXT NOT NULL DEFAULT 'openalex',
    status        TEXT NOT NULL DEFAULT 'fetched',   -- fetched | clean | rejected
    reject_reason TEXT,
    dedupe_key    TEXT,
    run_id        TEXT
);
CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status);
CREATE INDEX IF NOT EXISTS idx_papers_dedupe ON papers(dedupe_key);

CREATE TABLE IF NOT EXISTS stage_done (
    paper_id TEXT NOT NULL,
    stage    TEXT NOT NULL,
    done_at  TEXT,
    PRIMARY KEY (paper_id, stage)
);
CREATE INDEX IF NOT EXISTS idx_stage_done_stage ON stage_done(stage);

CREATE TABLE IF NOT EXISTS gene_hits (
    paper_id TEXT NOT NULL, gene TEXT NOT NULL, PRIMARY KEY (paper_id, gene)
);
CREATE TABLE IF NOT EXISTS cancer_terms (
    paper_id TEXT NOT NULL, term TEXT NOT NULL, PRIMARY KEY (paper_id, term)
);
CREATE TABLE IF NOT EXISTS treatment_hits (
    paper_id TEXT NOT NULL, treatment TEXT NOT NULL, PRIMARY KEY (paper_id, treatment)
);
CREATE TABLE IF NOT EXISTS variant_llm (
    paper_id TEXT PRIMARY KEY, model TEXT, prompt_id TEXT, response TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS study_design (
    paper_id TEXT PRIMARY KEY, model TEXT, label TEXT, response TEXT, created_at TEXT
);
-- Association verification (candidates asked and the raw answer; parsed at build time)
CREATE TABLE IF NOT EXISTS verify_llm (
    paper_id TEXT PRIMARY KEY, model TEXT, prompt_id TEXT, candidates_json TEXT, truncated INTEGER,
    response TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, mode TEXT, started_at TEXT, finished_at TEXT,
    from_date TEXT, to_date TEXT, stages TEXT, status TEXT, summary TEXT
);
-- Publication-date windows whose fetch completed, so interrupted runs resume
CREATE TABLE IF NOT EXISTS fetch_windows (
    run_id TEXT NOT NULL, from_date TEXT NOT NULL, to_date TEXT NOT NULL,
    returned INTEGER, new_papers INTEGER, done_at TEXT,
    PRIMARY KEY (run_id, from_date, to_date)
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ------------------------------------------------------------------ #
    # Generic helpers
    # ------------------------------------------------------------------ #
    def query_df(self, sql: str, params: Sequence = ()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn, params=params)

    def scalar(self, sql: str, params: Sequence = ()):
        row = self.conn.execute(sql, params).fetchone()
        return row[0] if row else None

    def existing_ids(self, ids: Iterable[str]) -> set[str]:
        ids = list(ids)
        found: set[str] = set()
        for i in range(0, len(ids), 900):
            chunk = ids[i:i + 900]
            q = f"SELECT paper_id FROM papers WHERE paper_id IN ({','.join('?' * len(chunk))})"
            found.update(r[0] for r in self.conn.execute(q, chunk))
        return found

    def existing_dedupe_keys(self, keys: Iterable[str]) -> set[str]:
        keys = [k for k in keys if k]
        found: set[str] = set()
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = (f"SELECT dedupe_key FROM papers WHERE status = 'clean' "
                 f"AND dedupe_key IN ({','.join('?' * len(chunk))})")
            found.update(r[0] for r in self.conn.execute(q, chunk))
        return found

    # ------------------------------------------------------------------ #
    # Papers
    # ------------------------------------------------------------------ #
    def insert_papers(self, records: Sequence[dict], run_id: str) -> int:
        rows = [
            (r["paper_id"], r.get("title"), r.get("abstract"), r.get("authors"),
             r.get("pub_date"), r.get("pub_year"), r.get("source", "openalex"),
             r.get("status", "fetched"), run_id)
            for r in records
        ]
        with self.transaction() as c:
            cur = c.executemany(
                "INSERT OR IGNORE INTO papers (paper_id, title, abstract, authors, pub_date, "
                "pub_year, source, status, run_id) VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )
        return cur.rowcount if cur.rowcount is not None else 0

    def update_clean(self, kept: pd.DataFrame, rejected: dict[str, str]) -> None:
        with self.transaction() as c:
            c.executemany(
                "UPDATE papers SET title=?, abstract=?, status='clean', dedupe_key=? WHERE paper_id=?",
                [(r.title, r.abstract, r.dedupe_key, r.paper_id) for r in kept.itertuples()],
            )
            # Keep the title of rejected papers for auditing but drop the abstract to save space
            c.executemany(
                "UPDATE papers SET status='rejected', reject_reason=?, abstract=NULL WHERE paper_id=?",
                [(reason, pid) for pid, reason in rejected.items()],
            )

    def papers_by_status(self, status: str, limit: int | None = None) -> pd.DataFrame:
        sql = "SELECT paper_id, title, abstract, authors, pub_date FROM papers WHERE status=?"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.query_df(sql, (status,))

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #
    def mark_done(self, paper_ids: Iterable[str], stage: str) -> None:
        ts = now_iso()
        with self.transaction() as c:
            c.executemany(
                "INSERT OR REPLACE INTO stage_done (paper_id, stage, done_at) VALUES (?,?,?)",
                [(pid, stage, ts) for pid in paper_ids],
            )

    def insert_pairs(self, table: str, column: str, pairs: Iterable[tuple[str, str]]) -> None:
        with self.transaction() as c:
            c.executemany(
                f"INSERT OR IGNORE INTO {table} (paper_id, {column}) VALUES (?, ?)", list(pairs)
            )

    def pairs_by_paper(self, table: str, column: str) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for pid, value in self.conn.execute(f"SELECT paper_id, {column} FROM {table}"):
            out.setdefault(pid, set()).add(value)
        return out

    def ids_done(self, stage: str) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT paper_id FROM stage_done WHERE stage=?", (stage,))}

    def texts(self, paper_ids: Sequence[str]) -> pd.DataFrame:
        frames = []
        for i in range(0, len(paper_ids), 900):
            chunk = list(paper_ids[i:i + 900])
            frames.append(self.query_df(
                f"SELECT paper_id, title, abstract FROM papers WHERE paper_id IN ({','.join('?' * len(chunk))})",
                chunk,
            ))
        if not frames:
            return pd.DataFrame(columns=["paper_id", "title", "abstract"])
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ #
    # LLM results
    # ------------------------------------------------------------------ #
    def save_variant_response(self, paper_id: str, model: str, prompt_id: str, response: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO variant_llm VALUES (?,?,?,?,?)",
            (paper_id, model, prompt_id, response, now_iso()),
        )

    def save_study_design(self, paper_id: str, model: str, label: str, response: str | None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO study_design VALUES (?,?,?,?,?)",
            (paper_id, model, label, response, now_iso()),
        )

    def save_verification(self, paper_id: str, model: str, prompt_id: str, candidates_json: str,
                          truncated: bool, response: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO verify_llm VALUES (?,?,?,?,?,?,?)",
            (paper_id, model, prompt_id, candidates_json, int(truncated), response, now_iso()),
        )

    def commit(self) -> None:
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Runs
    # ------------------------------------------------------------------ #
    def start_run(self, run_id: str, mode: str, from_date: str | None, to_date: str | None,
                  stages: Sequence[str]) -> None:
        with self.transaction() as c:
            c.execute(
                "INSERT OR REPLACE INTO runs (run_id, mode, started_at, from_date, to_date, stages, status) "
                "VALUES (?,?,?,?,?,?,?)",
                (run_id, mode, now_iso(), from_date, to_date, json.dumps(list(stages)), "running"),
            )

    def finish_run(self, run_id: str, status: str, summary: dict) -> None:
        with self.transaction() as c:
            c.execute(
                "UPDATE runs SET finished_at=?, status=?, summary=? WHERE run_id=?",
                (now_iso(), status, json.dumps(summary, default=str), run_id),
            )

    def set_run_end_date(self, run_id: str, to_date: str | None) -> None:
        with self.transaction() as c:
            c.execute("UPDATE runs SET to_date=? WHERE run_id=?", (to_date, run_id))

    def last_successful_fetch_run(self) -> dict | None:
        """Most recent completed run that fetched papers (the start point of an incremental run)."""
        row = self.conn.execute(
            "SELECT run_id, mode, from_date, to_date FROM runs "
            "WHERE status='ok' AND to_date IS NOT NULL ORDER BY to_date DESC, finished_at DESC LIMIT 1"
        ).fetchone()
        return dict(zip(("run_id", "mode", "from_date", "to_date"), row)) if row else None

    def latest_unfinished_run(self) -> dict | None:
        row = self.conn.execute(
            "SELECT run_id, mode, from_date, to_date, stages FROM runs "
            "WHERE status != 'ok' AND started_at > "
            "(SELECT COALESCE(MAX(started_at), '') FROM runs WHERE status = 'ok') "
            "ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        run = dict(zip(("run_id", "mode", "from_date", "to_date", "stages"), row))
        run["stages"] = json.loads(run["stages"]) if run["stages"] else None
        return run

    def fetched_windows(self, run_id: str) -> set[tuple[str, str]]:
        return {(f, t) for f, t in self.conn.execute(
            "SELECT from_date, to_date FROM fetch_windows WHERE run_id=?", (run_id,))}

    def mark_window_fetched(self, run_id: str, from_date: str, to_date: str, returned: int, new: int) -> None:
        with self.transaction() as c:
            c.execute("INSERT OR REPLACE INTO fetch_windows VALUES (?,?,?,?,?,?)",
                      (run_id, from_date, to_date, returned, new, now_iso()))
