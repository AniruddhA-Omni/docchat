"""SQLite catalog: documents, SQL-queryable tables (stored as Parquet), code symbols, chat threads."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from docchat.ingest.tables import sql_identifier
from docchat.schemas import TableData

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    file_name TEXT NOT NULL,
    path TEXT NOT NULL,
    format TEXT NOT NULL,
    parser TEXT,
    ocr_engine TEXT,
    page_count INTEGER,
    n_chunks INTEGER DEFAULT 0,
    n_tables INTEGER DEFAULT 0,
    size_bytes INTEGER,
    status TEXT NOT NULL,
    error TEXT,
    warnings TEXT DEFAULT '[]',
    index_key TEXT,
    added_at REAL,
    elapsed_s REAL
);
CREATE TABLE IF NOT EXISTS sql_tables (
    name TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    label TEXT,
    title TEXT,
    sheet TEXT,
    page INTEGER,
    parquet_path TEXT NOT NULL,
    columns TEXT NOT NULL,
    card TEXT,
    n_rows INTEGER
);
CREATE TABLE IF NOT EXISTS symbols (
    doc_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    kind TEXT,
    line_start INTEGER,
    line_end INTEGER
);
CREATE INDEX IF NOT EXISTS symbols_name ON symbols(symbol);
CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY,
    title TEXT,
    created_at REAL,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS messages (
    thread_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    payload TEXT,
    PRIMARY KEY (thread_id, idx)
);
"""


@dataclass
class DocRecord:
    doc_id: str
    sha256: str
    file_name: str
    path: str
    format: str
    status: str
    parser: str | None = None
    ocr_engine: str | None = None
    page_count: int | None = None
    n_chunks: int = 0
    n_tables: int = 0
    size_bytes: int | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    index_key: str | None = None
    added_at: float | None = None
    elapsed_s: float | None = None


@dataclass
class TableRecord:
    name: str
    doc_id: str
    label: str
    title: str | None
    sheet: str | None
    page: int | None
    parquet_path: str
    columns: dict[str, str]
    card: str
    n_rows: int
    file_name: str = ""


class Catalog:
    def __init__(self, db_path: Path, tables_dir: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir = tables_dir
        self._lock = threading.RLock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(_SCHEMA)
            self._db.commit()
        self.version = 0  # bumped on every change; used to invalidate caches / SQL sandboxes

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _exec(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._db.execute(sql, params)
            self._db.commit()
            return cur

    def _touch(self) -> None:
        self.version += 1

    # -- documents -------------------------------------------------------------------------------
    def upsert_document(self, rec: DocRecord) -> None:
        data = rec.__dict__ | {"warnings": json.dumps(rec.warnings)}
        cols = ", ".join(data)
        marks = ", ".join(f":{k}" for k in data)
        self._exec(f"INSERT OR REPLACE INTO documents ({cols}) VALUES ({marks})", data)
        self._touch()

    def get_document(self, doc_id: str) -> DocRecord | None:
        row = self._exec("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return self._doc(row) if row else None

    def list_documents(self) -> list[DocRecord]:
        rows = self._exec("SELECT * FROM documents ORDER BY added_at").fetchall()
        return [self._doc(r) for r in rows]

    def ready_documents(self) -> list[DocRecord]:
        return [d for d in self.list_documents() if d.status == "ready"]

    @staticmethod
    def _doc(row: sqlite3.Row) -> DocRecord:
        data = dict(row)
        data["warnings"] = json.loads(data.get("warnings") or "[]")
        return DocRecord(**data)

    def delete_document(self, doc_id: str, keep_record: bool = False) -> None:
        """Remove a document's tables and symbols (and, unless keep_record, the document row)."""
        for t in self.tables_for(doc_id):
            Path(t.parquet_path).unlink(missing_ok=True)
        tables = (
            ("sql_tables", "symbols") if keep_record else ("documents", "sql_tables", "symbols")
        )
        with self._lock:
            for table in tables:
                self._db.execute(f"DELETE FROM {table} WHERE doc_id = ?", (doc_id,))
            self._db.commit()
        self._touch()

    # -- tables ----------------------------------------------------------------------------------
    def unique_table_name(self, base: str) -> str:
        base = sql_identifier(base, fallback="t")[:48]
        existing = {r["name"] for r in self._exec("SELECT name FROM sql_tables").fetchall()}
        name, k = base, 2
        while name in existing:
            name, k = f"{base}_{k}", k + 1
        return name

    def add_table(self, doc_id: str, table: TableData, card: str) -> TableRecord:
        path = self.tables_dir / f"{table.name}.parquet"
        table.df.to_parquet(path, index=False)
        rec = TableRecord(
            name=table.name, doc_id=doc_id, label=table.label, title=table.title,
            sheet=table.sheet, page=table.page, parquet_path=str(path), columns=table.columns,
            card=card, n_rows=len(table.df),
        )  # fmt: skip
        data = rec.__dict__ | {"columns": json.dumps(rec.columns)}
        data.pop("file_name")
        cols = ", ".join(data)
        self._exec(f"INSERT OR REPLACE INTO sql_tables ({cols}) VALUES "
                   f"({', '.join(':' + k for k in data)})", data)  # fmt: skip
        self._touch()
        return rec

    def tables_for(self, doc_id: str | None = None) -> list[TableRecord]:
        sql = ("SELECT t.*, d.file_name FROM sql_tables t JOIN documents d USING (doc_id)"
               if doc_id is None else
               "SELECT t.*, d.file_name FROM sql_tables t LEFT JOIN documents d USING (doc_id) "
               "WHERE t.doc_id = ?")  # fmt: skip
        rows = self._exec(sql, () if doc_id is None else (doc_id,)).fetchall()
        out = []
        for r in rows:
            data = dict(r)
            data["columns"] = json.loads(data["columns"])
            data["file_name"] = data.get("file_name") or ""
            out.append(TableRecord(**data))
        return out

    def get_table(self, name: str) -> TableRecord | None:
        return next((t for t in self.tables_for() if t.name == name), None)

    @staticmethod
    def load_frame(rec: TableRecord) -> pd.DataFrame:
        return pd.read_parquet(rec.parquet_path)

    # -- symbols ---------------------------------------------------------------------------------
    def add_symbols(self, doc_id: str, symbols: list[dict[str, Any]]) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT INTO symbols VALUES (?, ?, ?, ?, ?)",
                [(doc_id, s["symbol"], s["kind"], s["line_start"], s["line_end"]) for s in symbols],
            )
            self._db.commit()

    def find_symbols(self, names: list[str], doc_ids: list[str] | None = None) -> list[dict]:
        if not names:
            return []
        marks = ",".join("?" * len(names))
        rows = self._exec(
            f"SELECT s.*, d.file_name FROM symbols s JOIN documents d USING (doc_id) "
            f"WHERE s.symbol IN ({marks}) OR substr(s.symbol, instr(s.symbol, '.') + 1) IN ({marks})",
            (*names, *names),
        ).fetchall()
        out = [dict(r) for r in rows]
        return [r for r in out if doc_ids is None or r["doc_id"] in doc_ids]

    def all_symbol_names(self) -> set[str]:
        rows = self._exec("SELECT DISTINCT symbol FROM symbols").fetchall()
        names = set()
        for r in rows:
            names.add(r["symbol"])
            names.add(r["symbol"].rsplit(".", 1)[-1])
        return names

    # -- threads ---------------------------------------------------------------------------------
    def touch_thread(self, thread_id: str, title: str) -> None:
        now = time.time()
        self._exec(
            "INSERT INTO threads VALUES (?, ?, ?, ?) ON CONFLICT(thread_id) DO UPDATE SET "
            "updated_at = excluded.updated_at",
            (thread_id, title[:80], now, now),
        )

    def list_threads(self, limit: int = 20) -> list[dict]:
        rows = self._exec("SELECT * FROM threads ORDER BY updated_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows.fetchall()]

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
            self._db.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
            self._db.commit()

    # -- chat transcript (for re-opening past chats in the UI) -----------------------------------
    def add_message(
        self, thread_id: str, role: str, content: str, payload: dict | None = None
    ) -> None:
        with self._lock:
            row = self._db.execute("SELECT COALESCE(MAX(idx), -1) + 1 FROM messages WHERE thread_id = ?",
                                   (thread_id,)).fetchone()  # fmt: skip
            self._db.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
                             (thread_id, row[0], role, content,
                              json.dumps(payload, default=str) if payload else None))  # fmt: skip
            self._db.commit()

    def get_messages(self, thread_id: str) -> list[dict]:
        rows = self._exec("SELECT role, content, payload FROM messages WHERE thread_id = ? ORDER BY idx",
                          (thread_id,)).fetchall()  # fmt: skip
        return [{"role": r["role"], "content": r["content"],
                 "payload": json.loads(r["payload"]) if r["payload"] else None} for r in rows]  # fmt: skip
