#!/usr/bin/env python3
"""sentinel-ledger — sidecar FastAPI service for Sentinel council audit logging.

Port: 8790
Auth: Bearer token via LEDGER_AUTH_TOKEN env var
DB:   SQLite at LEDGER_DB_PATH (default: ./ledger.db)

Endpoints:
  POST /record   — append a council verdict record
  GET  /stats    — aggregated verdict counts + latency stats
  GET  /health   — liveness probe

Usage:
    LEDGER_AUTH_TOKEN=<token> uvicorn ledger:app --port 8790
"""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LEDGER_DB_PATH: str = os.environ.get("LEDGER_DB_PATH", "./ledger.db")
LEDGER_AUTH_TOKEN: str = os.environ.get("LEDGER_AUTH_TOKEN", "").strip()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="sentinel-ledger", version="1.0.0")
_bearer = HTTPBearer(auto_error=True)


def _check_token(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> None:
    if not LEDGER_AUTH_TOKEN:
        raise HTTPException(status_code=503, detail="ledger auth not configured")
    if creds.credentials != LEDGER_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _init_db() -> None:
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL    NOT NULL,           -- unix epoch float
                session_id  TEXT,
                provider    TEXT,                       -- e.g. nvidia, groq, cerebras
                model       TEXT,
                prompt_hash TEXT,                       -- sha256[:16] of prompt
                verdict     TEXT    NOT NULL,           -- CLEAN | MINOR | MAJOR | BLOCK
                latency_ms  REAL,
                tokens_in   INTEGER,
                tokens_out  INTEGER,
                notes       TEXT
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_verdict  ON records(verdict)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_ts       ON records(ts)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_provider ON records(provider)")
        # Forward-compat migration: add provider column to existing databases
        try:
            con.execute("ALTER TABLE records ADD COLUMN provider TEXT")
        except Exception:
            pass  # column already exists


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    con = sqlite3.connect(LEDGER_DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

VALID_VERDICTS = {"CLEAN", "MINOR", "MAJOR", "BLOCK"}


class RecordIn(BaseModel):
    verdict: str                              = Field(..., description="CLEAN | MINOR | MAJOR | BLOCK")
    session_id: Optional[str]                 = None
    provider: Optional[str]                   = None
    model: Optional[str]                      = None
    prompt_hash: Optional[str]                = None
    latency_ms: Optional[float]               = None
    tokens_in: Optional[int]                  = None
    tokens_out: Optional[int]                 = None
    notes: Optional[str]                      = None


class RecordOut(BaseModel):
    id: int
    ts: float
    verdict: str
    session_id: Optional[str]
    provider: Optional[str]
    model: Optional[str]
    latency_ms: Optional[float]


class StatsOut(BaseModel):
    total: int
    by_verdict: dict[str, int]
    avg_latency_ms: Optional[float]
    p95_latency_ms: Optional[float]
    top_models: list[dict]
    top_providers: list[dict]
    window_24h: dict[str, int]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _startup() -> None:
    _init_db()


@app.get("/health")
def health() -> dict:
    """Liveness probe — no auth required."""
    return {"status": "ok", "ts": datetime.now(timezone.utc).isoformat()}


@app.post("/record", response_model=RecordOut, dependencies=[Depends(_check_token)])
def record(body: RecordIn) -> RecordOut:
    verdict = body.verdict.upper()
    if verdict not in VALID_VERDICTS:
        raise HTTPException(
            status_code=422,
            detail=f"verdict must be one of {sorted(VALID_VERDICTS)}"
        )
    now = time.time()
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO records
               (ts, session_id, provider, model, prompt_hash, verdict,
                latency_ms, tokens_in, tokens_out, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                now,
                body.session_id,
                body.provider,
                body.model,
                body.prompt_hash,
                verdict,
                body.latency_ms,
                body.tokens_in,
                body.tokens_out,
                body.notes,
            ),
        )
        row_id = cur.lastrowid

    return RecordOut(
        id=row_id,
        ts=now,
        verdict=verdict,
        session_id=body.session_id,
        provider=body.provider,
        model=body.model,
        latency_ms=body.latency_ms,
    )


@app.get("/stats", response_model=StatsOut, dependencies=[Depends(_check_token)])
def stats() -> StatsOut:
    with _conn() as con:
        total = con.execute("SELECT COUNT(*) FROM records").fetchone()[0]

        by_verdict_rows = con.execute(
            "SELECT verdict, COUNT(*) AS n FROM records GROUP BY verdict"
        ).fetchall()
        by_verdict = {r["verdict"]: r["n"] for r in by_verdict_rows}

        latencies = [
            r[0]
            for r in con.execute(
                "SELECT latency_ms FROM records WHERE latency_ms IS NOT NULL ORDER BY latency_ms"
            ).fetchall()
        ]
        avg_lat: Optional[float] = None
        p95_lat: Optional[float] = None
        if latencies:
            avg_lat = sum(latencies) / len(latencies)
            p95_idx = max(0, int(len(latencies) * 0.95) - 1)
            p95_lat = latencies[p95_idx]

        top_model_rows = con.execute(
            """SELECT model, COUNT(*) AS n FROM records
               WHERE model IS NOT NULL
               GROUP BY model ORDER BY n DESC LIMIT 5"""
        ).fetchall()
        top_models = [{"model": r["model"], "count": r["n"]} for r in top_model_rows]

        top_provider_rows = con.execute(
            """SELECT provider, COUNT(*) AS n FROM records
               WHERE provider IS NOT NULL
               GROUP BY provider ORDER BY n DESC LIMIT 10"""
        ).fetchall()
        top_providers = [{"provider": r["provider"], "count": r["n"]} for r in top_provider_rows]

        cutoff_24h = time.time() - 86400
        w24_rows = con.execute(
            "SELECT verdict, COUNT(*) AS n FROM records WHERE ts >= ? GROUP BY verdict",
            (cutoff_24h,),
        ).fetchall()
        window_24h = {r["verdict"]: r["n"] for r in w24_rows}

    return StatsOut(
        total=total,
        by_verdict=by_verdict,
        avg_latency_ms=round(avg_lat, 2) if avg_lat is not None else None,
        p95_latency_ms=round(p95_lat, 2) if p95_lat is not None else None,
        top_models=top_models,
        top_providers=top_providers,
        window_24h=window_24h,
    )
