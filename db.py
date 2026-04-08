"""IncidentRoom persistence — SQLite for runs, ELO ratings, and custom scenarios."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "incidentroom.db"


def _conn():
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db():
    c = _conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        agent_type TEXT NOT NULL,
        model TEXT,
        system_prompt_hash TEXT,
        seed INTEGER,
        difficulty TEXT,
        score REAL,
        max_score REAL DEFAULT 100,
        outcome TEXT,
        faults_resolved INTEGER,
        faults_total INTEGER,
        ticks_used INTEGER,
        max_ticks INTEGER,
        total_prompt_tokens INTEGER DEFAULT 0,
        total_completion_tokens INTEGER DEFAULT 0,
        cost_usd REAL DEFAULT 0,
        events TEXT
    );
    CREATE TABLE IF NOT EXISTS elo_ratings (
        model TEXT PRIMARY KEY,
        rating REAL DEFAULT 1500,
        wins INTEGER DEFAULT 0,
        losses INTEGER DEFAULT 0,
        draws INTEGER DEFAULT 0,
        total_runs INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS scenarios (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        yaml_content TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """)
    c.commit()
    c.close()


# ── Runs ──────────────────────────────────────────────────────────────

def save_run(data: dict) -> str:
    rid = uuid.uuid4().hex[:8]
    c = _conn()
    c.execute(
        """INSERT INTO runs
           (id,timestamp,agent_type,model,system_prompt_hash,seed,difficulty,
            score,max_score,outcome,faults_resolved,faults_total,
            ticks_used,max_ticks,total_prompt_tokens,total_completion_tokens,
            cost_usd,events)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (rid, datetime.now(timezone.utc).isoformat(),
         data.get("agent_type", ""), data.get("model"),
         data.get("system_prompt_hash"), data.get("seed"),
         data.get("difficulty"), data.get("score"),
         data.get("max_score", 100), data.get("outcome"),
         data.get("faults_resolved"), data.get("faults_total"),
         data.get("ticks_used"), data.get("max_ticks"),
         data.get("total_prompt_tokens", 0),
         data.get("total_completion_tokens", 0),
         data.get("cost_usd", 0),
         json.dumps(data.get("events", []), default=str)))
    c.commit()
    c.close()
    return rid


def get_run(rid: str) -> dict | None:
    c = _conn()
    r = c.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
    c.close()
    if not r:
        return None
    d = dict(r)
    d["events"] = json.loads(d["events"]) if d["events"] else []
    return d


def list_runs(limit: int = 100) -> list[dict]:
    c = _conn()
    rows = c.execute(
        """SELECT id,timestamp,agent_type,model,seed,difficulty,score,outcome,
                  faults_resolved,faults_total,ticks_used,
                  total_prompt_tokens,total_completion_tokens,cost_usd
           FROM runs ORDER BY timestamp DESC LIMIT ?""", (limit,)).fetchall()
    c.close()
    return [dict(r) for r in rows]


# ── ELO ───────────────────────────────────────────────────────────────

def get_leaderboard(limit: int = 50) -> list[dict]:
    c = _conn()
    rows = c.execute(
        "SELECT * FROM elo_ratings ORDER BY rating DESC LIMIT ?", (limit,)
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]


def update_elo(model_a: str, model_b: str, score_a: float, score_b: float) -> dict:
    c = _conn()
    for m in (model_a, model_b):
        c.execute(
            "INSERT OR IGNORE INTO elo_ratings(model,rating,wins,losses,draws,total_runs) "
            "VALUES(?,1500,0,0,0,0)", (m,))

    ra = c.execute("SELECT rating FROM elo_ratings WHERE model=?", (model_a,)).fetchone()["rating"]
    rb = c.execute("SELECT rating FROM elo_ratings WHERE model=?", (model_b,)).fetchone()["rating"]

    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    eb = 1 - ea
    K = 32

    if score_a > score_b:
        sa, sb, wa, la, wb, lb, da, db = 1, 0, 1, 0, 0, 1, 0, 0
    elif score_b > score_a:
        sa, sb, wa, la, wb, lb, da, db = 0, 1, 0, 1, 1, 0, 0, 0
    else:
        sa, sb, wa, la, wb, lb, da, db = .5, .5, 0, 0, 0, 0, 1, 1

    new_ra = ra + K * (sa - ea)
    new_rb = rb + K * (sb - eb)

    c.execute("UPDATE elo_ratings SET rating=?,wins=wins+?,losses=losses+?,draws=draws+?,total_runs=total_runs+1 WHERE model=?",
              (new_ra, wa, la, da, model_a))
    c.execute("UPDATE elo_ratings SET rating=?,wins=wins+?,losses=losses+?,draws=draws+?,total_runs=total_runs+1 WHERE model=?",
              (new_rb, wb, lb, db, model_b))
    c.commit()
    c.close()
    return {"a": round(new_ra, 1), "b": round(new_rb, 1)}


def record_model_run(model: str, score: float):
    """Record a solo run for leaderboard stats (no ELO change)."""
    c = _conn()
    c.execute(
        "INSERT OR IGNORE INTO elo_ratings(model,rating,wins,losses,draws,total_runs) "
        "VALUES(?,1500,0,0,0,0)", (model,))
    c.execute("UPDATE elo_ratings SET total_runs=total_runs+1 WHERE model=?", (model,))
    c.commit()
    c.close()


# ── Scenarios ─────────────────────────────────────────────────────────

def save_scenario(name: str, description: str, yaml_content: str) -> str:
    sid = uuid.uuid4().hex[:8]
    c = _conn()
    c.execute("INSERT INTO scenarios(id,name,description,yaml_content,created_at) VALUES(?,?,?,?,?)",
              (sid, name, description, yaml_content, datetime.now(timezone.utc).isoformat()))
    c.commit()
    c.close()
    return sid


def get_scenario(sid: str) -> dict | None:
    c = _conn()
    r = c.execute("SELECT * FROM scenarios WHERE id=?", (sid,)).fetchone()
    c.close()
    return dict(r) if r else None


def list_scenarios() -> list[dict]:
    c = _conn()
    rows = c.execute("SELECT id,name,description,created_at FROM scenarios ORDER BY created_at DESC").fetchall()
    c.close()
    return [dict(r) for r in rows]


def delete_scenario(sid: str):
    c = _conn()
    c.execute("DELETE FROM scenarios WHERE id=?", (sid,))
    c.commit()
    c.close()


init_db()
