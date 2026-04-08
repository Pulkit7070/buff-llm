# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

IncidentRoom v2 — a tick-based SRE incident-response simulator for the Meta OpenEnv hackathon. Agents diagnose and remediate infrastructure faults under a tick budget. The environment is fully deterministic (no LLM in the sim), grading is a pure function over world state, and wrong actions inject new faults rather than just penalizing score.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Local demo (no API key needed)
python demo.py              # single easy episode
python demo.py --all        # easy/medium/hard

# CLI inference (requires env vars — OpenEnv spec format)
export HF_TOKEN="your-key"
export MODEL_NAME="gpt-4.1-mini"
export API_BASE_URL="https://api.openai.com/v1"
python inference.py

# Web dashboard (PORT env var, default 8000, HF Spaces uses 7860)
python webapp.py

# Docker build
docker build -t incidentroom .
docker run -p 7860:7860 incidentroom
```

No test suite exists. Verify changes by running `python demo.py --all` (rule-based agent across difficulties) or starting the webapp and exercising endpoints.

## Architecture

```
openenv.yaml        — OpenEnv spec: tasks, action/observation space, scoring config
Dockerfile          — Container build (python:3.11-slim, port 7860 for HF Spaces)
inference.py        — OpenEnv-compliant agent: [START]/[STEP]/[END] stdout, scores in [0,1]
demo.py             — Local rule-based agent demo with colored output
webapp.py           — FastAPI server: REST + SSE streaming, embeds RuleBasedAgent & NaiveAgent
llm_agent.py        — 3 async generator modes: standard, streaming (token-by-token), HITL
db.py               — SQLite persistence (runs, elo_ratings, scenarios tables), auto-inits on import
cost.py             — Per-model token pricing lookup and cost calculation
scenarios.py        — YAML custom scenario parser → World objects

server/
  env.py            — IncidentRoomEnv: reset(seed, difficulty) / step(tool, args) / grade()
  world.py          — World & Service dataclasses, health(), tick_world()
  faults.py         — Base Fault class + 6 primary faults + 2 consequence faults
  generator.py      — Procedural world generation (seeded RNG, difficulty configs)
  tools.py          — 13 tool handlers + TOOL_SCHEMAS (OpenAI function-calling format)
  grader.py         — Pure scoring: 40 faults + 25 health + 20 impact + 15 efficiency = 100

static/index.html   — Single-file frontend (dark theme, SSE, topology canvas, service cards)
```

## Key Design Patterns

**Episode lifecycle:** `env.reset()` → generator creates World with services + injected faults → agent loop calls `env.step(tool, args)` which runs the tool handler, checks fault resolution/consequences, then `tick_world()` → `env.grade()` returns final score.

**SSE streaming:** GET endpoints use `EventSource`, POST endpoints (carrying API keys) use `fetch()` + `ReadableStream` with manual SSE frame parsing. All agent generators yield `{"event": str, "data": dict}` dicts.

**Fault consequence mechanic:** Each fault defines `check_wrong_action(world, tool, args)`. Wrong actions don't just penalize — they inject new `LatentDefect` or `FlappingService` faults into the world, making the problem actively harder.

**Determinism:** Same `(seed, difficulty)` pair always produces the identical world. The RNG is `random.Random(seed)`, never the global random state.

## Important Conventions

- Every tool call (even reads like `list_services`) advances the tick counter by 1. Agents must be efficient.
- `server/tools.py` has two registries: `TOOL_SCHEMAS` (OpenAI function format for the LLM) and `TOOL_HANDLERS` (actual execution functions).
- `_snap_env()` / `_snap()` helpers serialize world state for SSE — duplicated in both `webapp.py` and `llm_agent.py` to avoid circular imports.
- The `db.py` module auto-initializes the SQLite database on import (`init_db()` at module level).
- Custom scenarios use YAML with `services` and `faults` lists; fault types map via `FAULT_MAP` in `scenarios.py`.
