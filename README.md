# IncidentRoom v2

A procedurally generated SRE incident-response environment for the **Meta OpenEnv hackathon**.

IncidentRoom teaches stateful decision-making in a world that degrades while the agent thinks. That learning signal does not exist anywhere else in the OpenEnv hub.

## What it is

A tick-based simulator where services have metrics that drift, dependencies that cascade, and faults that grow over time. The agent has 13 tools and a step budget. The episode ends when the world is healthy again — or the budget runs out.

- **No LLM in the env.** The simulator is fully deterministic.
- **No real cloud SDKs.** Everything is Python dicts.
- **No string matching.** The grader is a pure function over final World state.
- **Wrong actions have consequences.** The world gets worse, not just a penalty.

## Architecture

```
inference.py          ← OpenAI function-calling agent loop
server/
  env.py              ← IncidentRoomEnv: reset() / step() / grade()
  world.py            ← World + Service dataclasses, tick()
  faults.py           ← 6 fault templates + 2 consequence faults
  generator.py        ← generate(difficulty, seed) → World
  tools.py            ← 13 tool handlers + OpenAI schemas
  grader.py           ← Pure scoring function
```

## Fault Templates

| Fault | Target | Fix | Wrong Action Consequence |
|-------|--------|-----|--------------------------|
| Memory leak after deploy | API/worker with recent deploy | `rollback(target)` | Rolling back wrong service plants latent defect |
| Cache eviction storm | Cache service | `restart_pod(cache)` or `scale_up(cache)` | Restarting a dependent causes it to flap |
| Config drift | API/worker with flag change | `toggle_feature_flag(target, flag)` | Rollback adds latency without fixing flag |
| Dependency timeout amplification | API with slow dep | `enable_circuit_breaker(target)` | `scale_up` multiplies retry load |
| DB pool exhaustion | Database | `kill_long_queries(db)` or `scale_up(db)` | Restarting a DB client kills in-flight connections |
| Cert expiry between services | API/gateway pair | `rotate_cert(svc_a, svc_b)` | Nothing else helps — just wastes ticks |

## Tools

**Read (no side-effects):** `list_services`, `get_metrics`, `get_logs`, `get_topology`, `get_recent_changes`

**Action (mutate world):** `restart_pod`, `rollback`, `scale_up`, `toggle_feature_flag`, `enable_circuit_breaker`, `drain_region`, `kill_long_queries`, `rotate_cert`

## Difficulty

| Level | Services | Faults | Red Herrings | Tick Budget |
|-------|----------|--------|--------------|-------------|
| easy | 3 | 1 | 0 | 15 |
| medium | 6 | 1 | 2 | 20 |
| hard | 8 | 2 | 2 | 25 |

Red herrings: recent deploys or config changes on healthy services that look suspicious but are unrelated.

## Scoring (100 pts)

- **Fault resolution** (40 pts) — fraction of faults resolved
- **Service health** (25 pts) — average health at episode end
- **User impact** (20 pts) — lower cumulative impact is better
- **Time efficiency** (15 pts) — bonus for fast resolution (only if all faults fixed)

## Quickstart

```bash
pip install -r requirements.txt

# Run inference (requires HF_TOKEN)
export HF_TOKEN="your-token"
export MODEL_NAME="gpt-4.1-mini"
python inference.py
```

## Env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `MODEL_NAME` | `gpt-4.1-mini` | Model to use |
| `HF_TOKEN` | *(required)* | API key |
