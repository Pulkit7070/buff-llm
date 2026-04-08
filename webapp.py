"""IncidentRoom web dashboard — FastAPI backend with SSE agent streaming."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from server.env import IncidentRoomEnv
from server.world import health
from llm_agent import (
    run_llm_agent, run_llm_agent_streaming, run_hitl_agent,
    hitl_respond as _hitl_respond, DEFAULT_SYSTEM_PROMPT,
)
from cost import calc_cost
import db

app = FastAPI(title="IncidentRoom v2")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

env = IncidentRoomEnv()


# ── Pages ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


# ── Manual / basic endpoints ──────────────────────────────────────────

@app.post("/api/reset")
async def reset(req: Request):
    body = await req.json()
    seed = int(body.get("seed", 42))
    difficulty = body.get("difficulty", "easy")
    scenario_id = body.get("scenario_id")
    if scenario_id:
        sc = db.get_scenario(scenario_id)
        if sc:
            from scenarios import build_world_from_scenario
            env.world = build_world_from_scenario(sc["yaml_content"], seed=seed)
            obs = env._observation(
                tool_result={"message": "Incident detected. Begin investigation."},
                tick_info={"done": False, "outcome": "running", "tick": 0})
            return _enrich(obs)
    obs = env.reset(seed=seed, difficulty=difficulty)
    return _enrich(obs)


@app.post("/api/step")
async def step(req: Request):
    body = await req.json()
    obs = env.step(body["tool_name"], body.get("args", {}))
    return _enrich(obs)


@app.get("/api/tools")
async def tools():
    return env.get_tools()


@app.get("/api/grade")
async def grade_endpoint():
    if env.world is None:
        return {"error": "no episode running"}
    return env.grade()


@app.get("/api/state")
async def state():
    if env.world is None:
        return {"error": "no episode running"}
    return _snap(env)


# ── OpenEnv spec endpoints (root-level aliases) ──────────────────────

@app.post("/reset")
@app.get("/reset")
async def openenv_reset(req: Request):
    try:
        body = await req.json()
    except Exception:
        body = {}
    seed = int(body.get("seed", 42))
    difficulty = body.get("difficulty", "easy")
    obs = env.reset(seed=seed, difficulty=difficulty)
    return _enrich(obs)


@app.post("/step")
async def openenv_step(req: Request):
    try:
        body = await req.json()
    except Exception:
        body = {}
    tool_name = body.get("tool_name") or body.get("action", "list_services")
    args = body.get("args", {})
    obs = env.step(tool_name, args)
    return _enrich(obs)


@app.get("/state")
async def openenv_state():
    if env.world is None:
        env.reset(seed=42, difficulty="easy")
    return _snap(env)


@app.get("/grade")
async def openenv_grade():
    if env.world is None:
        env.reset(seed=42, difficulty="easy")
    return env.grade()


@app.get("/tools")
async def openenv_tools():
    return env.get_tools()


# ── Rule-based / naive agent SSE ──────────────────────────────────────

@app.get("/api/agent/run")
async def agent_run(seed: int = 42, difficulty: str = "easy", agent: str = "smart"):
    async def generate():
        obs = env.reset(seed=seed, difficulty=difficulty)
        yield _sse("reset", {"observation": obs, "state": _snap(env)})
        await asyncio.sleep(0.6)

        agent_obj = NaiveAgent() if agent == "naive" else RuleBasedAgent()
        done = False
        step_num = 0
        events: list[dict] = []

        while not done and step_num < 40:
            thought, tool_name, tool_args = agent_obj.decide(env.world)
            yield _sse("thought", {"step": step_num, "text": thought})
            await asyncio.sleep(1.2)
            if tool_name is None:
                break
            obs = env.step(tool_name, tool_args)
            done = obs["done"]
            result = obs["tool_result"]
            ev_data = {"step": step_num, "tool": tool_name, "args": tool_args,
                       "result": result, "state": _snap(env), "tick": obs["tick"], "done": done}
            events.append({"event": "action", "data": ev_data})
            yield _sse("action", ev_data)
            await asyncio.sleep(0.8)
            agent_obj.observe(tool_name, tool_args, result, obs)
            step_num += 1

        grade = env.grade()
        yield _sse("done", {"grade": grade,
                            "outcome": obs["tick"].get("outcome", "unknown"),
                            "state": _snap(env)})
        # save run
        db.save_run({"agent_type": "naive" if agent == "naive" else "rule-based",
                      "seed": seed, "difficulty": difficulty,
                      "score": grade["score"], "outcome": obs["tick"].get("outcome"),
                      "faults_resolved": grade["faults_resolved"],
                      "faults_total": grade["faults_total"],
                      "ticks_used": grade["final_tick"],
                      "max_ticks": env.world.max_ticks, "events": events})

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── LLM Agent (standard) ─────────────────────────────────────────────

@app.post("/api/llm-agent/run")
async def llm_agent_run(req: Request):
    body = await req.json()
    api_key = body.get("api_key", "")
    if not api_key:
        return JSONResponse({"error": "api_key is required"}, status_code=400)

    async def generate():
        run_meta = {}
        async for event in run_llm_agent(
            api_key=api_key,
            api_base_url=body.get("api_base_url", "https://api.openai.com/v1"),
            model=body.get("model", "gpt-4.1-mini"),
            system_prompt=body.get("system_prompt"),
            seed=int(body.get("seed", 42)),
            difficulty=body.get("difficulty", "easy"),
            scenario_yaml=body.get("scenario_yaml"),
        ):
            if event["event"] == "done":
                run_meta = event["data"].pop("run_meta", {})
                grade = event["data"]["grade"]
                tok = event["data"].get("tokens", {})
                db.save_run({**run_meta,
                             "score": grade["score"],
                             "outcome": event["data"]["outcome"],
                             "faults_resolved": grade["faults_resolved"],
                             "faults_total": grade["faults_total"],
                             "ticks_used": grade["final_tick"],
                             "max_ticks": event["data"]["state"]["max_ticks"],
                             "total_prompt_tokens": tok.get("total_prompt", 0),
                             "total_completion_tokens": tok.get("total_completion", 0),
                             "cost_usd": tok.get("cost_usd", 0)})
                model = run_meta.get("model", "")
                if model:
                    db.record_model_run(model, grade["score"])
            yield _sse(event["event"], event["data"])

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── LLM Agent (streaming) ────────────────────────────────────────────

@app.post("/api/llm-agent/stream")
async def llm_agent_stream(req: Request):
    body = await req.json()
    api_key = body.get("api_key", "")
    if not api_key:
        return JSONResponse({"error": "api_key is required"}, status_code=400)

    async def generate():
        async for event in run_llm_agent_streaming(
            api_key=api_key,
            api_base_url=body.get("api_base_url", "https://api.openai.com/v1"),
            model=body.get("model", "gpt-4.1-mini"),
            system_prompt=body.get("system_prompt"),
            seed=int(body.get("seed", 42)),
            difficulty=body.get("difficulty", "easy"),
            scenario_yaml=body.get("scenario_yaml"),
        ):
            if event["event"] == "done":
                meta = event["data"].pop("run_meta", {})
                grade = event["data"]["grade"]
                tok = event["data"].get("tokens", {})
                db.save_run({**meta,
                             "score": grade["score"],
                             "outcome": event["data"]["outcome"],
                             "faults_resolved": grade["faults_resolved"],
                             "faults_total": grade["faults_total"],
                             "ticks_used": grade["final_tick"],
                             "max_ticks": event["data"]["state"]["max_ticks"],
                             "total_prompt_tokens": tok.get("total_prompt", 0),
                             "total_completion_tokens": tok.get("total_completion", 0),
                             "cost_usd": tok.get("cost_usd", 0)})
            yield _sse(event["event"], event["data"])

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Human-in-the-Loop ────────────────────────────────────────────────

@app.post("/api/hitl/run")
async def hitl_run(req: Request):
    body = await req.json()
    api_key = body.get("api_key", "")
    if not api_key:
        return JSONResponse({"error": "api_key is required"}, status_code=400)
    session_id = uuid.uuid4().hex[:8]

    async def generate():
        async for event in run_hitl_agent(
            session_id=session_id,
            api_key=api_key,
            api_base_url=body.get("api_base_url", "https://api.openai.com/v1"),
            model=body.get("model", "gpt-4.1-mini"),
            system_prompt=body.get("system_prompt"),
            seed=int(body.get("seed", 42)),
            difficulty=body.get("difficulty", "easy"),
            scenario_yaml=body.get("scenario_yaml"),
        ):
            if event["event"] == "done":
                event["data"].pop("run_meta", None)
            yield _sse(event["event"], event["data"])

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/hitl/respond")
async def hitl_respond_endpoint(req: Request):
    body = await req.json()
    sid = body.get("session_id", "")
    ok = _hitl_respond(sid, body.get("action", "approve"),
                       body.get("tool"), body.get("args"))
    if not ok:
        return JSONResponse({"error": "session not found"}, status_code=404)
    return {"ok": True}


# ── Multi-Model Arena ────────────────────────────────────────────────

@app.post("/api/arena/llm")
async def arena_llm(req: Request):
    body = await req.json()
    cfg_a = body.get("model_a", {})
    cfg_b = body.get("model_b", {})
    seed = int(body.get("seed", 42))
    difficulty = body.get("difficulty", "easy")

    for cfg, label in [(cfg_a, "model_a"), (cfg_b, "model_b")]:
        if not cfg.get("api_key"):
            return JSONResponse({"error": f"{label}.api_key is required"}, status_code=400)

    async def generate():
        queue: asyncio.Queue = asyncio.Queue()

        async def run_lane(lane: str, cfg: dict):
            async for ev in run_llm_agent(
                api_key=cfg["api_key"],
                api_base_url=cfg.get("api_base_url", "https://api.openai.com/v1"),
                model=cfg.get("model", "gpt-4.1-mini"),
                system_prompt=cfg.get("system_prompt"),
                seed=seed, difficulty=difficulty,
            ):
                ev["data"]["lane"] = lane
                await queue.put(ev)
            await queue.put({"event": "_done", "data": {"lane": lane}})

        task_a = asyncio.create_task(run_lane("a", cfg_a))
        task_b = asyncio.create_task(run_lane("b", cfg_b))

        done_lanes = 0
        grade_a = grade_b = None
        while done_lanes < 2:
            ev = await queue.get()
            if ev["event"] == "_done":
                done_lanes += 1
                continue
            if ev["event"] == "done":
                if ev["data"]["lane"] == "a":
                    grade_a = ev["data"]["grade"]
                else:
                    grade_b = ev["data"]["grade"]
            yield _sse(ev["event"], ev["data"])

        await task_a
        await task_b

        model_a_name = cfg_a.get("model", "model-a")
        model_b_name = cfg_b.get("model", "model-b")
        sa = grade_a["score"] if grade_a else 0
        sb = grade_b["score"] if grade_b else 0

        elo = db.update_elo(model_a_name, model_b_name, sa, sb)

        yield _sse("arena_done", {
            "grade_a": grade_a, "grade_b": grade_b,
            "model_a": model_a_name, "model_b": model_b_name,
            "elo": elo,
        })

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Prompt A/B ───────────────────────────────────────────────────────

@app.post("/api/arena/prompt-ab")
async def arena_prompt_ab(req: Request):
    body = await req.json()
    api_key = body.get("api_key", "")
    if not api_key:
        return JSONResponse({"error": "api_key is required"}, status_code=400)

    model = body.get("model", "gpt-4.1-mini")
    base_url = body.get("api_base_url", "https://api.openai.com/v1")
    prompt_a = body.get("prompt_a", DEFAULT_SYSTEM_PROMPT)
    prompt_b = body.get("prompt_b", DEFAULT_SYSTEM_PROMPT)
    seed = int(body.get("seed", 42))
    difficulty = body.get("difficulty", "easy")

    async def generate():
        queue: asyncio.Queue = asyncio.Queue()

        async def run_lane(lane: str, prompt: str):
            async for ev in run_llm_agent(
                api_key=api_key, api_base_url=base_url,
                model=model, system_prompt=prompt,
                seed=seed, difficulty=difficulty,
            ):
                ev["data"]["lane"] = lane
                await queue.put(ev)
            await queue.put({"event": "_done", "data": {"lane": lane}})

        task_a = asyncio.create_task(run_lane("a", prompt_a))
        task_b = asyncio.create_task(run_lane("b", prompt_b))

        done_lanes = 0
        grade_a = grade_b = None
        while done_lanes < 2:
            ev = await queue.get()
            if ev["event"] == "_done":
                done_lanes += 1
                continue
            if ev["event"] == "done":
                if ev["data"]["lane"] == "a":
                    grade_a = ev["data"]["grade"]
                else:
                    grade_b = ev["data"]["grade"]
            yield _sse(ev["event"], ev["data"])

        await task_a
        await task_b
        yield _sse("arena_done", {"grade_a": grade_a, "grade_b": grade_b,
                                   "model": model, "prompt_a_hash": hashlib.sha256(prompt_a.encode()).hexdigest()[:8],
                                   "prompt_b_hash": hashlib.sha256(prompt_b.encode()).hexdigest()[:8]})

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Batch Eval ───────────────────────────────────────────────────────

@app.post("/api/batch/run")
async def batch_run(req: Request):
    body = await req.json()
    api_key = body.get("api_key", "")
    if not api_key:
        return JSONResponse({"error": "api_key is required"}, status_code=400)

    model = body.get("model", "gpt-4.1-mini")
    base_url = body.get("api_base_url", "https://api.openai.com/v1")
    prompt = body.get("system_prompt")
    seeds = body.get("seeds", [42, 137, 256, 7, 1024])
    difficulties = body.get("difficulties", ["easy", "medium", "hard"])

    async def generate():
        results = []
        total = len(seeds) * len(difficulties)
        idx = 0

        for diff in difficulties:
            for seed in seeds:
                idx += 1
                yield _sse("batch_progress", {
                    "index": idx, "total": total,
                    "seed": seed, "difficulty": diff, "status": "running"})

                grade_data = None
                tok_data = {}
                outcome = "error"
                try:
                    async for ev in run_llm_agent(
                        api_key=api_key, api_base_url=base_url,
                        model=model, system_prompt=prompt,
                        seed=seed, difficulty=diff,
                    ):
                        if ev["event"] == "done":
                            grade_data = ev["data"]["grade"]
                            tok_data = ev["data"].get("tokens", {})
                            outcome = ev["data"]["outcome"]
                        elif ev["event"] == "error":
                            outcome = "error"
                            break
                except Exception:
                    outcome = "error"

                row = {
                    "seed": seed, "difficulty": diff,
                    "score": grade_data["score"] if grade_data else 0,
                    "outcome": outcome,
                    "faults": f'{grade_data["faults_resolved"]}/{grade_data["faults_total"]}' if grade_data else "0/0",
                    "ticks": grade_data["final_tick"] if grade_data else 0,
                    "tokens": tok_data.get("total_prompt", 0) + tok_data.get("total_completion", 0),
                    "cost": tok_data.get("cost_usd", 0),
                }
                results.append(row)
                yield _sse("batch_result", {**row, "index": idx, "total": total})

        # Summary
        scores = [r["score"] for r in results if r["outcome"] != "error"]
        total_cost = sum(r["cost"] for r in results)
        yield _sse("batch_done", {
            "results": results,
            "model": model,
            "avg_score": round(sum(scores) / max(len(scores), 1), 1),
            "total_cost": round(total_cost, 4),
            "episodes": len(results),
            "successes": sum(1 for r in results if r["outcome"] == "success"),
        })

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Leaderboard & History ────────────────────────────────────────────

@app.get("/api/leaderboard")
async def leaderboard():
    return db.get_leaderboard()


@app.get("/api/runs")
async def runs_list(limit: int = 100):
    return db.list_runs(limit)


@app.get("/api/runs/{run_id}")
async def run_detail(run_id: str):
    r = db.get_run(run_id)
    if not r:
        return JSONResponse({"error": "not found"}, status_code=404)
    return r


# ── Scenarios ────────────────────────────────────────────────────────

@app.get("/api/scenarios")
async def scenarios_list():
    return db.list_scenarios()


@app.get("/api/scenarios/example")
async def scenario_example():
    from scenarios import EXAMPLE_YAML
    return {"yaml": EXAMPLE_YAML}


@app.get("/api/scenarios/{sid}")
async def scenario_detail(sid: str):
    s = db.get_scenario(sid)
    if not s:
        return JSONResponse({"error": "not found"}, status_code=404)
    return s


@app.post("/api/scenarios")
async def scenario_create(req: Request):
    body = await req.json()
    from scenarios import parse_scenario
    try:
        parse_scenario(body.get("yaml_content", ""))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    sid = db.save_scenario(
        body.get("name", "Untitled"),
        body.get("description", ""),
        body["yaml_content"])
    return {"id": sid}


@app.delete("/api/scenarios/{sid}")
async def scenario_delete(sid: str):
    db.delete_scenario(sid)
    return {"ok": True}


# ── Rule-based arena (existing) ──────────────────────────────────────

@app.get("/api/arena/run")
async def arena_run(seed: int = 42, difficulty: str = "easy"):
    async def generate():
        env_a, env_b = IncidentRoomEnv(), IncidentRoomEnv()
        obs_a = env_a.reset(seed=seed, difficulty=difficulty)
        obs_b = env_b.reset(seed=seed, difficulty=difficulty)
        yield _sse("reset", {"a": {"state": _snap(env_a)}, "b": {"state": _snap(env_b)}})
        await asyncio.sleep(0.5)

        agent_a, agent_b = RuleBasedAgent(), NaiveAgent()
        done_a = done_b = False
        step = 0

        while (not done_a or not done_b) and step < 45:
            for lane, ag, ev, done_flag in [
                ("a", agent_a, env_a, done_a), ("b", agent_b, env_b, done_b)]:
                if done_flag:
                    continue
                thought, tool, args = ag.decide(ev.world)
                if tool is None:
                    continue
                yield _sse("thought", {"lane": lane, "step": step, "text": thought})
                await asyncio.sleep(0.5)
                obs = ev.step(tool, args)
                ag.observe(tool, args, obs["tool_result"], obs)
                is_done = obs["done"]
                yield _sse("action", {"lane": lane, "step": step, "tool": tool, "args": args,
                                       "result": obs["tool_result"], "state": _snap(ev),
                                       "tick": obs["tick"], "done": is_done})
                await asyncio.sleep(0.3)
                if lane == "a":
                    done_a = is_done
                else:
                    done_b = is_done
            step += 1

        grade_a, grade_b = env_a.grade(), env_b.grade()
        yield _sse("done", {"grade_a": grade_a, "grade_b": grade_b,
                             "a": {"grade": grade_a, "state": _snap(env_a)},
                             "b": {"grade": grade_b, "state": _snap(env_b)}})

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Agents ───────────────────────────────────────────────────────────

class RuleBasedAgent:
    """Deterministic SRE agent that narrates its reasoning."""
    def __init__(self):
        self.phase = "recon"
        self.svc_queue: list[str] = []
        self.triage_started = False
        self.unhealthy: list[dict] = []
        self.investigated: set[str] = set()
        self.acted_on: set[str] = set()
        self.findings: dict[str, dict] = {}
        self.topology_fetched = False
        self.services_listed = False

    def decide(self, world):
        w = world
        if not self.services_listed:
            self.services_listed = True
            return ("First, I need to understand what services exist. Let me list all services.", "list_services", {})
        if not self.topology_fetched:
            self.topology_fetched = True
            return ("Now I need the dependency graph to trace cascading failures.", "get_topology", {})
        if self.phase == "recon":
            if not self.triage_started:
                self.svc_queue = list(w.services.keys())
                self.triage_started = True
            if self.svc_queue:
                svc = self.svc_queue.pop(0)
                return (f"Checking metrics for {svc}.", "get_metrics", {"service": svc})
            self.phase = "investigate"
        if self.phase == "investigate":
            for name, info in list(self.findings.items()):
                if info.get("status") != "healthy":
                    sub = info.get("_sub", "logs")
                    if sub == "logs":
                        info["_sub"] = "changes"
                        return (f"{name} is {info['status']}. Pulling logs.", "get_logs", {"service": name, "n": 15})
                    if sub == "changes":
                        info["_sub"] = "done"
                        self.investigated.add(name)
                        return (f"Checking recent changes on {name}.", "get_recent_changes", {"service": name})
            for name, svc in w.services.items():
                if svc.status != "healthy" and name not in self.investigated:
                    self.investigated.add(name)
                    self.findings[name] = {"status": svc.status, "_sub": "done"}
                    return (f"{name} became {svc.status} — checking logs.", "get_logs", {"service": name, "n": 10})
            self.phase = "remediate"
        if self.phase == "remediate":
            for name, svc in w.services.items():
                if svc.status == "healthy" or name in self.acted_on:
                    continue
                self.acted_on.add(name)
                logs_text = " ".join(l.get("msg", "") for l in svc.log_buffer[-20:])
                if svc.kind == "cache" and "cache_eviction" in logs_text:
                    return (f"DIAGNOSIS: {name} cache eviction storm. Restarting.", "restart_pod", {"service": name})
                if svc.kind == "database" and "conn_pool_active" in logs_text:
                    return (f"DIAGNOSIS: {name} pool exhausted. Killing queries.", "kill_long_queries", {"service": name})
                if "tls_handshake_fail" in logs_text:
                    peer = None
                    for log in svc.log_buffer:
                        if "peer=" in log.get("msg", ""):
                            peer = log["msg"].split("peer=")[1].split()[0]
                            break
                    if peer:
                        return (f"DIAGNOSIS: TLS failure {name}<->{peer}. Rotating cert.", "rotate_cert", {"service_a": name, "service_b": peer})
                if "retry_amplification" in logs_text or "retry_rate" in logs_text:
                    return (f"DIAGNOSIS: {name} retry storm. Circuit breaker.", "enable_circuit_breaker", {"service": name})
                if svc.config_flags:
                    for flag, val in svc.config_flags.items():
                        if val and ("config_change" in logs_text or "intermittent" in logs_text):
                            return (f"DIAGNOSIS: {name} config drift flag '{flag}'. Toggling.", "toggle_feature_flag", {"service": name, "flag_name": flag})
                deploys = [c for c in svc.recent_changes if c.get("type") == "deploy"]
                if deploys and svc.memory_pct > 75:
                    return (f"DIAGNOSIS: {name} memory leak from deploy. Rolling back.", "rollback", {"service": name})
                if svc.kind == "cache":
                    return (f"{name} unhealthy cache. Restarting.", "restart_pod", {"service": name})
                return (f"{name} is {svc.status}. Attempting restart.", "restart_pod", {"service": name})
            self.phase = "monitor"
        if self.phase == "monitor":
            sick = [n for n, s in w.services.items() if s.status != "healthy"]
            if not sick:
                return ("All healthy. Verifying.", "list_services", {})
            unacted = [n for n in sick if n not in self.acted_on]
            if unacted:
                self.phase = "remediate"
                return self.decide(w)
            for name in sick:
                rk = f"_retry_{name}"
                if self.findings.get(rk):
                    continue
                self.findings[rk] = True
                svc = w.services[name]
                logs_text = " ".join(l.get("msg", "") for l in svc.log_buffer[-20:])
                if "tls_handshake_fail" in logs_text:
                    for log in svc.log_buffer:
                        if "peer=" in log.get("msg", ""):
                            peer = log["msg"].split("peer=")[1].split()[0]
                            return (f"RETRY: {name} TLS with {peer}.", "rotate_cert", {"service_a": name, "service_b": peer})
                if "retry_amplification" in logs_text:
                    return (f"RETRY: {name} circuit breaker.", "enable_circuit_breaker", {"service": name})
                if svc.config_flags:
                    for flag, val in svc.config_flags.items():
                        if val:
                            return (f"RETRY: {name} toggle '{flag}'.", "toggle_feature_flag", {"service": name, "flag_name": flag})
                deploys = [c for c in svc.recent_changes if c.get("type") == "deploy"]
                if deploys and svc.memory_pct > 70:
                    return (f"RETRY: {name} rollback.", "rollback", {"service": name})
                if svc.kind == "database":
                    return (f"RETRY: {name} kill queries.", "kill_long_queries", {"service": name})
                return (f"RETRY: {name} restart.", "restart_pod", {"service": name})
            self._mon_count = getattr(self, '_mon_count', 0) + 1
            target = sick[0]
            return (f"Monitoring ({self._mon_count}). Checking {target}.", "get_metrics", {"service": target})
        return ("No more actions.", None, None)

    def observe(self, tool_name, tool_args, result, obs):
        if tool_name == "get_metrics" and "service" in result:
            self.findings[result["service"]] = {**result, "_sub": "logs"}
        if tool_name == "list_services" and "services" in result:
            for s in result["services"]:
                if s["name"] not in self.findings:
                    self.findings[s["name"]] = {"status": s["status"], "_sub": "logs"}


class NaiveAgent:
    def __init__(self):
        self.listed = False
        self.restarted: set[str] = set()

    def decide(self, world):
        if not self.listed:
            self.listed = True
            return ("Listing services...", "list_services", {})
        for name, svc in world.services.items():
            if svc.status != "healthy" and name not in self.restarted:
                self.restarted.add(name)
                return (f"{name} is {svc.status}. Restarting.", "restart_pod", {"service": name})
        for name, svc in world.services.items():
            if svc.status != "healthy":
                return (f"Checking {name}...", "get_metrics", {"service": name})
        return ("Waiting...", "list_services", {})

    def observe(self, *a):
        pass


# ── Helpers ──────────────────────────────────────────────────────────

def _snap(e: IncidentRoomEnv) -> dict:
    w = e.world
    return {
        "tick": w.tick, "max_ticks": w.max_ticks,
        "difficulty": w.difficulty, "seed": w.seed,
        "user_impact_total": round(w.user_impact_total, 2),
        "wasted_actions": w.wasted_actions,
        "services": {
            name: {
                "name": s.name, "kind": s.kind, "status": s.status,
                "error_rate": round(s.error_rate, 4),
                "latency_p99": round(s.latency_p99, 1),
                "cpu_pct": round(s.cpu_pct, 1),
                "memory_pct": round(s.memory_pct, 1),
                "health": round(health(s), 3),
                "dependencies": s.dependencies, "region": s.region,
                "restart_cooldown": s.restart_cooldown,
                "history": w.metric_history.get(name, []),
            }
            for name, s in w.services.items()
        },
        "topology": w.topology,
        "faults_total": len(w.active_faults),
        "faults_resolved": sum(1 for f in w.active_faults if f.resolved),
    }


def _enrich(obs):
    if env.world:
        w = env.world
        obs["state"] = {
            "tick": w.tick, "max_ticks": w.max_ticks,
            "user_impact_total": round(w.user_impact_total, 2),
            "faults_total": len(w.active_faults),
            "faults_resolved": sum(1 for f in w.active_faults if f.resolved),
        }
    if obs.get("done"):
        obs["grade"] = env.grade()
    return obs


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  IncidentRoom v2 Dashboard")
    print(f"  http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
