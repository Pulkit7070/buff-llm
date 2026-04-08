"""IncidentRoom web dashboard — FastAPI backend with SSE agent streaming."""
from __future__ import annotations

import asyncio
import json
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from server.env import IncidentRoomEnv
from server.world import health

app = FastAPI(title="IncidentRoom v2")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

env = IncidentRoomEnv()


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.post("/api/reset")
async def reset(req: Request):
    body = await req.json()
    seed = body.get("seed", 42)
    difficulty = body.get("difficulty", "easy")
    obs = env.reset(seed=int(seed), difficulty=difficulty)
    return _enrich(obs)


@app.post("/api/step")
async def step(req: Request):
    body = await req.json()
    tool_name = body["tool_name"]
    tool_args = body.get("args", {})
    obs = env.step(tool_name, tool_args)
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
    return _world_snapshot()


@app.get("/api/agent/run")
async def agent_run(seed: int = 42, difficulty: str = "easy"):
    """SSE endpoint — streams agent thought + action events in real time."""
    async def generate():
        obs = env.reset(seed=seed, difficulty=difficulty)
        yield _sse("reset", {
            "observation": obs,
            "state": _world_snapshot(),
        })
        await asyncio.sleep(0.6)

        agent = RuleBasedAgent()
        done = False
        step_num = 0

        while not done and step_num < 40:
            # Agent thinks
            thought, tool_name, tool_args = agent.decide(env.world)
            yield _sse("thought", {"step": step_num, "text": thought})
            await asyncio.sleep(1.2)

            if tool_name is None:
                break

            # Agent acts
            obs = env.step(tool_name, tool_args)
            done = obs["done"]
            result = obs["tool_result"]

            yield _sse("action", {
                "step": step_num,
                "tool": tool_name,
                "args": tool_args,
                "result": result,
                "state": _world_snapshot(),
                "tick": obs["tick"],
                "done": done,
            })
            await asyncio.sleep(0.8)

            agent.observe(tool_name, tool_args, result, obs)
            step_num += 1

        # Final grade
        grade = env.grade()
        yield _sse("done", {
            "grade": grade,
            "outcome": obs["tick"].get("outcome", "unknown"),
            "state": _world_snapshot(),
        })

    return StreamingResponse(generate(), media_type="text/event-stream")


# -----------------------------------------------------------------------
# Rule-based agent with explicit reasoning
# -----------------------------------------------------------------------

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

        # Phase 1: Recon
        if not self.services_listed:
            self.services_listed = True
            return (
                "First, I need to understand what services exist in this system. "
                "Let me list all services and their current status.",
                "list_services", {}
            )

        if not self.topology_fetched:
            self.topology_fetched = True
            return (
                "Now I need to understand the dependency graph. Which services "
                "depend on which? This is critical for tracing cascading failures.",
                "get_topology", {}
            )

        # Phase 2: Triage — get metrics for all services (once)
        if self.phase == "recon":
            if not self.triage_started:
                self.svc_queue = list(w.services.keys())
                self.triage_started = True
            if self.svc_queue:
                svc = self.svc_queue.pop(0)
                return (
                    f"Checking metrics for {svc} to identify if it's healthy or degraded.",
                    "get_metrics", {"service": svc}
                )
            self.phase = "investigate"

        # Phase 3: Investigate unhealthy services
        if self.phase == "investigate":
            for name, info in list(self.findings.items()):
                if info.get("status") != "healthy":
                    sub = info.get("_sub", "logs")
                    if sub == "logs":
                        info["_sub"] = "changes"
                        return (
                            f"{name} is {info['status']} (error_rate={info.get('error_rate',0)}, "
                            f"latency={info.get('latency_p99_ms',0)}ms, "
                            f"memory={info.get('memory_pct',0)}%). "
                            f"Let me pull its logs to find the root cause.",
                            "get_logs", {"service": name, "n": 15}
                        )
                    if sub == "changes":
                        info["_sub"] = "done"
                        self.investigated.add(name)
                        return (
                            f"Logs show anomalies on {name}. Now checking recent deploys "
                            f"and config changes — these are common incident triggers.",
                            "get_recent_changes", {"service": name}
                        )

            # Re-scan for any new unhealthy that appeared (cascading)
            for name, svc in w.services.items():
                if svc.status != "healthy" and name not in self.investigated:
                    self.investigated.add(name)
                    self.findings[name] = {"status": svc.status, "_sub": "done"}
                    return (
                        f"{name} has become {svc.status} — possible cascade. Checking logs.",
                        "get_logs", {"service": name, "n": 10}
                    )

            self.phase = "remediate"

        # Phase 4: Remediate
        if self.phase == "remediate":
            for name, svc in w.services.items():
                if svc.status == "healthy" or name in self.acted_on:
                    continue
                self.acted_on.add(name)

                info = self.findings.get(name, {})
                logs_text = " ".join(
                    l.get("msg", "") for l in svc.log_buffer[-20:]
                )

                # Decision tree with reasoning
                if svc.kind == "cache" and "cache_eviction" in logs_text:
                    return (
                        f"DIAGNOSIS: {name} is a cache service showing eviction storm. "
                        f"The cache is thrashing and dependents will start timing out. "
                        f"PLAN: Restart the cache pod to clear state and stop the storm.",
                        "restart_pod", {"service": name}
                    )

                if svc.kind == "database" and "conn_pool_active" in logs_text:
                    return (
                        f"DIAGNOSIS: {name} has exhausted its connection pool — "
                        f"queries are queueing up, clients are timing out. "
                        f"PLAN: Kill long-running queries to free pool slots.",
                        "kill_long_queries", {"service": name}
                    )

                if "tls_handshake_fail" in logs_text:
                    peer = None
                    for log in svc.log_buffer:
                        msg = log.get("msg", "")
                        if "peer=" in msg:
                            peer = msg.split("peer=")[1].split()[0]
                            break
                    if peer:
                        return (
                            f"DIAGNOSIS: TLS handshake failures between {name} and {peer}. "
                            f"Certificate has expired. "
                            f"PLAN: Rotate the certificate between the two services.",
                            "rotate_cert", {"service_a": name, "service_b": peer}
                        )

                if "retry_amplification" in logs_text or "retry_rate" in logs_text:
                    dep = ""
                    for log in svc.log_buffer:
                        msg = log.get("msg", "")
                        if "dep=" in msg:
                            dep = msg.split("dep=")[1].split()[0]
                            break
                    return (
                        f"DIAGNOSIS: {name} is caught in a retry amplification loop "
                        f"against {dep or 'a dependency'}. More retries = more load = more failures. "
                        f"PLAN: Enable circuit breaker to stop the cascade.",
                        "enable_circuit_breaker", {"service": name}
                    )

                if svc.config_flags:
                    for flag, val in svc.config_flags.items():
                        if val:
                            if "config_change" in logs_text or "intermittent" in logs_text:
                                return (
                                    f"DIAGNOSIS: {name} has config flag '{flag}' enabled and "
                                    f"shows intermittent errors — classic config drift pattern. "
                                    f"PLAN: Toggle the flag back off.",
                                    "toggle_feature_flag", {"service": name, "flag_name": flag}
                                )

                deploys = [c for c in svc.recent_changes if c.get("type") == "deploy"]
                if deploys and ("heap_used_pct" in logs_text or svc.memory_pct > 75):
                    return (
                        f"DIAGNOSIS: {name} has a recent deploy ({deploys[-1].get('id','?')}) "
                        f"and memory is climbing at {svc.memory_pct:.1f}% — likely a memory leak "
                        f"introduced by the new code. "
                        f"PLAN: Roll back the deploy.",
                        "rollback", {"service": name}
                    )

                if svc.kind == "cache":
                    return (
                        f"{name} is an unhealthy cache. Restarting to clear state.",
                        "restart_pod", {"service": name}
                    )

                return (
                    f"{name} is {svc.status} but root cause unclear. "
                    f"Attempting restart as a general recovery action.",
                    "restart_pod", {"service": name}
                )

            self.phase = "monitor"

        # Phase 5: Monitor — check recovery, retry remediation if needed
        if self.phase == "monitor":
            sick = [n for n, s in w.services.items() if s.status != "healthy"]
            if not sick:
                return (
                    "All services appear healthy. Verifying with a full status check.",
                    "list_services", {}
                )

            # Unacted sick services → back to remediate
            unacted = [n for n in sick if n not in self.acted_on]
            if unacted:
                self.phase = "remediate"
                return self.decide(w)

            # Retry alternative remediation for already-acted services
            for name in sick:
                retry_key = f"_retry_{name}"
                if self.findings.get(retry_key):
                    continue
                self.findings[retry_key] = True
                svc = w.services[name]
                logs_text = " ".join(
                    l.get("msg", "") for l in svc.log_buffer[-20:]
                )
                if "tls_handshake_fail" in logs_text:
                    for log in svc.log_buffer:
                        msg = log.get("msg", "")
                        if "peer=" in msg:
                            peer = msg.split("peer=")[1].split()[0]
                            return (
                                f"RETRY: {name} still showing TLS failures with {peer}. Rotating cert.",
                                "rotate_cert", {"service_a": name, "service_b": peer}
                            )
                if "retry_amplification" in logs_text or "retry_rate" in logs_text:
                    return (
                        f"RETRY: {name} still in retry storm. Enabling circuit breaker.",
                        "enable_circuit_breaker", {"service": name}
                    )
                if svc.config_flags:
                    for flag, val in svc.config_flags.items():
                        if val:
                            return (
                                f"RETRY: {name} has active flag '{flag}'. Toggling off.",
                                "toggle_feature_flag", {"service": name, "flag_name": flag}
                            )
                deploys = [c for c in svc.recent_changes if c.get("type") == "deploy"]
                if deploys and svc.memory_pct > 70:
                    return (
                        f"RETRY: {name} memory at {svc.memory_pct:.1f}% with recent deploy. Rolling back.",
                        "rollback", {"service": name}
                    )
                if svc.kind == "database":
                    return (
                        f"RETRY: {name} database still unhealthy. Killing long queries.",
                        "kill_long_queries", {"service": name}
                    )
                return (
                    f"RETRY: {name} still {svc.status}. Attempting restart.",
                    "restart_pod", {"service": name}
                )

            # All retries done — just monitor briefly
            self._mon_count = getattr(self, '_mon_count', 0) + 1
            target = sick[0]
            return (
                f"Waiting for recovery ({self._mon_count}). "
                f"{len(sick)} service(s) unhealthy. Checking {target}...",
                "get_metrics", {"service": target}
            )

        return ("No more actions to take.", None, None)

    def observe(self, tool_name, tool_args, result, obs):
        if tool_name == "get_metrics" and "service" in result:
            self.findings[result["service"]] = {**result, "_sub": "logs"}
        if tool_name == "list_services" and "services" in result:
            for s in result["services"]:
                if s["name"] not in self.findings:
                    self.findings[s["name"]] = {"status": s["status"], "_sub": "logs"}


class NaiveAgent:
    """Dumb agent: restarts everything it sees that isn't healthy."""

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
                return (
                    f"{name} is {svc.status}. Restarting it.",
                    "restart_pod", {"service": name}
                )
        # keep checking
        for name, svc in world.services.items():
            if svc.status != "healthy":
                return (f"Checking {name}...", "get_metrics", {"service": name})
        return ("Waiting...", "list_services", {})

    def observe(self, *a):
        pass


@app.get("/api/arena/run")
async def arena_run(seed: int = 42, difficulty: str = "easy"):
    """SSE — two agents race on the same incident, side by side."""
    async def generate():
        env_a, env_b = IncidentRoomEnv(), IncidentRoomEnv()
        obs_a = env_a.reset(seed=seed, difficulty=difficulty)
        obs_b = env_b.reset(seed=seed, difficulty=difficulty)

        yield _sse("reset", {
            "a": {"state": _snap(env_a)},
            "b": {"state": _snap(env_b)},
        })
        await asyncio.sleep(0.5)

        agent_a, agent_b = RuleBasedAgent(), NaiveAgent()
        done_a = done_b = False
        step = 0

        while (not done_a or not done_b) and step < 45:
            for lane, ag, ev, done_flag in [
                ("a", agent_a, env_a, done_a),
                ("b", agent_b, env_b, done_b),
            ]:
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

                yield _sse("action", {
                    "lane": lane, "step": step,
                    "tool": tool, "args": args,
                    "result": obs["tool_result"],
                    "state": _snap(ev),
                    "tick": obs["tick"], "done": is_done,
                })
                await asyncio.sleep(0.3)

                if lane == "a":
                    done_a = is_done
                else:
                    done_b = is_done

            step += 1

        grade_a = env_a.grade()
        grade_b = env_b.grade()
        yield _sse("done", {
            "a": {"grade": grade_a, "outcome": (obs_a := env_a.step("list_services", {}))["tick"].get("outcome", "timeout") if not done_a else "timeout", "state": _snap(env_a)},
            "b": {"grade": grade_b, "state": _snap(env_b)},
            "grade_a": grade_a,
            "grade_b": grade_b,
        })

    return StreamingResponse(generate(), media_type="text/event-stream")


def _snap(e: IncidentRoomEnv) -> dict:
    """Snapshot an env's world."""
    w = e.world
    return {
        "tick": w.tick,
        "max_ticks": w.max_ticks,
        "difficulty": w.difficulty,
        "seed": w.seed,
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
                "dependencies": s.dependencies,
                "region": s.region,
                "restart_cooldown": s.restart_cooldown,
                "history": w.metric_history.get(name, []),
            }
            for name, s in w.services.items()
        },
        "topology": w.topology,
        "faults_total": len(w.active_faults),
        "faults_resolved": sum(1 for f in w.active_faults if f.resolved),
    }


def _world_snapshot():
    return _snap(env)


def _enrich(obs):
    if env.world:
        w = env.world
        obs["state"] = {
            "tick": w.tick,
            "max_ticks": w.max_ticks,
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
    print("\n  IncidentRoom v2 Dashboard")
    print("  http://localhost:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
