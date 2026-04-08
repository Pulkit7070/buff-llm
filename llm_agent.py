"""IncidentRoom LLM agent — async generators for real LLM agent runs.

Three modes:
  run_llm_agent()          — standard (non-streaming) function-calling loop
  run_llm_agent_streaming() — token-by-token streaming with live reasoning
  run_hitl_agent()         — human-in-the-loop: proposes actions, waits for approval

All yield ``{"event": str, "data": dict}`` dicts suitable for SSE.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, AsyncGenerator

from openai import AsyncOpenAI

from server.env import IncidentRoomEnv
from server.world import health
from cost import calc_cost

MAX_TURNS = 40

DEFAULT_SYSTEM_PROMPT = """\
You are an expert SRE responding to a live incident in a microservice system.
The system degrades every tick.  Your goal: diagnose and remediate all faults
before the tick budget runs out.

Strategy:
1. list_services + get_topology to understand the architecture.
2. get_metrics / get_logs on unhealthy services to find root causes.
3. get_recent_changes to check for suspicious deploys or config changes.
4. Take targeted action ONLY after you understand the root cause.
5. Wrong actions have real consequences — they can make things worse.
6. Every tool call (even reads) costs one tick.  Be efficient.
"""


# ── Shared helpers ────────────────────────────────────────────────────

def _snap_env(env: IncidentRoomEnv) -> dict:
    w = env.world
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


def _msg_to_dict(msg) -> dict:
    d: dict = {"role": msg.role}
    if msg.content:
        d["content"] = msg.content
    if msg.tool_calls:
        d["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return d


def _prompt_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def _token_info(sp, sc, tp, tc):
    return {"prompt": sp, "completion": sc, "total_prompt": tp, "total_completion": tc}


def _make_env(scenario_yaml: str | None, seed: int, difficulty: str):
    """Create env and reset — either from scenario YAML or standard generator."""
    env = IncidentRoomEnv()
    if scenario_yaml:
        from scenarios import build_world_from_scenario
        env.world = build_world_from_scenario(scenario_yaml, seed=seed)
        obs = env._observation(
            tool_result={"message": "Incident detected. Begin investigation."},
            tick_info={"done": False, "outcome": "running", "tick": 0},
        )
    else:
        obs = env.reset(seed=seed, difficulty=difficulty)
    return env, obs


# ── 1. Standard (non-streaming) agent ─────────────────────────────────

async def run_llm_agent(
    api_key: str, api_base_url: str, model: str,
    system_prompt: str | None,
    seed: int, difficulty: str,
    scenario_yaml: str | None = None,
) -> AsyncGenerator[dict, None]:
    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    client = AsyncOpenAI(base_url=api_base_url, api_key=api_key)
    env, obs = _make_env(scenario_yaml, seed, difficulty)
    tools = env.get_tools()

    yield {"event": "reset", "data": {"observation": obs, "state": _snap_env(env)}}

    messages: list[dict] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(obs, default=str)},
    ]

    done = False
    step = 0
    tp = tc = 0
    cost = 0.0
    last_result: dict[str, Any] = {}
    events_log: list[dict] = []

    while not done and step < MAX_TURNS:
        t0 = time.monotonic()
        try:
            response = await client.chat.completions.create(
                model=model, messages=messages, tools=tools, tool_choice="auto",
            )
        except Exception as e:
            err = str(e)
            lo = err.lower()
            et = ("auth" if "401" in err or "api key" in lo else
                  "rate_limit" if "429" in err else
                  "not_found" if "404" in err else "api_error")
            ev = {"event": "error", "data": {"step": step, "error_type": et, "message": err}}
            events_log.append(ev)
            yield ev
            return

        lat = round((time.monotonic() - t0) * 1000)
        msg = response.choices[0].message
        u = response.usage
        sp = u.prompt_tokens if u else 0
        sc = u.completion_tokens if u else 0
        tp += sp; tc += sc
        cost = calc_cost(model, tp, tc)
        tok = {**_token_info(sp, sc, tp, tc), "cost_usd": cost}

        messages.append(_msg_to_dict(msg))

        if msg.content:
            ev = {"event": "thinking", "data": {
                "step": step, "text": msg.content, "tokens": tok, "latency_ms": lat}}
            events_log.append(ev)
            yield ev

        if not msg.tool_calls:
            step += 1
            continue

        for t in msg.tool_calls:
            fn = t.function.name
            try:
                args = json.loads(t.function.arguments)
            except json.JSONDecodeError:
                args = {}
            last_result = env.step(fn, args)
            done = last_result["done"]
            ev = {"event": "tool_call", "data": {
                "step": step, "tool": fn, "args": args,
                "result": last_result["tool_result"],
                "state": _snap_env(env), "tick": last_result["tick"],
                "done": done, "tokens": tok, "latency_ms": lat}}
            events_log.append(ev)
            yield ev
            messages.append({"role": "tool", "tool_call_id": t.id,
                             "content": json.dumps(last_result["tool_result"], default=str)})
            if done:
                break
        step += 1

    grade = env.grade()
    outcome = last_result.get("tick", {}).get("outcome", "timeout")
    ev = {"event": "done", "data": {
        "grade": grade, "outcome": outcome, "state": _snap_env(env),
        "tokens": {**_token_info(0, 0, tp, tc), "cost_usd": cost},
        "run_meta": {"model": model, "seed": seed, "difficulty": difficulty,
                     "agent_type": "llm", "system_prompt_hash": _prompt_hash(prompt),
                     "events": events_log}}}
    yield ev


# ── 2. Streaming agent (token-by-token) ──────────────────────────────

async def run_llm_agent_streaming(
    api_key: str, api_base_url: str, model: str,
    system_prompt: str | None,
    seed: int, difficulty: str,
    scenario_yaml: str | None = None,
) -> AsyncGenerator[dict, None]:
    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    client = AsyncOpenAI(base_url=api_base_url, api_key=api_key)
    env, obs = _make_env(scenario_yaml, seed, difficulty)
    tools = env.get_tools()

    yield {"event": "reset", "data": {"observation": obs, "state": _snap_env(env)}}

    messages: list[dict] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(obs, default=str)},
    ]

    done = False
    step = 0
    tp = tc = 0
    cost = 0.0
    last_result: dict[str, Any] = {}

    while not done and step < MAX_TURNS:
        t0 = time.monotonic()
        try:
            stream = await client.chat.completions.create(
                model=model, messages=messages, tools=tools,
                tool_choice="auto", stream=True,
                stream_options={"include_usage": True},
            )
        except Exception as e:
            err = str(e)
            lo = err.lower()
            et = ("auth" if "401" in err or "api key" in lo else
                  "rate_limit" if "429" in err else "api_error")
            yield {"event": "error", "data": {"step": step, "error_type": et, "message": err}}
            return

        content_parts: list[str] = []
        tool_calls_acc: dict[int, dict] = {}
        finish_reason = None
        sp = sc = 0

        async for chunk in stream:
            if chunk.usage:
                sp = chunk.usage.prompt_tokens
                sc = chunk.usage.completion_tokens

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason or finish_reason

            if delta and delta.content:
                content_parts.append(delta.content)
                yield {"event": "token", "data": {"step": step, "token": delta.content}}

            if delta and delta.tool_calls:
                for tc_d in delta.tool_calls:
                    idx = tc_d.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": tc_d.id or "", "name": "", "args": ""}
                    if tc_d.id:
                        tool_calls_acc[idx]["id"] = tc_d.id
                    if tc_d.function:
                        if tc_d.function.name:
                            tool_calls_acc[idx]["name"] = tc_d.function.name
                        if tc_d.function.arguments:
                            tool_calls_acc[idx]["args"] += tc_d.function.arguments

        lat = round((time.monotonic() - t0) * 1000)
        tp += sp; tc += sc
        cost = calc_cost(model, tp, tc)
        tok = {**_token_info(sp, sc, tp, tc), "cost_usd": cost}

        full_content = "".join(content_parts) or None

        # Build assistant message for history
        asst_msg: dict = {"role": "assistant"}
        if full_content:
            asst_msg["content"] = full_content
        if tool_calls_acc:
            asst_msg["tool_calls"] = [
                {"id": v["id"], "type": "function",
                 "function": {"name": v["name"], "arguments": v["args"]}}
                for v in tool_calls_acc.values()
            ]
        messages.append(asst_msg)

        if full_content:
            yield {"event": "thinking", "data": {
                "step": step, "text": full_content, "tokens": tok, "latency_ms": lat,
                "streamed": True}}

        if not tool_calls_acc:
            step += 1
            continue

        for v in tool_calls_acc.values():
            fn = v["name"]
            try:
                args = json.loads(v["args"])
            except json.JSONDecodeError:
                args = {}
            last_result = env.step(fn, args)
            done = last_result["done"]
            yield {"event": "tool_call", "data": {
                "step": step, "tool": fn, "args": args,
                "result": last_result["tool_result"],
                "state": _snap_env(env), "tick": last_result["tick"],
                "done": done, "tokens": tok, "latency_ms": lat}}
            messages.append({"role": "tool", "tool_call_id": v["id"],
                             "content": json.dumps(last_result["tool_result"], default=str)})
            if done:
                break
        step += 1

    grade = env.grade()
    outcome = last_result.get("tick", {}).get("outcome", "timeout")
    yield {"event": "done", "data": {
        "grade": grade, "outcome": outcome, "state": _snap_env(env),
        "tokens": {**_token_info(0, 0, tp, tc), "cost_usd": cost},
        "run_meta": {"model": model, "seed": seed, "difficulty": difficulty,
                     "agent_type": "llm-stream",
                     "system_prompt_hash": _prompt_hash(prompt)}}}


# ── 3. Human-in-the-Loop agent ───────────────────────────────────────

_hitl_sessions: dict[str, dict] = {}


def hitl_respond(session_id: str, action: str, tool: str | None = None,
                 args: dict | None = None):
    """Called from webapp when user approves/rejects/edits a proposal."""
    sess = _hitl_sessions.get(session_id)
    if not sess:
        return False
    sess["response"] = {"action": action, "tool": tool, "args": args}
    sess["event"].set()
    return True


async def run_hitl_agent(
    session_id: str,
    api_key: str, api_base_url: str, model: str,
    system_prompt: str | None,
    seed: int, difficulty: str,
    scenario_yaml: str | None = None,
) -> AsyncGenerator[dict, None]:
    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    client = AsyncOpenAI(base_url=api_base_url, api_key=api_key)
    env, obs = _make_env(scenario_yaml, seed, difficulty)
    tools = env.get_tools()

    yield {"event": "reset", "data": {"observation": obs, "state": _snap_env(env)}}

    messages: list[dict] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(obs, default=str)},
    ]

    done = False
    step = 0
    tp = tc = 0
    cost = 0.0
    last_result: dict[str, Any] = {}

    try:
        while not done and step < MAX_TURNS:
            t0 = time.monotonic()
            try:
                response = await client.chat.completions.create(
                    model=model, messages=messages, tools=tools, tool_choice="auto",
                )
            except Exception as e:
                yield {"event": "error", "data": {"step": step, "error_type": "api_error", "message": str(e)}}
                return

            lat = round((time.monotonic() - t0) * 1000)
            msg = response.choices[0].message
            u = response.usage
            sp = u.prompt_tokens if u else 0
            sc_tok = u.completion_tokens if u else 0
            tp += sp; tc += sc_tok
            cost = calc_cost(model, tp, tc)
            tok = {**_token_info(sp, sc_tok, tp, tc), "cost_usd": cost}

            messages.append(_msg_to_dict(msg))

            if msg.content:
                yield {"event": "thinking", "data": {
                    "step": step, "text": msg.content, "tokens": tok, "latency_ms": lat}}

            if not msg.tool_calls:
                step += 1
                continue

            for t_call in msg.tool_calls:
                fn = t_call.function.name
                try:
                    fn_args = json.loads(t_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                # Yield proposal and wait for human
                yield {"event": "proposal", "data": {
                    "session_id": session_id, "step": step,
                    "tool": fn, "args": fn_args,
                    "reasoning": msg.content or "", "tokens": tok}}

                evt = asyncio.Event()
                _hitl_sessions[session_id] = {"event": evt, "response": None}
                try:
                    await asyncio.wait_for(evt.wait(), timeout=300)
                except asyncio.TimeoutError:
                    yield {"event": "error", "data": {"step": step, "error_type": "timeout",
                                                      "message": "No response within 5 minutes"}}
                    return

                resp = _hitl_sessions.pop(session_id, {}).get("response", {})
                action = resp.get("action", "approve") if resp else "approve"

                if action == "reject":
                    messages.append({"role": "tool", "tool_call_id": t_call.id,
                                     "content": json.dumps({"error": "Action rejected by operator"})})
                    yield {"event": "rejected", "data": {"step": step, "tool": fn, "args": fn_args}}
                    continue

                if action == "edit":
                    fn = resp.get("tool") or fn
                    fn_args = resp.get("args") or fn_args

                last_result = env.step(fn, fn_args)
                done = last_result["done"]
                yield {"event": "tool_call", "data": {
                    "step": step, "tool": fn, "args": fn_args,
                    "result": last_result["tool_result"],
                    "state": _snap_env(env), "tick": last_result["tick"],
                    "done": done, "tokens": tok, "latency_ms": lat}}
                messages.append({"role": "tool", "tool_call_id": t_call.id,
                                 "content": json.dumps(last_result["tool_result"], default=str)})
                if done:
                    break
            step += 1

        grade = env.grade()
        outcome = last_result.get("tick", {}).get("outcome", "timeout")
        yield {"event": "done", "data": {
            "grade": grade, "outcome": outcome, "state": _snap_env(env),
            "tokens": {**_token_info(0, 0, tp, tc), "cost_usd": cost},
            "run_meta": {"model": model, "seed": seed, "difficulty": difficulty,
                         "agent_type": "hitl",
                         "system_prompt_hash": _prompt_hash(prompt)}}}
    finally:
        _hitl_sessions.pop(session_id, None)
