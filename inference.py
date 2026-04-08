"""IncidentRoom inference — OpenAI function-calling agent loop.

Env vars:
  API_BASE_URL  — OpenAI-compatible endpoint (default https://api.openai.com/v1)
  MODEL_NAME    — model id (default gpt-4.1-mini)
  HF_TOKEN      — API key (required)
"""
from __future__ import annotations

import json
import os
import sys

from openai import OpenAI

from server.env import IncidentRoomEnv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
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

EPISODES = [
    {"seed": 42,   "difficulty": "easy"},
    {"seed": 137,  "difficulty": "medium"},
    {"seed": 256,  "difficulty": "hard"},
    {"seed": 7,    "difficulty": "easy"},
    {"seed": 1024, "difficulty": "medium"},
]

MAX_TURNS = 40  # safety cap per episode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable is required")

    client = OpenAI(base_url=api_base, api_key=hf_token)

    print("[START]", flush=True)
    try:
        total_score = 0.0
        for ep_idx, ep_cfg in enumerate(EPISODES):
            score = _run_episode(
                client, model, ep_idx,
                seed=ep_cfg["seed"],
                difficulty=ep_cfg["difficulty"],
            )
            total_score += score

        avg = total_score / max(len(EPISODES), 1)
        print(f"[STEP] summary avg_score={avg:.1f}", flush=True)
    finally:
        print("[END]", flush=True)


def _run_episode(
    client: OpenAI,
    model: str,
    ep_idx: int,
    seed: int,
    difficulty: str,
) -> float:
    env = IncidentRoomEnv()
    obs = env.reset(seed=seed, difficulty=difficulty)
    tools = env.get_tools()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(obs, default=str)},
    ]

    done = False
    step = 0
    tick_info = obs["tick"]

    while not done and step < MAX_TURNS:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # append assistant message
        messages.append(_msg_to_dict(msg))

        if not msg.tool_calls:
            # model emitted reasoning text — continue
            step += 1
            continue

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            result = env.step(fn_name, fn_args)
            tick_info = result["tick"]
            done = result["done"]

            print(
                f"[STEP] ep={ep_idx} step={step} tool={fn_name} "
                f"tick={tick_info['tick']} done={done}",
                flush=True,
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result["tool_result"], default=str),
            })

            if done:
                break

        step += 1

    score_info = env.grade()
    print(
        f"[STEP] ep={ep_idx} outcome={tick_info.get('outcome','?')} "
        f"score={score_info['score']:.1f}/{score_info['max_score']} "
        f"faults={score_info['faults_resolved']}/{score_info['faults_total']}",
        flush=True,
    )
    return score_info["score"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg_to_dict(msg) -> dict:
    """Convert an OpenAI ChatCompletionMessage to a plain dict for re-send."""
    d: dict = {"role": msg.role}
    if msg.content:
        d["content"] = msg.content
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return d


if __name__ == "__main__":
    main()
