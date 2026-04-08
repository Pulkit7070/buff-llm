"""IncidentRoom inference — OpenAI function-calling agent loop.

Mandatory env vars:
  API_BASE_URL  — OpenAI-compatible endpoint
  MODEL_NAME    — model id
  HF_TOKEN      — API key

Stdout format follows OpenEnv spec: [START], [STEP], [END].
Each task returns score in [0.0, 1.0].
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

from openai import OpenAI

from server.env import IncidentRoomEnv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
BENCHMARK = "incidentroom"
MAX_TURNS = 40
TEMPERATURE = 0.7

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

# 3+ tasks with distinct difficulties (seed, difficulty)
TASKS = [
    {"name": "sre-incident-easy",   "seed": 42,   "difficulty": "easy"},
    {"name": "sre-incident-medium", "seed": 137,  "difficulty": "medium"},
    {"name": "sre-incident-hard",   "seed": 256,  "difficulty": "hard"},
]


# ---------------------------------------------------------------------------
# Logging helpers (strict OpenEnv format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(
        f"[END] success={success_val} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Agent loop for one task
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, model: str, task_cfg: dict) -> float:
    """Run a single task episode. Returns normalized score in [0, 1]."""
    task_name = task_cfg["name"]
    seed = task_cfg["seed"]
    difficulty = task_cfg["difficulty"]

    env = IncidentRoomEnv()
    obs = env.reset(seed=seed, difficulty=difficulty)
    tools = env.get_tools()

    log_start(task=task_name, env=BENCHMARK, model=model)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(obs, default=str)},
    ]

    step_num = 0
    done = False
    rewards: List[float] = []
    last_error: Optional[str] = None

    try:
        while not done and step_num < MAX_TURNS:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=TEMPERATURE,
                )
            except Exception as e:
                last_error = str(e)
                step_num += 1
                rewards.append(0.0)
                log_step(step=step_num, action="api_error", reward=0.0, done=False, error=last_error)
                break

            msg = response.choices[0].message
            messages.append(_msg_to_dict(msg))

            if not msg.tool_calls:
                # Model emitted reasoning only — continue
                step_num += 1
                continue

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                result = env.step(fn_name, fn_args)
                done = result["done"]
                step_num += 1

                # Compute reward: 0 for intermediate steps, normalized score on done
                if done:
                    grade_info = env.grade()
                    reward = round(grade_info["score"] / 100.0, 2)
                else:
                    reward = 0.0

                rewards.append(reward)
                last_error = result["tool_result"].get("error")

                # Format action string
                args_str = ",".join(f"{k}={v}" for k, v in fn_args.items())
                action_str = f"{fn_name}({args_str})"

                log_step(
                    step=step_num,
                    action=action_str,
                    reward=reward,
                    done=done,
                    error=last_error,
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result["tool_result"], default=str),
                })

                if done:
                    break

        # Final grading
        grade_info = env.grade()
        score = round(grade_info["score"] / 100.0, 2)  # normalize to [0, 1]
        success = grade_info["faults_resolved"] == grade_info["faults_total"]

        # If we didn't get a done step, add the final reward
        if not done:
            rewards.append(score)
            step_num += 1
            log_step(
                step=step_num,
                action="timeout",
                reward=score,
                done=True,
                error=None,
            )

    except Exception as e:
        score = 0.0
        success = False
        if not rewards:
            rewards = [0.0]
        last_error = str(e)

    finally:
        log_end(success=success, steps=len(rewards), score=score, rewards=rewards)
        env.close()

    return score


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN environment variable is required")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task_cfg in TASKS:
        run_task(client, MODEL_NAME, task_cfg)


if __name__ == "__main__":
    main()
