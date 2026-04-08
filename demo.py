"""IncidentRoom local demo — rule-based agent, no API key needed.

Run:  python demo.py
"""
from __future__ import annotations

import json
import sys
import os

# Enable UTF-8 output on Windows
if sys.platform == "win32":
    os.system("")  # enable ANSI on Windows
    sys.stdout.reconfigure(encoding="utf-8")

from server.env import IncidentRoomEnv

# ANSI colors
G = "\033[92m"  # green
Y = "\033[93m"  # yellow
R = "\033[91m"  # red
C = "\033[96m"  # cyan
B = "\033[1m"   # bold
RST = "\033[0m"


def color_status(status: str) -> str:
    if status == "healthy":
        return f"{G}{status}{RST}"
    if status == "degraded":
        return f"{Y}{status}{RST}"
    return f"{R}{status}{RST}"


def banner(text: str) -> None:
    print(f"\n{B}{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}{RST}\n")


def print_services(obs: dict) -> None:
    for s in obs["services_summary"]:
        bar_len = int(s["health"] * 20)
        bar = f"{'#' * bar_len}{'.' * (20 - bar_len)}"
        print(f"  {s['name']:25s} {color_status(s['status']):20s} [{bar}] {s['health']:.1%}")


def print_tool(name: str, args: dict, result: dict) -> None:
    args_str = ", ".join(f"{k}={v}" for k, v in args.items())
    print(f"  {C}> {name}({args_str}){RST}")
    if "error" in result:
        print(f"    {R}x {result['error']}{RST}")
    elif "logs" in result:
        for log in result["logs"][-5:]:
            lvl = log["level"]
            c = R if lvl == "ERROR" else Y if lvl == "WARN" else RST
            print(f"    {c}{log['msg']}{RST}")
    elif "services" in result:
        for s in result["services"]:
            print(f"    {s['name']:25s} {s['kind']:10s} {color_status(s['status'])}")
    elif "topology" in result:
        for name_, info in result["topology"].items():
            deps = ", ".join(info.get("dependencies", []))
            print(f"    {name_:25s} deps=[{deps}]")
    elif "changes" in result:
        for ch in result["changes"]:
            print(f"    {ch}")
        if result.get("config_flags"):
            print(f"    flags={result['config_flags']}")
    else:
        for k, v in result.items():
            print(f"    {k}: {v}")


def step(env: IncidentRoomEnv, tool: str, args: dict | None = None) -> dict:
    """Execute one tool call, print results, return observation."""
    args = args or {}
    obs = env.step(tool, args)
    print_tool(tool, args, obs["tool_result"])
    return obs


def rule_based_agent(env: IncidentRoomEnv) -> None:
    """A simple rule-based SRE agent that demonstrates the workflow."""
    obs = env.reset(seed=42, difficulty="easy")
    print(f"  {B}Difficulty:{RST} easy  |  {B}Tick budget:{RST} {env.world.max_ticks}  |  {B}Faults:{RST} {len(env.world.active_faults)}")
    print(f"\n  {B}Initial service status:{RST}")
    print_services(obs)

    # --- Phase 1: Reconnaissance ---
    banner("PHASE 1: Reconnaissance")

    print(f"  {B}Step 1 — List all services{RST}")
    obs = step(env, "list_services")

    print(f"\n  {B}Step 2 — Get topology{RST}")
    obs = step(env, "get_topology")

    # --- Phase 2: Triage ---
    banner("PHASE 2: Triage — check metrics for each service")

    unhealthy = []
    for svc in env.world.services.values():
        obs = step(env, "get_metrics", {"service": svc.name})
        m = obs["tool_result"]
        if m.get("status") != "healthy":
            unhealthy.append(m)

    if not unhealthy:
        print(f"\n  {G}All services healthy — nothing to do!{RST}")
        return

    print(f"\n  {B}Unhealthy services:{RST}")
    for m in unhealthy:
        print(f"    {R}{m['service']}: status={m['status']} "
              f"err={m['error_rate']} lat={m['latency_p99_ms']}ms "
              f"mem={m['memory_pct']}% cpu={m['cpu_pct']}%{RST}")

    # --- Phase 3: Deep investigation ---
    banner("PHASE 3: Investigation — logs & recent changes")

    target = unhealthy[0]["service"]
    print(f"  {B}Investigating: {target}{RST}\n")

    print(f"  {B}Logs:{RST}")
    obs = step(env, "get_logs", {"service": target, "n": 10})

    print(f"\n  {B}Recent changes:{RST}")
    obs = step(env, "get_recent_changes", {"service": target})
    changes = obs["tool_result"]

    # --- Phase 4: Remediation ---
    banner("PHASE 4: Remediation")

    svc = env.world.services[target]

    # Decision tree
    if svc.kind == "cache":
        action, args = "restart_pod", {"service": target}
        reason = "Cache eviction detected — restarting cache service"
    elif svc.kind == "database" and svc.latency_p99 > 200:
        action, args = "kill_long_queries", {"service": target}
        reason = "DB pool saturated — killing long queries"
    elif changes.get("config_flags"):
        flag = list(changes["config_flags"].keys())[0]
        action, args = "toggle_feature_flag", {"service": target, "flag_name": flag}
        reason = f"Config drift detected — toggling {flag}"
    elif any(c.get("type") == "deploy" for c in changes.get("changes", [])):
        action, args = "rollback", {"service": target}
        reason = "Recent deploy found — rolling back"
    elif svc.latency_p99 > 300:
        action, args = "enable_circuit_breaker", {"service": target}
        reason = "High latency with retries — enabling circuit breaker"
    else:
        action, args = "restart_pod", {"service": target}
        reason = "Fallback — restarting service"

    print(f"  {B}Decision:{RST} {reason}\n")
    obs = step(env, action, args)

    # --- Phase 5: Monitor ---
    banner("PHASE 5: Monitor recovery")

    while not obs["done"]:
        obs = step(env, "get_metrics", {"service": target})
        m = obs["tool_result"]
        print(f"    tick={obs['tick']['tick']} status={color_status(m.get('status','?'))} health={m.get('health','?')}")
        if obs["done"]:
            break

    # --- Results ---
    banner("EPISODE COMPLETE")
    outcome = obs["tick"]["outcome"]
    color = G if outcome == "success" else R
    print(f"  {B}Outcome:{RST} {color}{outcome}{RST}")
    print(f"  {B}Final tick:{RST} {obs['tick']['tick']}/{env.world.max_ticks}")
    print()
    print(f"  {B}Final service status:{RST}")
    print_services(obs)

    score = env.grade()
    print(f"\n  {B}Score: {G}{score['score']:.1f}{RST} / {score['max_score']}")
    print(f"  {B}Breakdown:{RST}")
    for k, v in score["breakdown"].items():
        bar = "#" * int(v) + "." * (int(score["max_score"] / 4) - int(v))
        print(f"    {k:25s} {v:5.1f}  [{bar}]")


def run_all_difficulties() -> None:
    """Run the agent across easy, medium, hard."""
    for diff, seed in [("easy", 42), ("medium", 137), ("hard", 256)]:
        banner(f"EPISODE: {diff.upper()} (seed={seed})")
        env = IncidentRoomEnv()
        obs = env.reset(seed=seed, difficulty=diff)
        print(f"  Services: {len(env.world.services)}")
        print(f"  Faults:   {len(env.world.active_faults)}")
        for f in env.world.active_faults:
            print(f"    -> {type(f).__name__} on {f.target}")
        print()

        # Quick automated run
        done = False
        tick = 0
        # recon
        obs = env.step("list_services", {})
        obs = env.step("get_topology", {})

        # investigate each service
        for name in list(env.world.services):
            obs = env.step("get_metrics", {"service": name})
            if obs["done"]:
                break

        if not obs["done"]:
            # try to fix each fault
            for fault in env.world.active_faults:
                t = fault.target
                obs = env.step("get_logs", {"service": t, "n": 10})
                if obs["done"]:
                    break
                obs = env.step("get_recent_changes", {"service": t})
                if obs["done"]:
                    break

                svc = env.world.services[t]
                if hasattr(fault, "flag_name") and fault.flag_name:
                    obs = env.step("toggle_feature_flag", {"service": t, "flag_name": fault.flag_name})
                elif hasattr(fault, "service_b") and fault.service_b:
                    obs = env.step("rotate_cert", {"service_a": t, "service_b": fault.service_b})
                elif hasattr(fault, "deploy_id") and fault.deploy_id:
                    obs = env.step("rollback", {"service": t})
                elif hasattr(fault, "ext_dep") and fault.ext_dep:
                    obs = env.step("enable_circuit_breaker", {"service": t})
                elif svc.kind == "cache":
                    obs = env.step("restart_pod", {"service": t})
                elif svc.kind == "database":
                    obs = env.step("kill_long_queries", {"service": t})
                else:
                    obs = env.step("restart_pod", {"service": t})
                if obs["done"]:
                    break

        # drain remaining ticks
        while not obs["done"]:
            obs = env.step("list_services", {})

        score = env.grade()
        outcome = obs["tick"]["outcome"]
        color = G if outcome == "success" else R
        print(f"  {B}Outcome:{RST} {color}{outcome}{RST}  |  "
              f"{B}Score:{RST} {G}{score['score']:.1f}{RST}/{score['max_score']}  |  "
              f"Faults fixed: {score['faults_resolved']}/{score['faults_total']}")
        print()


if __name__ == "__main__":
    if "--all" in sys.argv:
        run_all_difficulties()
    else:
        banner("IncidentRoom v2 — Local Demo")
        print("  A rule-based SRE agent walks through one easy episode.\n"
              "  Run with --all to see easy/medium/hard.\n")
        rule_based_agent(IncidentRoomEnv())
