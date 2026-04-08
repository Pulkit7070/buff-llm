"""World model for the IncidentRoom tick-based simulator.

The World is the single source of truth: services, topology, faults, and
a tick counter.  Every mutation flows through tick_world() or through
tool handlers.  No LLM, no randomness at runtime — fully deterministic
given the same seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

@dataclass
class Service:
    name: str
    kind: str  # gateway | api | cache | database | worker | queue
    status: str = "healthy"  # healthy | degraded | failing | down
    error_rate: float = 0.01
    latency_p99: float = 10.0  # ms
    cpu_pct: float = 15.0
    memory_pct: float = 30.0
    dependencies: list[str] = field(default_factory=list)
    recent_changes: list[dict] = field(default_factory=list)
    pending_effects: list[tuple] = field(default_factory=list)  # (fire_at_tick, callable)
    criticality: float = 0.5  # 0‑1, weight in user-impact calc
    config_flags: dict[str, Any] = field(default_factory=dict)
    region: str = "us-east-1"
    restart_cooldown: int = 0
    log_buffer: list[dict] = field(default_factory=list)
    # baselines (set by generator, used for healing / restart recovery)
    baseline_error_rate: float = 0.01
    baseline_latency: float = 10.0
    baseline_cpu: float = 15.0
    baseline_memory: float = 30.0
    # transient — reset every tick
    _cascade_error: float = field(default=0.0, repr=False)


def health(svc: Service) -> float:
    """Scalar health 0.0–1.0 derived from the four key metrics."""
    err = min(svc.error_rate / 0.5, 1.0)
    lat = min(max(svc.latency_p99 - 100, 0) / 900.0, 1.0)
    cpu = min(max(svc.cpu_pct - 80, 0) / 20.0, 1.0)
    mem = min(max(svc.memory_pct - 85, 0) / 15.0, 1.0)
    return max(0.0, 1.0 - max(err, lat, cpu, mem))


def recompute_status(svc: Service) -> None:
    h = health(svc)
    if h >= 0.8:
        svc.status = "healthy"
    elif h >= 0.5:
        svc.status = "degraded"
    elif h >= 0.2:
        svc.status = "failing"
    else:
        svc.status = "down"


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

@dataclass
class World:
    tick: int = 0
    services: dict[str, Service] = field(default_factory=dict)
    topology: dict[str, list[str]] = field(default_factory=dict)  # svc -> dependents
    user_impact_total: float = 0.0
    wasted_actions: int = 0
    max_ticks: int = 20
    seed: int = 0
    active_faults: list[Any] = field(default_factory=list)  # list[Fault]
    difficulty: str = "easy"
    action_log: list[dict] = field(default_factory=list)
    metric_history: dict[str, list[dict]] = field(default_factory=dict)  # svc -> [{tick,h,e,l,c,m}]


# ---------------------------------------------------------------------------
# tick
# ---------------------------------------------------------------------------

def tick_world(world: World) -> dict:
    """Advance the simulation by one tick.  Returns terminal info dict."""
    world.tick += 1

    # 1 — pending effects
    for svc in world.services.values():
        keep: list[tuple] = []
        for fire_at, fn in svc.pending_effects:
            if world.tick >= fire_at:
                fn(world)
            else:
                keep.append((fire_at, fn))
        svc.pending_effects = keep
        if svc.restart_cooldown > 0:
            svc.restart_cooldown -= 1

    # 2 — fault progression
    for fault in world.active_faults:
        if not fault.resolved:
            fault.progress(world)

    # 3 — natural healing for services NOT under active fault
    _heal(world)

    # 4 — cascade: degraded services push errors to dependents
    for src_name, dependents in world.topology.items():
        src = world.services.get(src_name)
        if src is None:
            continue
        if src.status in ("degraded", "failing", "down"):
            for dep_name in dependents:
                dep = world.services.get(dep_name)
                if dep:
                    dep._cascade_error += src.error_rate * 0.3

    for svc in world.services.values():
        if svc._cascade_error > 0:
            svc.error_rate = min(1.0, svc.error_rate + svc._cascade_error)
            svc.latency_p99 += svc._cascade_error * 150
        svc._cascade_error = 0.0

    # 5 — recompute status
    for svc in world.services.values():
        recompute_status(svc)

    # 6 — user impact
    world.user_impact_total += sum(
        svc.criticality * (1.0 - health(svc))
        for svc in world.services.values()
    )

    # 7 — synthetic logs
    _emit_logs(world)

    # 7b — record metric history
    for svc in world.services.values():
        if svc.name not in world.metric_history:
            world.metric_history[svc.name] = []
        world.metric_history[svc.name].append({
            "t": world.tick,
            "h": round(health(svc), 3),
            "e": round(svc.error_rate, 4),
            "l": round(svc.latency_p99, 1),
            "c": round(svc.cpu_pct, 1),
            "m": round(svc.memory_pct, 1),
        })

    # 8 — terminal conditions
    all_healthy = all(s.status == "healthy" for s in world.services.values())
    all_resolved = all(f.resolved for f in world.active_faults)
    n_down = sum(1 for s in world.services.values() if s.status == "down")
    catastrophic = n_down > len(world.services) * 0.8

    if all_healthy and all_resolved:
        return {"done": True, "outcome": "success", "tick": world.tick}
    if world.tick >= world.max_ticks:
        return {"done": True, "outcome": "timeout", "tick": world.tick}
    if catastrophic:
        return {"done": True, "outcome": "catastrophic", "tick": world.tick}
    return {"done": False, "outcome": "running", "tick": world.tick}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _heal(world: World) -> None:
    """Slowly recover services that have no unresolved fault targeting them."""
    faulty = {f.target for f in world.active_faults if not f.resolved}
    rate = 0.15  # 15 % of gap per tick
    for svc in world.services.values():
        if svc.name in faulty or svc.restart_cooldown > 0:
            continue
        svc.error_rate += (svc.baseline_error_rate - svc.error_rate) * rate
        svc.latency_p99 += (svc.baseline_latency - svc.latency_p99) * rate
        svc.cpu_pct += (svc.baseline_cpu - svc.cpu_pct) * rate
        svc.memory_pct += (svc.baseline_memory - svc.memory_pct) * rate


def _emit_logs(world: World) -> None:
    """Append mechanical, metric-based log lines for the current tick."""
    t = world.tick
    tag = f"T+{t:03d}"
    for svc in world.services.values():
        svc.log_buffer.append(
            {"tick": t, "level": "INFO",
             "msg": f"{tag} svc={svc.name} heartbeat status={svc.status}"}
        )
        if svc.error_rate > 0.05:
            svc.log_buffer.append(
                {"tick": t, "level": "ERROR",
                 "msg": f"{tag} svc={svc.name} metric=error_rate val={svc.error_rate:.4f} threshold=0.0500"}
            )
        if svc.memory_pct > 75:
            svc.log_buffer.append(
                {"tick": t, "level": "WARN",
                 "msg": f"{tag} svc={svc.name} metric=memory_pct val={svc.memory_pct:.1f} threshold=75.0"}
            )
        if svc.latency_p99 > 150:
            svc.log_buffer.append(
                {"tick": t, "level": "WARN",
                 "msg": f"{tag} svc={svc.name} metric=latency_p99 val={svc.latency_p99:.1f}ms threshold=150.0ms"}
            )
        if svc.cpu_pct > 70:
            svc.log_buffer.append(
                {"tick": t, "level": "WARN",
                 "msg": f"{tag} svc={svc.name} metric=cpu_pct val={svc.cpu_pct:.1f} threshold=70.0"}
            )
        if svc.restart_cooldown > 0:
            svc.log_buffer.append(
                {"tick": t, "level": "INFO",
                 "msg": f"{tag} svc={svc.name} event=pod_restart cooldown={svc.restart_cooldown}"}
            )
        # trim to last 100
        if len(svc.log_buffer) > 100:
            svc.log_buffer = svc.log_buffer[-100:]
