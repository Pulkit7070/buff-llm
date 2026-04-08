"""Pure grading function over the final World state.

No string matching, no synonym handling, no LLM judge — just arithmetic
on observable simulation state.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server.world import World

from server.world import health


def grade(world: World) -> dict:
    """Score the episode.  Returns dict with score, max_score, breakdown."""
    n_faults = len(world.active_faults)
    n_resolved = sum(1 for f in world.active_faults if f.resolved)

    # ---- 1. Fault resolution  (40 pts) ----
    fault_score = (n_resolved / max(n_faults, 1)) * 40.0

    # ---- 2. Final service health  (25 pts) ----
    if world.services:
        avg_health = sum(health(s) for s in world.services.values()) / len(world.services)
    else:
        avg_health = 1.0
    health_score = avg_health * 25.0

    # ---- 3. User-impact efficiency  (20 pts) ----
    max_impact = world.max_ticks * sum(s.criticality for s in world.services.values())
    if max_impact > 0:
        impact_ratio = world.user_impact_total / max_impact
    else:
        impact_ratio = 0.0
    impact_score = max(0.0, 1.0 - impact_ratio) * 20.0

    # ---- 4. Time efficiency  (15 pts) — only if all faults resolved ----
    if n_faults > 0 and n_resolved == n_faults:
        tick_ratio = world.tick / max(world.max_ticks, 1)
        efficiency_score = (1.0 - tick_ratio) * 15.0
    else:
        efficiency_score = 0.0

    total = fault_score + health_score + impact_score + efficiency_score

    return {
        "score": round(total, 1),
        "max_score": 100,
        "breakdown": {
            "fault_resolution": round(fault_score, 1),
            "service_health": round(health_score, 1),
            "user_impact": round(impact_score, 1),
            "efficiency": round(efficiency_score, 1),
        },
        "faults_resolved": n_resolved,
        "faults_total": n_faults,
        "final_tick": world.tick,
        "user_impact_total": round(world.user_impact_total, 2),
    }
