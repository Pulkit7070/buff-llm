"""Tests for the grading system (server/grader.py).

Score must be in [0, 100], components must sum to total, and the
efficiency bonus is only available when all faults are resolved.
"""
import pytest

from server.env import IncidentRoomEnv
from server.generator import generate
from server.grader import grade
from server.world import World, Service, tick_world


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_to_completion(env):
    """Step with list_services until the episode ends."""
    while True:
        obs = env.step("list_services", {})
        if obs["done"]:
            return obs


def _make_minimal_world(**overrides):
    """Build a tiny world with a single healthy service and no faults."""
    svc = Service(
        name="test-svc",
        kind="api",
        baseline_error_rate=0.01,
        baseline_latency=10.0,
        baseline_cpu=15.0,
        baseline_memory=30.0,
    )
    svc.error_rate = svc.baseline_error_rate
    svc.latency_p99 = svc.baseline_latency
    svc.cpu_pct = svc.baseline_cpu
    svc.memory_pct = svc.baseline_memory
    defaults = dict(
        tick=0,
        services={"test-svc": svc},
        topology={"test-svc": []},
        max_ticks=15,
        seed=1,
        difficulty="easy",
    )
    defaults.update(overrides)
    return World(**defaults)


# ---------------------------------------------------------------------------
# Score bounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_score_in_valid_range(difficulty):
    """Score must be between 0 and 100 inclusive."""
    env = IncidentRoomEnv()
    env.reset(seed=42, difficulty=difficulty)
    _run_to_completion(env)
    result = env.grade()
    assert 0 <= result["score"] <= 100
    env.close()


@pytest.mark.parametrize("seed", [1, 42, 999, 31415])
def test_score_range_across_seeds(seed):
    env = IncidentRoomEnv()
    env.reset(seed=seed, difficulty="easy")
    _run_to_completion(env)
    result = env.grade()
    assert 0 <= result["score"] <= 100
    env.close()


# ---------------------------------------------------------------------------
# Component breakdown sums to total
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_breakdown_sums_to_total(difficulty):
    """The four breakdown components must sum to the total score."""
    env = IncidentRoomEnv()
    env.reset(seed=42, difficulty=difficulty)
    _run_to_completion(env)
    result = env.grade()

    bd = result["breakdown"]
    component_sum = (
        bd["fault_resolution"]
        + bd["service_health"]
        + bd["user_impact"]
        + bd["efficiency"]
    )
    assert abs(component_sum - result["score"]) < 0.2, (
        f"Components sum to {component_sum} but score is {result['score']}"
    )
    env.close()


# ---------------------------------------------------------------------------
# Perfect run → high score
# ---------------------------------------------------------------------------

def test_perfect_run_high_score():
    """Resolving all faults quickly with healthy services yields a high score."""
    world = _make_minimal_world()
    # No faults, all healthy, tick 0 — this is as perfect as it gets
    result = grade(world)
    # With 0 faults, fault_resolution = 0/max(0,1)*40 = 0,
    # health = 25, impact = 20, efficiency = 0 (0 faults means no bonus)
    assert result["score"] >= 40, f"Expected high score, got {result['score']}"


def test_all_faults_resolved_early_gives_efficiency_bonus():
    """When all faults are resolved and world tick is low, efficiency > 0."""
    env = IncidentRoomEnv()
    env.reset(seed=42, difficulty="easy")

    # Find the fault and fix it
    world = env.world
    for fault in world.active_faults:
        fault.resolved = True

    # Advance just one tick so world.tick = 1 (much less than max_ticks)
    tick_world(world)

    result = grade(world)
    assert result["breakdown"]["efficiency"] > 0, (
        "Efficiency bonus should be positive when all faults resolved early"
    )
    env.close()


# ---------------------------------------------------------------------------
# Zero-effort run → low score
# ---------------------------------------------------------------------------

def test_zero_effort_low_score():
    """Running out the tick budget with no actions gives a low score."""
    env = IncidentRoomEnv()
    env.reset(seed=42, difficulty="easy")
    _run_to_completion(env)
    result = env.grade()

    # With unresolved faults, score should be low
    assert result["faults_resolved"] == 0
    # The score will be hurt by unresolved faults and degraded health
    assert result["breakdown"]["fault_resolution"] == 0
    env.close()


# ---------------------------------------------------------------------------
# Efficiency bonus requires all faults resolved
# ---------------------------------------------------------------------------

def test_efficiency_zero_when_faults_unresolved():
    """If any fault is unresolved, efficiency must be 0."""
    env = IncidentRoomEnv()
    env.reset(seed=42, difficulty="easy")

    # Do nothing; just let it time out
    _run_to_completion(env)
    result = env.grade()

    assert result["faults_resolved"] < result["faults_total"]
    assert result["breakdown"]["efficiency"] == 0, (
        "Efficiency should be 0 when faults remain unresolved"
    )
    env.close()


def test_efficiency_positive_only_when_all_resolved():
    """Manually resolve one fault of two and verify efficiency = 0,
    then resolve both and verify efficiency > 0."""
    env = IncidentRoomEnv()
    env.reset(seed=42, difficulty="hard")
    world = env.world

    # Resolve first fault only
    if len(world.active_faults) >= 2:
        world.active_faults[0].resolved = True
        tick_world(world)
        partial = grade(world)
        assert partial["breakdown"]["efficiency"] == 0

        # Now resolve all
        for f in world.active_faults:
            f.resolved = True
        tick_world(world)
        full = grade(world)
        assert full["breakdown"]["efficiency"] > 0
    else:
        # If only 1 fault, just resolve it
        world.active_faults[0].resolved = True
        tick_world(world)
        full = grade(world)
        assert full["breakdown"]["efficiency"] > 0

    env.close()


# ---------------------------------------------------------------------------
# Grade output structure
# ---------------------------------------------------------------------------

def test_grade_returns_expected_keys():
    """The grade dict must contain all required keys."""
    env = IncidentRoomEnv()
    env.reset(seed=42, difficulty="easy")
    _run_to_completion(env)
    result = env.grade()

    expected_keys = {"score", "max_score", "breakdown", "faults_resolved",
                     "faults_total", "final_tick", "user_impact_total"}
    assert expected_keys.issubset(result.keys())

    bd_keys = {"fault_resolution", "service_health", "user_impact", "efficiency"}
    assert bd_keys.issubset(result["breakdown"].keys())
    env.close()


def test_max_score_is_100():
    env = IncidentRoomEnv()
    env.reset(seed=42, difficulty="easy")
    result = env.grade()
    assert result["max_score"] == 100
    env.close()
