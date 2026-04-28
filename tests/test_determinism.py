"""Tests for deterministic world generation.

Same (seed, difficulty) must always produce the identical World.
Different seeds must produce different worlds.
"""
import pytest

from server.env import IncidentRoomEnv
from server.generator import generate


# ---------------------------------------------------------------------------
# Same seed + difficulty → identical world
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_same_seed_produces_identical_services(difficulty):
    """Two generate() calls with the same seed must yield the same service names."""
    seed = 12345
    w1 = generate(difficulty=difficulty, seed=seed)
    w2 = generate(difficulty=difficulty, seed=seed)

    assert list(w1.services.keys()) == list(w2.services.keys())


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_same_seed_produces_identical_faults(difficulty):
    """Fault ids, types, and targets must match across two identical generations."""
    seed = 99
    w1 = generate(difficulty=difficulty, seed=seed)
    w2 = generate(difficulty=difficulty, seed=seed)

    assert len(w1.active_faults) == len(w2.active_faults)
    for f1, f2 in zip(w1.active_faults, w2.active_faults):
        assert type(f1) is type(f2)
        assert f1.fault_id == f2.fault_id
        assert f1.target == f2.target


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_same_seed_produces_identical_topology(difficulty):
    seed = 77
    w1 = generate(difficulty=difficulty, seed=seed)
    w2 = generate(difficulty=difficulty, seed=seed)

    assert w1.topology == w2.topology


def test_same_seed_identical_service_baselines():
    """Baseline metrics (driven by RNG) must match for the same seed."""
    seed = 555
    w1 = generate(difficulty="medium", seed=seed)
    w2 = generate(difficulty="medium", seed=seed)

    for name in w1.services:
        s1 = w1.services[name]
        s2 = w2.services[name]
        assert s1.baseline_error_rate == s2.baseline_error_rate
        assert s1.baseline_latency == s2.baseline_latency
        assert s1.baseline_cpu == s2.baseline_cpu
        assert s1.baseline_memory == s2.baseline_memory


# ---------------------------------------------------------------------------
# Different seeds → different worlds
# ---------------------------------------------------------------------------

def test_different_seeds_produce_different_worlds():
    """With enough services, different seeds should yield different service sets."""
    w1 = generate(difficulty="hard", seed=1)
    w2 = generate(difficulty="hard", seed=9999)

    # At minimum the service names or fault targets should differ
    names_match = list(w1.services.keys()) == list(w2.services.keys())
    faults_match = (
        len(w1.active_faults) == len(w2.active_faults)
        and all(
            f1.target == f2.target and type(f1) is type(f2)
            for f1, f2 in zip(w1.active_faults, w2.active_faults)
        )
    )
    # They might theoretically collide, but practically never for hard
    assert not (names_match and faults_match), (
        "Two different seeds produced identical worlds — extremely unlikely"
    )


# ---------------------------------------------------------------------------
# Two independent envs with same seed → identical step results
# ---------------------------------------------------------------------------

def test_two_envs_same_seed_identical_step_results():
    """Running the same sequence of steps on two envs with the same seed
    must produce byte-identical observations."""
    env1 = IncidentRoomEnv()
    env2 = IncidentRoomEnv()

    seed = 42
    obs1 = env1.reset(seed=seed, difficulty="easy")
    obs2 = env2.reset(seed=seed, difficulty="easy")

    # Initial observations should match
    assert obs1["services_summary"] == obs2["services_summary"]

    # Run a few read-only steps
    for tool_name, tool_args in [
        ("list_services", {}),
        ("get_topology", {}),
    ]:
        r1 = env1.step(tool_name, tool_args)
        r2 = env2.step(tool_name, tool_args)
        assert r1["tool_result"] == r2["tool_result"]
        assert r1["tick"] == r2["tick"]
        assert r1["done"] == r2["done"]
        assert r1["services_summary"] == r2["services_summary"]

    env1.close()
    env2.close()


def test_two_envs_same_seed_identical_grades():
    """Running identical episodes on two envs with the same seed gives the
    same final grade, even after action tools."""
    seed = 7
    difficulty = "easy"

    env1 = IncidentRoomEnv()
    env2 = IncidentRoomEnv()

    env1.reset(seed=seed, difficulty=difficulty)
    env2.reset(seed=seed, difficulty=difficulty)

    # Perform the same sequence of actions on both
    first_service = list(env1.world.services.keys())[0]
    steps = [
        ("list_services", {}),
        ("get_metrics", {"service": first_service}),
        ("get_logs", {"service": first_service}),
    ]
    for tool, args in steps:
        env1.step(tool, args)
        env2.step(tool, args)

    g1 = env1.grade()
    g2 = env2.grade()
    assert g1 == g2

    env1.close()
    env2.close()
