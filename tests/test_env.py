"""Tests for the IncidentRoomEnv class (server/env.py).

Covers reset, step, tick advancement, episode termination, and grade.
"""
import pytest

from server.env import IncidentRoomEnv


# ---------------------------------------------------------------------------
# reset() returns valid observation
# ---------------------------------------------------------------------------

def test_reset_returns_dict():
    env = IncidentRoomEnv()
    obs = env.reset(seed=42, difficulty="easy")
    assert isinstance(obs, dict)
    env.close()


def test_reset_observation_has_expected_keys():
    env = IncidentRoomEnv()
    obs = env.reset(seed=42, difficulty="easy")
    assert "tool_result" in obs
    assert "tick" in obs
    assert "done" in obs
    assert "services_summary" in obs
    env.close()


def test_reset_tick_is_zero():
    env = IncidentRoomEnv()
    obs = env.reset(seed=42, difficulty="easy")
    assert obs["tick"]["tick"] == 0
    env.close()


def test_reset_done_is_false():
    env = IncidentRoomEnv()
    obs = env.reset(seed=42, difficulty="easy")
    assert obs["done"] is False
    env.close()


def test_reset_services_summary_not_empty():
    env = IncidentRoomEnv()
    obs = env.reset(seed=42, difficulty="easy")
    assert len(obs["services_summary"]) > 0
    env.close()


def test_reset_services_summary_structure():
    env = IncidentRoomEnv()
    obs = env.reset(seed=42, difficulty="easy")
    for entry in obs["services_summary"]:
        assert "name" in entry
        assert "status" in entry
        assert "health" in entry
    env.close()


@pytest.mark.parametrize("difficulty,expected_min_services", [
    ("easy", 3),
    ("medium", 6),
    ("hard", 8),
])
def test_reset_service_count_matches_difficulty(difficulty, expected_min_services):
    env = IncidentRoomEnv()
    obs = env.reset(seed=42, difficulty=difficulty)
    assert len(obs["services_summary"]) >= expected_min_services
    env.close()


# ---------------------------------------------------------------------------
# step() with unknown tool → error without crash
# ---------------------------------------------------------------------------

def test_step_unknown_tool_returns_error():
    env = IncidentRoomEnv()
    env.reset(seed=42, difficulty="easy")
    obs = env.step("nonexistent_tool", {})
    assert "error" in obs["tool_result"]
    env.close()


def test_step_unknown_tool_does_not_crash():
    env = IncidentRoomEnv()
    env.reset(seed=42, difficulty="easy")
    # Should not raise
    obs = env.step("totally_bogus", {"foo": "bar"})
    assert isinstance(obs, dict)
    env.close()


def test_step_unknown_tool_does_not_advance_tick():
    """Unknown tools should return current tick without advancing."""
    env = IncidentRoomEnv()
    env.reset(seed=42, difficulty="easy")
    obs = env.step("fake_tool", {})
    # The tick in the response should still be 0 (not advanced)
    assert obs["tick"]["tick"] == 0
    env.close()


# ---------------------------------------------------------------------------
# step() advances tick by 1
# ---------------------------------------------------------------------------

def test_step_advances_tick():
    env = IncidentRoomEnv()
    env.reset(seed=42, difficulty="easy")
    obs = env.step("list_services", {})
    assert obs["tick"]["tick"] == 1
    env.close()


def test_multiple_steps_advance_tick_sequentially():
    env = IncidentRoomEnv()
    env.reset(seed=42, difficulty="easy")
    for expected_tick in range(1, 4):
        obs = env.step("list_services", {})
        assert obs["tick"]["tick"] == expected_tick
    env.close()


def test_action_tool_also_advances_tick():
    env = IncidentRoomEnv()
    env.reset(seed=42, difficulty="easy")
    first_svc = list(env.world.services.keys())[0]
    obs = env.step("get_metrics", {"service": first_svc})
    assert obs["tick"]["tick"] == 1
    env.close()


# ---------------------------------------------------------------------------
# Episode terminates when tick budget exhausted
# ---------------------------------------------------------------------------

def test_episode_terminates_at_max_ticks():
    env = IncidentRoomEnv()
    env.reset(seed=42, difficulty="easy")
    max_ticks = env.world.max_ticks

    done = False
    tick_count = 0
    while not done and tick_count < max_ticks + 5:  # safety margin
        obs = env.step("list_services", {})
        done = obs["done"]
        tick_count += 1

    assert done, "Episode should terminate when tick budget is exhausted"
    assert env.world.tick <= max_ticks + 1  # tick increments then checks
    env.close()


def test_episode_done_outcome_is_timeout_on_budget_exhaust():
    """When no faults are resolved and ticks run out, outcome should be 'timeout'."""
    env = IncidentRoomEnv()
    env.reset(seed=42, difficulty="easy")
    max_ticks = env.world.max_ticks

    obs = None
    for _ in range(max_ticks + 5):
        obs = env.step("list_services", {})
        if obs["done"]:
            break

    assert obs is not None
    assert obs["done"] is True
    # Outcome should be timeout or catastrophic (not success, since we did nothing useful)
    assert obs["tick"]["outcome"] in ("timeout", "catastrophic")
    env.close()


# ---------------------------------------------------------------------------
# grade() returns expected structure
# ---------------------------------------------------------------------------

def test_grade_returns_score():
    env = IncidentRoomEnv()
    env.reset(seed=42, difficulty="easy")
    result = env.grade()
    assert "score" in result
    assert isinstance(result["score"], (int, float))
    env.close()


def test_grade_returns_faults_resolved():
    env = IncidentRoomEnv()
    env.reset(seed=42, difficulty="easy")
    result = env.grade()
    assert "faults_resolved" in result
    assert "faults_total" in result
    assert result["faults_resolved"] >= 0
    assert result["faults_total"] >= 1  # easy has at least 1 fault
    env.close()


def test_grade_faults_resolved_leq_total():
    env = IncidentRoomEnv()
    env.reset(seed=42, difficulty="easy")
    result = env.grade()
    assert result["faults_resolved"] <= result["faults_total"]
    env.close()


# ---------------------------------------------------------------------------
# get_tools() returns schemas
# ---------------------------------------------------------------------------

def test_get_tools_returns_list():
    env = IncidentRoomEnv()
    tools = env.get_tools()
    assert isinstance(tools, list)
    assert len(tools) == 13


def test_get_tools_schema_structure():
    env = IncidentRoomEnv()
    tools = env.get_tools()
    for tool in tools:
        assert "type" in tool
        assert tool["type"] == "function"
        assert "function" in tool
        assert "name" in tool["function"]
        assert "parameters" in tool["function"]


# ---------------------------------------------------------------------------
# state() returns expected keys
# ---------------------------------------------------------------------------

def test_state_returns_expected_keys():
    env = IncidentRoomEnv()
    env.reset(seed=42, difficulty="easy")
    st = env.state()
    expected_keys = {"tick", "max_ticks", "difficulty", "seed", "services",
                     "faults_total", "faults_resolved"}
    assert expected_keys.issubset(st.keys())
    env.close()


def test_state_before_reset_returns_error():
    env = IncidentRoomEnv()
    st = env.state()
    assert "error" in st


# ---------------------------------------------------------------------------
# close() cleans up
# ---------------------------------------------------------------------------

def test_close_clears_world():
    env = IncidentRoomEnv()
    env.reset(seed=42, difficulty="easy")
    assert env.world is not None
    env.close()
    assert env.world is None


# ---------------------------------------------------------------------------
# Multiple resets
# ---------------------------------------------------------------------------

def test_multiple_resets_work():
    env = IncidentRoomEnv()
    for seed in [1, 2, 3]:
        obs = env.reset(seed=seed, difficulty="easy")
        assert obs["done"] is False
        assert env.world is not None
        assert env.world.seed == seed
    env.close()


# ---------------------------------------------------------------------------
# Full episode round-trip
# ---------------------------------------------------------------------------

def test_full_episode_roundtrip():
    """Run a complete episode: reset → steps → grade."""
    env = IncidentRoomEnv()
    obs = env.reset(seed=42, difficulty="easy")
    assert not obs["done"]

    # Take some steps
    for _ in range(3):
        obs = env.step("list_services", {})
        if obs["done"]:
            break

    # Grade (even mid-episode)
    result = env.grade()
    assert "score" in result
    assert 0 <= result["score"] <= 100

    env.close()
