"""Tests for the tool handlers in server/tools.py.

Verifies error handling (invalid service, wrong kind), expected side-effects
(cooldown, deploy removal), and read-tool output structure.
"""
import pytest

from server.world import World, Service, health
from server.tools import (
    list_services,
    get_metrics,
    get_logs,
    get_topology,
    get_recent_changes,
    restart_pod,
    rollback,
    scale_up,
    toggle_feature_flag,
    enable_circuit_breaker,
    drain_region,
    kill_long_queries,
    rotate_cert,
    TOOL_HANDLERS,
    TOOL_SCHEMAS,
    READ_TOOLS,
    ACTION_TOOLS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service(name="svc-a", kind="api", **overrides):
    defaults = dict(
        name=name,
        kind=kind,
        baseline_error_rate=0.01,
        baseline_latency=10.0,
        baseline_cpu=15.0,
        baseline_memory=30.0,
    )
    defaults.update(overrides)
    svc = Service(**defaults)
    svc.error_rate = svc.baseline_error_rate
    svc.latency_p99 = svc.baseline_latency
    svc.cpu_pct = svc.baseline_cpu
    svc.memory_pct = svc.baseline_memory
    return svc


def _make_world(services_list, topology=None, seed=42):
    services = {s.name: s for s in services_list}
    if topology is None:
        topology = {s.name: [] for s in services_list}
    return World(
        tick=0,
        services=services,
        topology=topology,
        max_ticks=20,
        seed=seed,
        difficulty="easy",
    )


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------

def test_all_schemas_have_handlers():
    """Every tool in TOOL_SCHEMAS must have a matching handler."""
    schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    handler_names = set(TOOL_HANDLERS.keys())
    assert schema_names == handler_names


def test_read_and_action_partition():
    """READ_TOOLS + ACTION_TOOLS must cover all handlers with no overlap."""
    assert READ_TOOLS | ACTION_TOOLS == set(TOOL_HANDLERS.keys())
    assert READ_TOOLS & ACTION_TOOLS == set()


# ---------------------------------------------------------------------------
# Invalid service name → error dict (not crash)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool_fn", [
    get_metrics,
    get_logs,
    get_recent_changes,
    restart_pod,
    rollback,
    scale_up,
    toggle_feature_flag,
    enable_circuit_breaker,
    kill_long_queries,
])
def test_invalid_service_returns_error(tool_fn):
    """Passing a nonexistent service name must return a dict with 'error'."""
    world = _make_world([_make_service("real-svc")])
    # Build kwargs depending on the function
    kwargs = {"service": "nonexistent-svc"}
    if tool_fn is toggle_feature_flag:
        kwargs["flag_name"] = "some_flag"
    result = tool_fn(world, **kwargs)
    assert "error" in result, f"{tool_fn.__name__} should return error for bad service"


def test_rotate_cert_invalid_service_a():
    svc = _make_service("svc-a")
    world = _make_world([svc])
    result = rotate_cert(world, service_a="nonexistent", service_b="svc-a")
    assert "error" in result


def test_rotate_cert_invalid_service_b():
    svc = _make_service("svc-a")
    world = _make_world([svc])
    result = rotate_cert(world, service_a="svc-a", service_b="nonexistent")
    assert "error" in result


def test_drain_region_invalid_region():
    world = _make_world([_make_service("svc-a")])
    result = drain_region(world, region="ap-south-1")
    assert "error" in result


# ---------------------------------------------------------------------------
# restart_pod sets cooldown
# ---------------------------------------------------------------------------

def test_restart_pod_sets_cooldown():
    svc = _make_service("api-svc")
    world = _make_world([svc])
    result = restart_pod(world, service="api-svc")
    assert result.get("success") is True
    assert svc.restart_cooldown == 2


def test_restart_pod_rejects_during_cooldown():
    svc = _make_service("api-svc")
    world = _make_world([svc])
    restart_pod(world, service="api-svc")
    result = restart_pod(world, service="api-svc")
    assert "error" in result


def test_restart_pod_temporarily_spikes_errors():
    svc = _make_service("api-svc")
    world = _make_world([svc])
    err_before = svc.error_rate
    restart_pod(world, service="api-svc")
    assert svc.error_rate > err_before, "restart should spike error rate temporarily"


# ---------------------------------------------------------------------------
# rollback removes deploy
# ---------------------------------------------------------------------------

def test_rollback_removes_deploy():
    svc = _make_service("api-svc")
    svc.recent_changes.append({"type": "deploy", "id": "d-1234", "version": "v1.0.0", "tick": -1})
    world = _make_world([svc])

    assert len([c for c in svc.recent_changes if c["type"] == "deploy"]) == 1
    result = rollback(world, service="api-svc")
    assert result.get("success") is True
    assert len([c for c in svc.recent_changes if c["type"] == "deploy"]) == 0


def test_rollback_no_deploy_returns_error():
    svc = _make_service("api-svc")
    world = _make_world([svc])
    result = rollback(world, service="api-svc")
    assert "error" in result


# ---------------------------------------------------------------------------
# kill_long_queries rejects non-database
# ---------------------------------------------------------------------------

def test_kill_long_queries_rejects_non_database():
    svc = _make_service("api-svc", kind="api")
    world = _make_world([svc])
    result = kill_long_queries(world, service="api-svc")
    assert "error" in result
    assert "not a database" in result["error"]


def test_kill_long_queries_works_on_database():
    svc = _make_service("pg-primary", kind="database")
    svc.latency_p99 = 500.0
    svc.cpu_pct = 80.0
    world = _make_world([svc])
    result = kill_long_queries(world, service="pg-primary")
    assert result.get("success") is True
    assert svc.latency_p99 < 500.0, "latency should decrease after kill"
    assert svc.cpu_pct < 80.0, "cpu should decrease after kill"


# ---------------------------------------------------------------------------
# rotate_cert requires both services exist
# ---------------------------------------------------------------------------

def test_rotate_cert_both_exist_succeeds():
    svc_a = _make_service("gw", kind="gateway")
    svc_b = _make_service("auth", kind="api")
    world = _make_world([svc_a, svc_b])
    result = rotate_cert(world, service_a="gw", service_b="auth")
    assert result.get("success") is True


def test_rotate_cert_missing_first_fails():
    svc_b = _make_service("auth", kind="api")
    world = _make_world([svc_b])
    result = rotate_cert(world, service_a="missing", service_b="auth")
    assert "error" in result


def test_rotate_cert_missing_second_fails():
    svc_a = _make_service("gw", kind="gateway")
    world = _make_world([svc_a])
    result = rotate_cert(world, service_a="gw", service_b="missing")
    assert "error" in result


# ---------------------------------------------------------------------------
# Read tools return expected structure
# ---------------------------------------------------------------------------

def test_list_services_structure():
    svc = _make_service("api-svc")
    world = _make_world([svc])
    result = list_services(world)
    assert "services" in result
    assert isinstance(result["services"], list)
    assert len(result["services"]) == 1
    entry = result["services"][0]
    assert set(entry.keys()) == {"name", "kind", "status", "region"}


def test_get_metrics_structure():
    svc = _make_service("api-svc")
    world = _make_world([svc])
    result = get_metrics(world, service="api-svc")
    expected_keys = {"service", "status", "error_rate", "latency_p99_ms",
                     "cpu_pct", "memory_pct", "health"}
    assert expected_keys.issubset(result.keys())


def test_get_logs_structure():
    svc = _make_service("api-svc")
    svc.log_buffer.append({"tick": 0, "level": "INFO", "msg": "test"})
    world = _make_world([svc])
    result = get_logs(world, service="api-svc")
    assert "logs" in result
    assert isinstance(result["logs"], list)
    assert len(result["logs"]) == 1


def test_get_logs_respects_n():
    svc = _make_service("api-svc")
    for i in range(30):
        svc.log_buffer.append({"tick": i, "level": "INFO", "msg": f"line {i}"})
    world = _make_world([svc])
    result = get_logs(world, service="api-svc", n=5)
    assert len(result["logs"]) == 5


def test_get_topology_structure():
    svc_a = _make_service("gw", kind="gateway")
    svc_b = _make_service("api", kind="api")
    world = _make_world(
        [svc_a, svc_b],
        topology={"gw": ["api"], "api": []},
    )
    svc_b.dependencies = ["gw"]
    result = get_topology(world)
    assert "topology" in result
    assert "gw" in result["topology"]
    assert result["topology"]["gw"]["dependents"] == ["api"]


def test_get_recent_changes_structure():
    svc = _make_service("api-svc")
    svc.recent_changes.append({"type": "deploy", "id": "d-0001"})
    svc.config_flags["flag_001"] = True
    world = _make_world([svc])
    result = get_recent_changes(world, service="api-svc")
    assert "changes" in result
    assert "config_flags" in result
    assert len(result["changes"]) == 1
    assert result["config_flags"]["flag_001"] is True


# ---------------------------------------------------------------------------
# scale_up reduces cpu/latency
# ---------------------------------------------------------------------------

def test_scale_up_reduces_cpu_and_latency():
    svc = _make_service("api-svc")
    svc.cpu_pct = 80.0
    svc.latency_p99 = 200.0
    svc.error_rate = 0.3  # unhealthy, so no wasted action
    world = _make_world([svc])
    result = scale_up(world, service="api-svc")
    assert result.get("success") is True
    assert svc.cpu_pct < 80.0
    assert svc.latency_p99 < 200.0


# ---------------------------------------------------------------------------
# toggle_feature_flag
# ---------------------------------------------------------------------------

def test_toggle_feature_flag_toggles_value():
    svc = _make_service("api-svc")
    svc.config_flags["my_flag"] = True
    world = _make_world([svc])
    result = toggle_feature_flag(world, service="api-svc", flag_name="my_flag")
    assert result.get("success") is True
    assert svc.config_flags["my_flag"] is False


def test_toggle_feature_flag_nonexistent_flag():
    svc = _make_service("api-svc")
    world = _make_world([svc])
    result = toggle_feature_flag(world, service="api-svc", flag_name="nonexistent")
    assert "error" in result


# ---------------------------------------------------------------------------
# enable_circuit_breaker
# ---------------------------------------------------------------------------

def test_enable_circuit_breaker_on_healthy_wastes_action():
    svc = _make_service("api-svc")
    # Service is healthy at baseline
    world = _make_world([svc])
    wasted_before = world.wasted_actions
    enable_circuit_breaker(world, service="api-svc")
    assert world.wasted_actions > wasted_before


# ---------------------------------------------------------------------------
# drain_region
# ---------------------------------------------------------------------------

def test_drain_region_affects_correct_services():
    svc_east = _make_service("east-svc", region="us-east-1")
    svc_west = _make_service("west-svc", region="us-west-2")
    world = _make_world([svc_east, svc_west])
    err_east_before = svc_east.error_rate
    err_west_before = svc_west.error_rate

    result = drain_region(world, region="us-east-1")
    assert result.get("success") is True
    assert svc_east.error_rate > err_east_before
    assert svc_west.error_rate == err_west_before  # unaffected
