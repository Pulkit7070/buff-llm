"""Tool definitions and handlers for the IncidentRoom simulator.

13 tools total: 5 read (no side-effects) + 8 action (mutate the World).
Each handler is a plain function(world, **kwargs) → dict.
TOOL_SCHEMAS holds the OpenAI function-calling JSON for every tool.
"""
from __future__ import annotations

from typing import Any
from server.world import World, health, recompute_status


# ===================================================================
# Read tools
# ===================================================================

def list_services(world: World, **kw) -> dict:
    return {
        "services": [
            {"name": s.name, "kind": s.kind, "status": s.status, "region": s.region}
            for s in world.services.values()
        ]
    }


def get_metrics(world: World, *, service: str, **kw) -> dict:
    svc = world.services.get(service)
    if not svc:
        return {"error": f"service '{service}' not found"}
    return {
        "service": svc.name,
        "status": svc.status,
        "error_rate": round(svc.error_rate, 4),
        "latency_p99_ms": round(svc.latency_p99, 1),
        "cpu_pct": round(svc.cpu_pct, 1),
        "memory_pct": round(svc.memory_pct, 1),
        "health": round(health(svc), 3),
    }


def get_logs(world: World, *, service: str, n: int = 20, **kw) -> dict:
    svc = world.services.get(service)
    if not svc:
        return {"error": f"service '{service}' not found"}
    return {"service": service, "logs": svc.log_buffer[-int(n):]}


def get_topology(world: World, **kw) -> dict:
    return {
        "topology": {
            name: {
                "kind": world.services[name].kind,
                "dependencies": world.services[name].dependencies,
                "dependents": deps,
            }
            for name, deps in world.topology.items()
            if name in world.services
        }
    }


def get_recent_changes(world: World, *, service: str, **kw) -> dict:
    svc = world.services.get(service)
    if not svc:
        return {"error": f"service '{service}' not found"}
    return {
        "service": service,
        "changes": svc.recent_changes,
        "config_flags": svc.config_flags,
    }


# ===================================================================
# Action tools
# ===================================================================

def restart_pod(world: World, *, service: str, **kw) -> dict:
    svc = world.services.get(service)
    if not svc:
        return {"error": f"service '{service}' not found"}
    if svc.restart_cooldown > 0:
        return {"error": f"'{service}' already restarting (cooldown={svc.restart_cooldown})"}

    svc.restart_cooldown = 2
    svc.error_rate = min(1.0, svc.error_rate + 0.5)
    svc.status = "failing"

    target_name = service  # capture for closure

    def _finish(w: World) -> None:
        s = w.services.get(target_name)
        if s:
            s.error_rate = s.baseline_error_rate
            s.latency_p99 = s.baseline_latency
            s.cpu_pct = s.baseline_cpu
            s.memory_pct = s.baseline_memory
            s.restart_cooldown = 0
            recompute_status(s)

    svc.pending_effects.append((world.tick + 3, _finish))
    svc.log_buffer.append({
        "tick": world.tick, "level": "INFO",
        "msg": f"T+{world.tick:03d} svc={svc.name} event=pod_restart initiated",
    })
    return {"success": True, "message": f"restart initiated for {service}, cooldown=2 ticks"}


def rollback(world: World, *, service: str, **kw) -> dict:
    svc = world.services.get(service)
    if not svc:
        return {"error": f"service '{service}' not found"}

    deploys = [c for c in svc.recent_changes if c.get("type") == "deploy"]
    if not deploys:
        return {"error": f"no recent deploys on {service}"}

    removed = deploys[-1]
    svc.recent_changes.remove(removed)
    svc.log_buffer.append({
        "tick": world.tick, "level": "INFO",
        "msg": f"T+{world.tick:03d} svc={svc.name} event=rollback deploy_id={removed.get('id', '?')}",
    })
    return {"success": True, "message": f"rolled back deploy {removed.get('id')} on {service}"}


def scale_up(world: World, *, service: str, **kw) -> dict:
    svc = world.services.get(service)
    if not svc:
        return {"error": f"service '{service}' not found"}

    was_healthy = health(svc) > 0.8
    svc.cpu_pct = max(5, svc.cpu_pct * 0.6)
    svc.latency_p99 = max(svc.baseline_latency, svc.latency_p99 * 0.5)
    svc.log_buffer.append({
        "tick": world.tick, "level": "INFO",
        "msg": f"T+{world.tick:03d} svc={svc.name} event=scale_up replicas_added=2",
    })
    if was_healthy:
        world.wasted_actions += 1
        return {"success": True, "message": f"scaled up {service} (warning: over-provisioning)"}
    return {"success": True, "message": f"scaled up {service}"}


def toggle_feature_flag(world: World, *, service: str, flag_name: str, **kw) -> dict:
    svc = world.services.get(service)
    if not svc:
        return {"error": f"service '{service}' not found"}
    if flag_name not in svc.config_flags:
        return {"error": f"flag '{flag_name}' not found on {service}"}

    svc.config_flags[flag_name] = not svc.config_flags[flag_name]
    svc.log_buffer.append({
        "tick": world.tick, "level": "INFO",
        "msg": (f"T+{world.tick:03d} svc={svc.name} "
                f"event=flag_toggle key={flag_name} val={svc.config_flags[flag_name]}"),
    })
    return {"success": True, "message": f"toggled {flag_name} on {service}"}


def enable_circuit_breaker(world: World, *, service: str, **kw) -> dict:
    svc = world.services.get(service)
    if not svc:
        return {"error": f"service '{service}' not found"}

    svc.log_buffer.append({
        "tick": world.tick, "level": "INFO",
        "msg": f"T+{world.tick:03d} svc={svc.name} event=circuit_breaker_enabled",
    })
    if health(svc) > 0.8:
        svc.error_rate = min(1.0, svc.error_rate + 0.10)
        world.wasted_actions += 1
        return {"success": True, "message": f"circuit breaker on {service} (warning: cuts healthy traffic)"}
    return {"success": True, "message": f"circuit breaker enabled on {service}"}


def drain_region(world: World, *, region: str, **kw) -> dict:
    affected = [s for s in world.services.values() if s.region == region]
    if not affected:
        return {"error": f"no services in region '{region}'"}
    for svc in affected:
        svc.error_rate = min(1.0, svc.error_rate + 0.3)
        svc.log_buffer.append({
            "tick": world.tick, "level": "WARN",
            "msg": f"T+{world.tick:03d} svc={svc.name} event=region_drain region={region}",
        })
    world.user_impact_total += len(affected) * 2.0
    return {"success": True, "message": f"drained {region}, {len(affected)} services affected"}


def kill_long_queries(world: World, *, service: str, **kw) -> dict:
    svc = world.services.get(service)
    if not svc:
        return {"error": f"service '{service}' not found"}
    if svc.kind != "database":
        return {"error": f"{service} is not a database service"}

    svc.latency_p99 = max(svc.baseline_latency, svc.latency_p99 * 0.3)
    svc.cpu_pct = max(svc.baseline_cpu, svc.cpu_pct * 0.5)
    svc.log_buffer.append({
        "tick": world.tick, "level": "INFO",
        "msg": f"T+{world.tick:03d} svc={svc.name} event=kill_long_queries terminated=12",
    })
    return {"success": True, "message": f"killed long-running queries on {service}"}


def rotate_cert(world: World, *, service_a: str, service_b: str, **kw) -> dict:
    for name in (service_a, service_b):
        if name not in world.services:
            return {"error": f"service '{name}' not found"}
    svc = world.services[service_a]
    svc.log_buffer.append({
        "tick": world.tick, "level": "INFO",
        "msg": f"T+{world.tick:03d} svc={svc.name} event=cert_rotated peer={service_b}",
    })
    return {"success": True, "message": f"rotated cert between {service_a} and {service_b}"}


# ===================================================================
# Registry
# ===================================================================

TOOL_HANDLERS: dict[str, Any] = {
    "list_services":        list_services,
    "get_metrics":          get_metrics,
    "get_logs":             get_logs,
    "get_topology":         get_topology,
    "get_recent_changes":   get_recent_changes,
    "restart_pod":          restart_pod,
    "rollback":             rollback,
    "scale_up":             scale_up,
    "toggle_feature_flag":  toggle_feature_flag,
    "enable_circuit_breaker": enable_circuit_breaker,
    "drain_region":         drain_region,
    "kill_long_queries":    kill_long_queries,
    "rotate_cert":          rotate_cert,
}

READ_TOOLS = {"list_services", "get_metrics", "get_logs", "get_topology", "get_recent_changes"}
ACTION_TOOLS = set(TOOL_HANDLERS) - READ_TOOLS

# ===================================================================
# OpenAI function-calling schemas
# ===================================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_services",
            "description": "List all services with name, kind, status, and region.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_metrics",
            "description": "Get numeric metrics for a service: error_rate, latency_p99_ms, cpu_pct, memory_pct, health.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"},
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_logs",
            "description": "Get the last N log entries for a service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"},
                    "n": {"type": "integer", "description": "Number of log lines (default 20)"},
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_topology",
            "description": "Get the full service dependency graph: each service's kind, dependencies, and dependents.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_changes",
            "description": "Get recent deploys, config changes, and feature flags for a service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"},
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restart_pod",
            "description": "Restart a service pod. 2-tick cooldown during which the service is down, then metrics reset to baseline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"},
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rollback",
            "description": "Roll back the most recent deploy on a service. Only works if the service has a recent deploy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"},
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scale_up",
            "description": "Add replicas to a service. Reduces CPU and latency. Over-provisioning a healthy service is wasteful.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"},
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_feature_flag",
            "description": "Toggle a feature flag on a service. The flag must already exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"},
                    "flag_name": {"type": "string", "description": "Flag key to toggle"},
                },
                "required": ["service", "flag_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enable_circuit_breaker",
            "description": "Enable circuit breaker on a service to stop retry amplification. Hurts healthy services by cutting traffic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"},
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drain_region",
            "description": "Drain all traffic from a region. High user-impact cost. Only use for regional failures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "Region identifier (e.g. us-east-1)"},
                },
                "required": ["region"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kill_long_queries",
            "description": "Kill long-running queries on a database service to free connection pool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Database service name"},
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rotate_cert",
            "description": "Rotate TLS certificate between two services to fix certificate expiry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_a": {"type": "string", "description": "First service name"},
                    "service_b": {"type": "string", "description": "Second service name"},
                },
                "required": ["service_a", "service_b"],
            },
        },
    },
]
