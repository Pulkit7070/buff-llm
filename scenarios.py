"""Custom scenario loader — parse YAML into a World for the simulator."""
from __future__ import annotations

import random
from typing import Any

import yaml

from server.world import World, Service
from server.faults import (
    MemoryLeakAfterDeploy,
    CacheEvictionStorm,
    ConfigDrift,
    DependencyTimeoutAmplification,
    DbPoolExhaustion,
    CertExpiryBetweenServices,
)

FAULT_MAP: dict[str, type] = {
    "memory_leak":          MemoryLeakAfterDeploy,
    "cache_eviction":       CacheEvictionStorm,
    "config_drift":         ConfigDrift,
    "dependency_timeout":   DependencyTimeoutAmplification,
    "db_pool_exhaustion":   DbPoolExhaustion,
    "cert_expiry":          CertExpiryBetweenServices,
}

EXAMPLE_YAML = """\
name: Redis Cache Storm
description: Cache eviction storm cascading to the API layer
max_ticks: 20
services:
  - name: api-gw-0
    kind: gateway
    region: us-east-1
  - name: user-svc-0
    kind: api
    region: us-east-1
    dependencies: [api-gw-0, redis-0, pg-0]
  - name: redis-0
    kind: cache
    region: us-east-1
  - name: pg-0
    kind: database
    region: us-east-1
faults:
  - type: cache_eviction
    target: redis-0
"""


def parse_scenario(yaml_text: str) -> dict:
    """Parse YAML text and return a validated scenario dict."""
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, dict):
        raise ValueError("Scenario must be a YAML mapping")
    if "services" not in data or not data["services"]:
        raise ValueError("Scenario must define at least one service")
    return data


def build_world_from_scenario(yaml_text: str, seed: int = 42) -> World:
    """Create a World from a YAML scenario definition."""
    rng = random.Random(seed)
    data = parse_scenario(yaml_text)

    max_ticks = data.get("max_ticks", 20)
    difficulty = data.get("difficulty", "custom")

    # Build services
    services: dict[str, Service] = {}
    for s_def in data["services"]:
        name = s_def["name"]
        svc = Service(
            name=name,
            kind=s_def.get("kind", "api"),
            criticality=s_def.get("criticality", 0.7),
            region=s_def.get("region", "us-east-1"),
            baseline_error_rate=round(rng.uniform(0.005, 0.02), 4),
            baseline_latency=round(rng.uniform(5, 30), 1),
            baseline_cpu=round(rng.uniform(10, 25), 1),
            baseline_memory=round(rng.uniform(20, 40), 1),
        )
        svc.error_rate = svc.baseline_error_rate
        svc.latency_p99 = svc.baseline_latency
        svc.cpu_pct = svc.baseline_cpu
        svc.memory_pct = svc.baseline_memory
        for dep in s_def.get("dependencies", []):
            svc.dependencies.append(dep)
        services[name] = svc

    # Build topology (dependency → dependents)
    svc_names = list(services.keys())
    topology: dict[str, list[str]] = {n: [] for n in svc_names}
    for name, svc in services.items():
        for dep in svc.dependencies:
            if dep in topology and name not in topology[dep]:
                topology[dep].append(name)

    world = World(
        tick=0,
        services=services,
        topology=topology,
        max_ticks=max_ticks,
        seed=seed,
        difficulty=difficulty,
    )

    # Inject faults
    for i, f_def in enumerate(data.get("faults", [])):
        f_type = f_def.get("type", "")
        f_target = f_def.get("target", "")
        if f_type not in FAULT_MAP:
            continue
        if f_target not in services:
            continue
        fault_cls = FAULT_MAP[f_type]
        fault = fault_cls(fault_id=f"fault-{i}", target=f_target, injected_at=0)
        fault.inject(world)
        world.active_faults.append(fault)

    return world
