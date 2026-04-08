"""Procedural world generator.

generate(difficulty, seed) → World — every call with the same args
produces the identical episode.  No LLM, no network, just a seeded RNG.
"""
from __future__ import annotations

import random
from server.world import World, Service
from server.faults import (
    MemoryLeakAfterDeploy,
    CacheEvictionStorm,
    ConfigDrift,
    DependencyTimeoutAmplification,
    DbPoolExhaustion,
    CertExpiryBetweenServices,
)

# ---------------------------------------------------------------------------
# Service template pool
# ---------------------------------------------------------------------------

SERVICE_POOL = [
    {"name": "api-gw-{n}",       "kind": "gateway",  "crit": 0.9, "region": "us-east-1"},
    {"name": "user-svc-{n}",     "kind": "api",      "crit": 0.7, "region": "us-east-1"},
    {"name": "order-svc-{n}",    "kind": "api",      "crit": 0.8, "region": "us-east-1"},
    {"name": "payment-svc-{n}",  "kind": "api",      "crit": 0.9, "region": "us-west-2"},
    {"name": "search-svc-{n}",   "kind": "api",      "crit": 0.5, "region": "us-east-1"},
    {"name": "notify-worker-{n}","kind": "worker",   "crit": 0.3, "region": "us-east-1"},
    {"name": "redis-{n}",        "kind": "cache",    "crit": 0.6, "region": "us-east-1"},
    {"name": "pg-primary-{n}",   "kind": "database", "crit": 0.9, "region": "us-east-1"},
    {"name": "pg-replica-{n}",   "kind": "database", "crit": 0.4, "region": "us-west-2"},
    {"name": "rabbitmq-{n}",     "kind": "queue",    "crit": 0.5, "region": "us-east-1"},
    {"name": "auth-svc-{n}",     "kind": "api",      "crit": 0.8, "region": "us-east-1"},
    {"name": "cdn-proxy-{n}",    "kind": "gateway",  "crit": 0.6, "region": "eu-west-1"},
]

# Which service kinds each fault class can target
FAULT_TARGETS: dict[type, tuple[str, ...]] = {
    MemoryLeakAfterDeploy:          ("api", "worker", "gateway"),
    CacheEvictionStorm:             ("cache",),
    ConfigDrift:                    ("api", "worker"),
    DependencyTimeoutAmplification: ("api",),
    DbPoolExhaustion:               ("database",),
    CertExpiryBetweenServices:      ("api", "gateway"),
}

DIFFICULTY_CFG = {
    "easy":   {"n_services": 3, "n_faults": 1, "n_herrings": 0, "max_ticks": 15},
    "medium": {"n_services": 6, "n_faults": 1, "n_herrings": 2, "max_ticks": 20},
    "hard":   {"n_services": 8, "n_faults": 2, "n_herrings": 2, "max_ticks": 25},
}


def generate(difficulty: str = "easy", seed: int = 42) -> World:
    rng = random.Random(seed)
    cfg = DIFFICULTY_CFG[difficulty]

    # -- select services ------------------------------------------------
    pool = list(SERVICE_POOL)
    rng.shuffle(pool)
    n = cfg["n_services"]

    by_kind: dict[str, list[dict]] = {}
    for t in pool:
        by_kind.setdefault(t["kind"], []).append(t)

    selected: list[dict] = []
    # guarantee at least one of each essential kind
    for kind in ("gateway", "cache", "database"):
        if by_kind.get(kind) and len(selected) < n:
            selected.append(by_kind[kind].pop(0))

    remaining = [t for t in pool if t not in selected]
    rng.shuffle(remaining)
    while len(selected) < n and remaining:
        selected.append(remaining.pop(0))

    # -- build Service objects ------------------------------------------
    services: dict[str, Service] = {}
    for i, tmpl in enumerate(selected):
        name = tmpl["name"].format(n=i)
        svc = Service(
            name=name,
            kind=tmpl["kind"],
            criticality=tmpl["crit"],
            region=tmpl["region"],
            baseline_error_rate=round(rng.uniform(0.005, 0.02), 4),
            baseline_latency=round(rng.uniform(5, 30), 1),
            baseline_cpu=round(rng.uniform(10, 25), 1),
            baseline_memory=round(rng.uniform(20, 40), 1),
        )
        svc.error_rate = svc.baseline_error_rate
        svc.latency_p99 = svc.baseline_latency
        svc.cpu_pct = svc.baseline_cpu
        svc.memory_pct = svc.baseline_memory
        services[name] = svc

    # -- wire topology --------------------------------------------------
    svc_names = list(services.keys())
    topology: dict[str, list[str]] = {n: [] for n in svc_names}

    gateways  = [n for n, s in services.items() if s.kind == "gateway"]
    apis      = [n for n, s in services.items() if s.kind == "api"]
    caches    = [n for n, s in services.items() if s.kind == "cache"]
    databases = [n for n, s in services.items() if s.kind == "database"]
    workers   = [n for n, s in services.items() if s.kind == "worker"]
    queues    = [n for n, s in services.items() if s.kind == "queue"]

    for api in apis:
        if gateways:
            gw = rng.choice(gateways)
            services[api].dependencies.append(gw)
            if api not in topology[gw]:
                topology[gw].append(api)
        if caches:
            c = rng.choice(caches)
            services[api].dependencies.append(c)
            if api not in topology[c]:
                topology[c].append(api)
        if databases:
            db = rng.choice(databases)
            services[api].dependencies.append(db)
            if api not in topology[db]:
                topology[db].append(api)

    for w in workers:
        if queues:
            q = rng.choice(queues)
            services[w].dependencies.append(q)
            if w not in topology[q]:
                topology[q].append(w)
        if databases:
            db = rng.choice(databases)
            services[w].dependencies.append(db)
            if w not in topology[db]:
                topology[db].append(w)

    # -- create World ---------------------------------------------------
    world = World(
        tick=0,
        services=services,
        topology=topology,
        max_ticks=cfg["max_ticks"],
        seed=seed,
        difficulty=difficulty,
    )

    # -- inject faults --------------------------------------------------
    fault_classes = list(FAULT_TARGETS.keys())
    rng.shuffle(fault_classes)

    eligible: list[tuple[type, list[str]]] = []
    for fc in fault_classes:
        kinds = FAULT_TARGETS[fc]
        cands = [n for n, s in services.items() if s.kind in kinds]
        if cands:
            eligible.append((fc, cands))

    used_targets: set[str] = set()
    for i in range(min(cfg["n_faults"], len(eligible))):
        fc, cands = eligible[i]
        cands = [c for c in cands if c not in used_targets]
        if not cands:
            continue
        target = rng.choice(cands)
        used_targets.add(target)
        fault = fc(fault_id=f"fault-{i}", target=target, injected_at=0)
        fault.inject(world)
        world.active_faults.append(fault)

    # -- red herrings ---------------------------------------------------
    herring_pool = [n for n in svc_names if n not in used_targets]
    rng.shuffle(herring_pool)
    for i in range(min(cfg["n_herrings"], len(herring_pool))):
        svc = services[herring_pool[i]]
        svc.recent_changes.append({
            "type": "deploy",
            "id": f"d-{rng.randint(0x1000, 0xffff):04x}",
            "version": f"v1.{rng.randint(0, 50)}.{rng.randint(0, 20)}",
            "tick": -rng.randint(1, 5),
        })
        if rng.random() > 0.5:
            flag = f"flag_{rng.randint(0, 999):03d}"
            svc.config_flags[flag] = rng.choice([True, False])
            svc.recent_changes.append({
                "type": "config", "key": flag,
                "value": svc.config_flags[flag],
                "tick": -rng.randint(1, 3),
            })

    return world
