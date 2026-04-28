# IncidentRoom v2 -- Architecture

Deep technical reference for the IncidentRoom tick-based SRE simulator.

## Episode Lifecycle

An episode flows through five modules in sequence:

```
generator.py --> env.py --> tools.py --> world.py --> grader.py
```

1. **`generator.py:generate(difficulty, seed)`** creates a `World` with services, topology, injected faults, and red herrings. Uses `random.Random(seed)` -- never the global random state.

2. **`env.py:IncidentRoomEnv.reset(seed, difficulty)`** calls `generate()` and returns an initial observation to the agent.

3. **Agent loop:** the agent calls `env.step(tool_name, tool_args)` repeatedly.
   - `env.step()` looks up the handler in `tools.py:TOOL_HANDLERS` and executes it against the `World`.
   - For action tools (not read-only), it iterates over `world.active_faults` checking `check_resolution()` and `check_wrong_action()`.
   - If a fault's `check_wrong_action()` returns a new `Fault`, that fault is injected into the world immediately.
   - Then `world.py:tick_world()` advances the simulation by one tick.

4. **`world.py:tick_world()`** runs the following substeps in order:
   1. **Pending effects** -- deferred callbacks (e.g., restart recovery after 2-tick cooldown) fire when their tick arrives.
   2. **Fault progression** -- every unresolved fault's `progress(world)` method runs, degrading metrics on its target service.
   3. **Natural healing** -- services not under an active fault slowly recover toward their baselines (15% of gap per tick).
   4. **Cascade propagation** -- degraded/failing/down services push `error_rate * 0.3` additional error onto their dependents via the topology graph.
   5. **Recompute status** -- each service's status is set based on `health()`: >=0.8 healthy, >=0.5 degraded, >=0.2 failing, otherwise down.
   6. **User impact accumulation** -- `sum(criticality * (1 - health))` is added to `world.user_impact_total`.
   7. **Log emission + metric history recording.**
   8. **Terminal check** -- episode ends on success (all healthy + all resolved), timeout (tick >= max_ticks), or catastrophe (>80% services down).

5. **`grader.py:grade(world)`** computes the final score as a pure function over the `World` state.

## Fault Injection and Progression Model

### 6 Primary Faults

These are the faults the generator can inject at episode start. Each targets specific service kinds.

| # | Fault Class | Target Kinds | Inject Effect | Progression (per tick) |
|---|-------------|--------------|---------------|------------------------|
| 1 | `MemoryLeakAfterDeploy` | api, worker, gateway | Adds fake deploy to `recent_changes`, memory +5% | memory +3%, at >=95% also error_rate +0.30 and cpu +10 |
| 2 | `CacheEvictionStorm` | cache | error_rate +0.05, latency +50ms | error_rate +0.04, latency +40ms, cpu +5; after 3 ticks dependents get error_rate +0.06 |
| 3 | `ConfigDrift` | api, worker | Adds config flag, error_rate +0.08 | Alternating: even ticks error +0.05, odd ticks error -0.02 (intermittent pattern) |
| 4 | `DependencyTimeoutAmplification` | api | latency +200ms, error_rate +0.10 | latency +80ms, cpu +4, error_rate +0.03 |
| 5 | `DbPoolExhaustion` | database | latency +300ms, error_rate +0.15, cpu +20 | error_rate +0.05, latency +100ms |
| 6 | `CertExpiryBetweenServices` | api, gateway | error_rate +0.40 on service_a | Holds error_rate >= 0.40, latency >= 500ms |

### 2 Consequence Faults

These are never generated at episode start. They are created only by `check_wrong_action()` when the agent takes an incorrect action.

| Fault Class | Created By | Effect |
|-------------|-----------|--------|
| `LatentDefect` | Rolling back a deploy on the wrong service (when MemoryLeakAfterDeploy is active) | error_rate +0.08 per tick on the wrongly-rolled-back service. Fix: `restart_pod(target)` |
| `ServiceFlap` | Restarting a dependent of a degraded cache (when CacheEvictionStorm is active) | Alternating: even cycles error +0.20, odd cycles error -0.10 (flapping pattern). Fix: `restart_pod(target)` |

## The Consequence Mechanic

This is the core design innovation. In most benchmarks, a wrong action subtracts points. In IncidentRoom, a wrong action **creates a new fault that must also be resolved**.

The flow in `env.py:step()`:

```python
for fault in world.active_faults:
    if fault.resolved:
        continue
    if fault.check_resolution(tool_name, tool_args, world):
        fault.resolved = True
    else:
        consequence = fault.check_wrong_action(tool_name, tool_args, world)
        if consequence is not None:
            consequence.inject(world)
            world.active_faults.append(consequence)
```

This means:
- The total number of faults in the episode is not fixed. It grows with agent mistakes.
- The fault_resolution score denominator increases: `resolved / total * 40` gets harder to max out.
- Consequence faults have their own `progress()` that further degrades the world each tick.
- The agent must now spend additional ticks fixing self-inflicted damage, eating into the efficiency bonus.

### Example Cascade

1. Agent sees CacheEvictionStorm on `redis-1`. Dependents of `redis-1` include `user-svc-3`.
2. Agent mistakenly calls `restart_pod(service="user-svc-3")` instead of targeting the cache.
3. `CacheEvictionStorm.check_wrong_action()` detects the restart targets a dependent, returns `ServiceFlap(target="user-svc-3")`.
4. `ServiceFlap` is injected: `user-svc-3` now alternates between error spikes and partial recovery every tick.
5. The agent now has 2 faults to fix instead of 1, and has wasted a tick.

## Cascade Propagation Through Topology

The `topology` dict maps each service to its list of **dependents** (services that depend on it). During `tick_world()`:

```python
for src_name, dependents in world.topology.items():
    src = world.services.get(src_name)
    if src.status in ("degraded", "failing", "down"):
        for dep_name in dependents:
            dep = world.services.get(dep_name)
            if dep:
                dep._cascade_error += src.error_rate * 0.3
```

After all cascades are computed, the accumulated `_cascade_error` is applied to each service's `error_rate` and `latency_p99`. This means a single degraded database can push errors into every API service that depends on it, which in turn can push errors into gateways -- a realistic failure propagation model.

## Scoring Formula

The grader (`server/grader.py`) computes four components that sum to 100:

### 1. Fault Resolution (40 points)

```
fault_score = (resolved / max(total, 1)) * 40
```

Note that `total` includes consequence faults. If the agent caused 2 consequence faults and only resolved the original, the score is `1/3 * 40 = 13.3`, not `1/1 * 40 = 40`.

### 2. Service Health (25 points)

```
avg_health = mean(health(svc) for svc in world.services)
health_score = avg_health * 25
```

The `health()` function computes a scalar 0.0-1.0 from four metrics:

```python
err = min(error_rate / 0.5, 1.0)
lat = min(max(latency_p99 - 100, 0) / 900.0, 1.0)
cpu = min(max(cpu_pct - 80, 0) / 20.0, 1.0)
mem = min(max(memory_pct - 85, 0) / 15.0, 1.0)
health = max(0.0, 1.0 - max(err, lat, cpu, mem))
```

Health is dominated by the worst metric. A service with error_rate=0.5 has health=0.0 regardless of how good its other metrics are.

### 3. User Impact (20 points)

```
max_impact = max_ticks * sum(svc.criticality for svc in services)
impact_ratio = user_impact_total / max_impact
impact_score = max(0, 1 - impact_ratio) * 20
```

User impact accumulates every tick: `sum(criticality * (1 - health))`. Faster resolution means less accumulated impact.

### 4. Time Efficiency (15 points)

```
if all_faults_resolved:
    efficiency_score = (1 - tick / max_ticks) * 15
else:
    efficiency_score = 0
```

This is an all-or-nothing bonus. If even one fault remains unresolved, the agent gets 0 efficiency points. If all are resolved, the score rewards finishing early.

### Worked Example

Setup: easy difficulty, 3 services, 1 fault, max_ticks=15.

- Agent resolves the fault at tick 6. All services return to healthy by tick 8.
- `fault_score = (1/1) * 40 = 40.0`
- `health_score = 1.0 * 25 = 25.0` (all healthy at end)
- `impact_score` depends on cumulative impact during ticks 0-8 while services were degraded. Suppose impact_ratio = 0.15: `(1 - 0.15) * 20 = 17.0`
- `efficiency_score = (1 - 8/15) * 15 = 7.0`
- **Total: 89.0 / 100**

## Determinism Guarantees

- Every call to `generate(difficulty, seed)` creates an isolated `random.Random(seed)` instance. The global `random` module state is never touched during world generation.
- All fault injection, service selection, topology wiring, and red herring placement use this local RNG.
- `tick_world()` is a pure function of `World` state -- no randomness at runtime.
- Tool handlers are deterministic functions of `(world, **kwargs)`.
- Result: `env.reset(seed=42, difficulty="easy")` followed by the same sequence of `env.step()` calls will always produce identical World states and scores.

## Difficulty Configuration

Defined in `server/generator.py:DIFFICULTY_CFG`:

| Level | Services | Faults | Red Herrings | Tick Budget |
|-------|----------|--------|--------------|-------------|
| easy | 3 | 1 | 0 | 15 |
| medium | 6 | 1 | 2 | 20 |
| hard | 8 | 2 | 2 | 25 |

**Red herrings** are recent deploys or config flag changes planted on healthy services. They show up in `get_recent_changes()` output and look suspicious, but acting on them wastes ticks (and may trigger consequence faults).

## Service Kinds

The generator selects from a pool of service templates with six kinds:

| Kind | Examples | Typical Criticality |
|------|----------|--------------------:|
| gateway | `api-gw-*`, `cdn-proxy-*` | 0.6 - 0.9 |
| api | `user-svc-*`, `order-svc-*`, `payment-svc-*`, `search-svc-*`, `auth-svc-*` | 0.5 - 0.9 |
| cache | `redis-*` | 0.6 |
| database | `pg-primary-*`, `pg-replica-*` | 0.4 - 0.9 |
| worker | `notify-worker-*` | 0.3 |
| queue | `rabbitmq-*` | 0.5 |

The generator guarantees at least one gateway, one cache, and one database in every episode (if the service count allows). Topology wiring follows realistic patterns: APIs depend on gateways, caches, and databases; workers depend on queues and databases.

## Tool Reference

### Read Tools (5) -- no side effects

| Tool | Parameters | Description |
|------|-----------|-------------|
| `list_services` | *(none)* | List all services with name, kind, status, and region |
| `get_metrics` | `service` | Numeric metrics: error_rate, latency_p99_ms, cpu_pct, memory_pct, health |
| `get_logs` | `service`, `n=20` | Last N log entries for a service |
| `get_topology` | *(none)* | Full dependency graph: each service's kind, dependencies, and dependents |
| `get_recent_changes` | `service` | Recent deploys, config changes, and feature flags |

### Action Tools (8) -- mutate World state

| Tool | Parameters | Description |
|------|-----------|-------------|
| `restart_pod` | `service` | Restart a service pod. 2-tick cooldown (service goes down), then metrics reset to baseline |
| `rollback` | `service` | Roll back the most recent deploy. Only works if the service has a recent deploy |
| `scale_up` | `service` | Add replicas. Reduces CPU and latency. Over-provisioning a healthy service is wasteful |
| `toggle_feature_flag` | `service`, `flag_name` | Toggle an existing feature flag on a service |
| `enable_circuit_breaker` | `service` | Stop retry amplification. Hurts healthy services by cutting traffic |
| `drain_region` | `region` | Drain all traffic from a region. High user-impact cost. Nuclear option for regional failures |
| `kill_long_queries` | `service` | Kill long-running queries on a database to free the connection pool |
| `rotate_cert` | `service_a`, `service_b` | Rotate TLS certificate between two services to fix certificate expiry |

**Important:** Every tool call -- including reads -- advances the tick counter by 1. Efficient agents minimize unnecessary reads.

## Service Health Function

Defined in `server/world.py:health()`:

```python
def health(svc: Service) -> float:
    err = min(svc.error_rate / 0.5, 1.0)
    lat = min(max(svc.latency_p99 - 100, 0) / 900.0, 1.0)
    cpu = min(max(svc.cpu_pct - 80, 0) / 20.0, 1.0)
    mem = min(max(svc.memory_pct - 85, 0) / 15.0, 1.0)
    return max(0.0, 1.0 - max(err, lat, cpu, mem))
```

Health is 1.0 when all metrics are in normal range. It degrades based on the **worst** metric:
- error_rate: 0.0 is perfect, 0.5+ means health=0
- latency_p99: <=100ms is perfect, >=1000ms means health=0
- cpu_pct: <=80% is perfect, >=100% means health=0
- memory_pct: <=85% is perfect, >=100% means health=0

Status thresholds based on health: healthy (>=0.8), degraded (>=0.5), failing (>=0.2), down (<0.2).

## SSE Streaming Architecture

The web dashboard (`webapp.py`) and LLM agent (`llm_agent.py`) both use Server-Sent Events for real-time updates:

- **GET endpoints** use standard `EventSource` on the client side.
- **POST endpoints** (which carry API keys in the request body) use `fetch()` with `ReadableStream` and manual SSE frame parsing on the client side.
- All agent generators yield `{"event": str, "data": dict}` dicts.
- The `_snap_env()` / `_snap()` helpers serialize `World` state for SSE frames. These are duplicated in both `webapp.py` and `llm_agent.py` to avoid circular imports.
