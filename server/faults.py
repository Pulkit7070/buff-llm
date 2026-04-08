"""Fault templates for the IncidentRoom simulator.

Six primary fault classes that the generator can inject, plus two
consequence-fault classes created only by wrong agent actions.

Every fault implements:
  inject(world)           — one-time setup at generation
  progress(world)         — called every tick while unresolved
  check_resolution(…)     — does this tool call fix the fault?
  check_wrong_action(…)   — does this tool call make things worse?
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from server.world import World


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class Fault(ABC):
    fault_id: str
    target: str  # primary target service name
    resolved: bool = False
    injected_at: int = 0

    @abstractmethod
    def inject(self, world: World) -> None: ...

    @abstractmethod
    def progress(self, world: World) -> None: ...

    @abstractmethod
    def check_resolution(self, tool_name: str, args: dict, world: World) -> bool: ...

    def check_wrong_action(
        self, tool_name: str, args: dict, world: World
    ) -> Optional[Fault]:
        """Return a new Fault to inject, or None."""
        return None


# ---------------------------------------------------------------------------
# 1. Memory leak after deploy
# ---------------------------------------------------------------------------

@dataclass
class MemoryLeakAfterDeploy(Fault):
    deploy_id: str = ""

    def inject(self, world: World) -> None:
        svc = world.services[self.target]
        self.deploy_id = f"d-{world.seed:04x}"
        svc.recent_changes.append({
            "type": "deploy", "id": self.deploy_id,
            "version": f"v2.{world.seed % 100}.0",
            "tick": max(0, world.tick - 2),
        })
        svc.memory_pct += 5
        svc.log_buffer.append({
            "tick": world.tick, "level": "INFO",
            "msg": f"T+{world.tick:03d} svc={svc.name} event=deploy id={self.deploy_id}",
        })

    def progress(self, world: World) -> None:
        svc = world.services[self.target]
        svc.memory_pct = min(100.0, svc.memory_pct + 3.0)
        if svc.memory_pct >= 95:
            svc.error_rate = min(1.0, svc.error_rate + 0.3)
            svc.cpu_pct = min(100.0, svc.cpu_pct + 10)
        svc.log_buffer.append({
            "tick": world.tick, "level": "WARN",
            "msg": (f"T+{world.tick:03d} svc={svc.name} "
                    f"metric=heap_used_pct val={svc.memory_pct:.1f} gc_pressure=high"),
        })

    def check_resolution(self, tool_name: str, args: dict, world: World) -> bool:
        return tool_name == "rollback" and args.get("service") == self.target

    def check_wrong_action(self, tool_name, args, world):
        if tool_name == "rollback" and args.get("service") != self.target:
            other = args.get("service", "")
            if other in world.services:
                svc = world.services[other]
                if any(c.get("type") == "deploy" for c in svc.recent_changes):
                    return LatentDefect(
                        fault_id=f"latent-{other}",
                        target=other,
                        injected_at=world.tick,
                        rate=0.08,
                    )
        return None


# ---------------------------------------------------------------------------
# 2. Cache eviction storm
# ---------------------------------------------------------------------------

@dataclass
class CacheEvictionStorm(Fault):
    ticks_active: int = 0

    def inject(self, world: World) -> None:
        svc = world.services[self.target]
        svc.error_rate += 0.05
        svc.latency_p99 += 50
        svc.log_buffer.append({
            "tick": world.tick, "level": "WARN",
            "msg": (f"T+{world.tick:03d} svc={svc.name} "
                    f"metric=cache_eviction_rate val=high keys_evicted=12453"),
        })

    def progress(self, world: World) -> None:
        self.ticks_active += 1
        svc = world.services[self.target]
        svc.error_rate = min(1.0, svc.error_rate + 0.04)
        svc.latency_p99 = min(2000, svc.latency_p99 + 40)
        svc.cpu_pct = min(100, svc.cpu_pct + 5)
        # after 3 ticks, dependents start timing out
        if self.ticks_active >= 3:
            for dep_name in world.topology.get(self.target, []):
                dep = world.services.get(dep_name)
                if dep:
                    dep.error_rate = min(1.0, dep.error_rate + 0.06)
                    dep.log_buffer.append({
                        "tick": world.tick, "level": "ERROR",
                        "msg": (f"T+{world.tick:03d} svc={dep.name} "
                                f"metric=upstream_timeout src={svc.name} "
                                f"val={svc.latency_p99:.0f}ms"),
                    })

    def check_resolution(self, tool_name, args, world) -> bool:
        return (args.get("service") == self.target
                and tool_name in ("restart_pod", "scale_up"))

    def check_wrong_action(self, tool_name, args, world):
        if tool_name == "restart_pod" and args.get("service") != self.target:
            other = args.get("service", "")
            if other in world.topology.get(self.target, []):
                return ServiceFlap(
                    fault_id=f"flap-{other}",
                    target=other,
                    injected_at=world.tick,
                )
        return None


# ---------------------------------------------------------------------------
# 3. Config drift
# ---------------------------------------------------------------------------

@dataclass
class ConfigDrift(Fault):
    flag_name: str = ""

    def inject(self, world: World) -> None:
        svc = world.services[self.target]
        self.flag_name = f"flag_{world.seed % 1000:03d}"
        svc.config_flags[self.flag_name] = True
        svc.recent_changes.append({
            "type": "config", "key": self.flag_name,
            "value": True, "tick": max(0, world.tick - 1),
        })
        svc.error_rate += 0.08
        svc.log_buffer.append({
            "tick": world.tick, "level": "INFO",
            "msg": (f"T+{world.tick:03d} svc={svc.name} "
                    f"event=config_change key={self.flag_name} val=true"),
        })

    def progress(self, world: World) -> None:
        svc = world.services[self.target]
        if world.tick % 2 == 0:
            svc.error_rate = min(1.0, svc.error_rate + 0.05)
        else:
            svc.error_rate = max(svc.baseline_error_rate, svc.error_rate - 0.02)
        svc.log_buffer.append({
            "tick": world.tick, "level": "ERROR",
            "msg": (f"T+{world.tick:03d} svc={svc.name} "
                    f"metric=error_rate val={svc.error_rate:.4f} pattern=intermittent"),
        })

    def check_resolution(self, tool_name, args, world) -> bool:
        return (tool_name == "toggle_feature_flag"
                and args.get("service") == self.target
                and args.get("flag_name") == self.flag_name)

    def check_wrong_action(self, tool_name, args, world):
        if tool_name == "rollback" and args.get("service") == self.target:
            svc = world.services[self.target]
            svc.latency_p99 += 100
            svc.log_buffer.append({
                "tick": world.tick, "level": "WARN",
                "msg": (f"T+{world.tick:03d} svc={svc.name} "
                        f"event=rollback_no_effect flag={self.flag_name} still_active=true"),
            })
        return None


# ---------------------------------------------------------------------------
# 4. Dependency timeout amplification
# ---------------------------------------------------------------------------

@dataclass
class DependencyTimeoutAmplification(Fault):
    ext_dep: str = ""

    def inject(self, world: World) -> None:
        svc = world.services[self.target]
        if svc.dependencies:
            self.ext_dep = svc.dependencies[0]
        svc.latency_p99 += 200
        svc.error_rate += 0.10
        svc.log_buffer.append({
            "tick": world.tick, "level": "ERROR",
            "msg": (f"T+{world.tick:03d} svc={svc.name} "
                    f"metric=retry_rate val=high dep={self.ext_dep} timeout=5000ms"),
        })

    def progress(self, world: World) -> None:
        svc = world.services[self.target]
        svc.latency_p99 = min(5000, svc.latency_p99 + 80)
        svc.cpu_pct = min(100, svc.cpu_pct + 4)
        svc.error_rate = min(1.0, svc.error_rate + 0.03)
        svc.log_buffer.append({
            "tick": world.tick, "level": "ERROR",
            "msg": (f"T+{world.tick:03d} svc={svc.name} "
                    f"metric=retry_amplification dep={self.ext_dep} "
                    f"queue_depth={int(svc.cpu_pct * 2)}"),
        })

    def check_resolution(self, tool_name, args, world) -> bool:
        return (tool_name == "enable_circuit_breaker"
                and args.get("service") == self.target)

    def check_wrong_action(self, tool_name, args, world):
        if tool_name == "scale_up" and args.get("service") == self.target:
            svc = world.services[self.target]
            svc.latency_p99 += 150
            svc.error_rate = min(1.0, svc.error_rate + 0.10)
            svc.log_buffer.append({
                "tick": world.tick, "level": "ERROR",
                "msg": (f"T+{world.tick:03d} svc={svc.name} "
                        f"event=scale_amplified retry_load_multiplied=true"),
            })
        return None


# ---------------------------------------------------------------------------
# 5. DB pool exhaustion
# ---------------------------------------------------------------------------

@dataclass
class DbPoolExhaustion(Fault):

    def inject(self, world: World) -> None:
        svc = world.services[self.target]
        svc.latency_p99 += 300
        svc.error_rate += 0.15
        svc.cpu_pct += 20
        svc.log_buffer.append({
            "tick": world.tick, "level": "ERROR",
            "msg": (f"T+{world.tick:03d} svc={svc.name} "
                    f"metric=conn_pool_active val=100 max=100 queue_depth=47"),
        })

    def progress(self, world: World) -> None:
        svc = world.services[self.target]
        svc.error_rate = min(1.0, svc.error_rate + 0.05)
        svc.latency_p99 = min(5000, svc.latency_p99 + 100)
        qd = 50 + world.tick * 10
        svc.log_buffer.append({
            "tick": world.tick, "level": "ERROR",
            "msg": (f"T+{world.tick:03d} svc={svc.name} "
                    f"metric=conn_pool_active val=100 max=100 queue_depth={qd}"),
        })

    def check_resolution(self, tool_name, args, world) -> bool:
        return (args.get("service") == self.target
                and tool_name in ("kill_long_queries", "scale_up"))

    def check_wrong_action(self, tool_name, args, world):
        if tool_name == "restart_pod":
            other = args.get("service", "")
            # restarting a client of the db closes connections in flight
            if other in world.topology.get(self.target, []):
                svc = world.services.get(other)
                if svc:
                    svc.error_rate = min(1.0, svc.error_rate + 0.20)
                    svc.log_buffer.append({
                        "tick": world.tick, "level": "ERROR",
                        "msg": (f"T+{world.tick:03d} svc={svc.name} "
                                f"event=connection_reset_in_flight src={self.target}"),
                    })
        return None


# ---------------------------------------------------------------------------
# 6. Cert expiry between services
# ---------------------------------------------------------------------------

@dataclass
class CertExpiryBetweenServices(Fault):
    service_b: str = ""

    def inject(self, world: World) -> None:
        svc_a = world.services[self.target]
        # pick a dependency as the peer
        if svc_a.dependencies:
            self.service_b = svc_a.dependencies[0]
        elif world.topology.get(self.target):
            self.service_b = world.topology[self.target][0]
        svc_a.error_rate += 0.40
        svc_a.log_buffer.append({
            "tick": world.tick, "level": "ERROR",
            "msg": (f"T+{world.tick:03d} svc={svc_a.name} "
                    f"metric=tls_handshake_fail peer={self.service_b} val=1.0"),
        })

    def progress(self, world: World) -> None:
        svc_a = world.services[self.target]
        svc_a.error_rate = min(1.0, max(0.40, svc_a.error_rate))
        svc_a.latency_p99 = max(svc_a.latency_p99, 500)
        svc_a.log_buffer.append({
            "tick": world.tick, "level": "ERROR",
            "msg": (f"T+{world.tick:03d} svc={svc_a.name} "
                    f"metric=tls_handshake_fail peer={self.service_b} "
                    f"err=certificate_expired"),
        })

    def check_resolution(self, tool_name, args, world) -> bool:
        if tool_name != "rotate_cert":
            return False
        pair = {args.get("service_a", ""), args.get("service_b", "")}
        return pair == {self.target, self.service_b}


# ---------------------------------------------------------------------------
# Consequence faults (not in generator — only created by wrong actions)
# ---------------------------------------------------------------------------

@dataclass
class LatentDefect(Fault):
    """Planted when the agent rolls back a deploy on the wrong service."""
    rate: float = 0.08

    def inject(self, world: World) -> None:
        pass  # already active when created

    def progress(self, world: World) -> None:
        svc = world.services.get(self.target)
        if svc:
            svc.error_rate = min(1.0, svc.error_rate + self.rate)

    def check_resolution(self, tool_name, args, world) -> bool:
        return tool_name == "restart_pod" and args.get("service") == self.target


@dataclass
class ServiceFlap(Fault):
    """Planted when the agent restarts a dependent of a degraded cache."""
    cycle: int = 0

    def inject(self, world: World) -> None:
        pass

    def progress(self, world: World) -> None:
        self.cycle += 1
        svc = world.services.get(self.target)
        if svc:
            if self.cycle % 2 == 0:
                svc.error_rate = min(1.0, svc.error_rate + 0.20)
            else:
                svc.error_rate = max(svc.baseline_error_rate, svc.error_rate - 0.10)

    def check_resolution(self, tool_name, args, world) -> bool:
        return tool_name == "restart_pod" and args.get("service") == self.target
