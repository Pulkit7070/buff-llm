"""IncidentRoom environment — the main interface for the agent.

env.reset(seed, difficulty) → initial observation
env.step(tool_name, tool_args) → {tool_result, tick, done}
env.get_tools() → OpenAI function-calling schemas
env.grade() → final score dict
"""
from __future__ import annotations

import json
import random
from typing import Any

from server.generator import generate
from server.grader import grade
from server.tools import TOOL_HANDLERS, TOOL_SCHEMAS, READ_TOOLS, ACTION_TOOLS
from server.world import World, tick_world, health


class IncidentRoomEnv:
    """Stateful, tick-based SRE incident simulator."""

    def __init__(self) -> None:
        self.world: World | None = None

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: int | None = None,
        difficulty: str = "easy",
    ) -> dict:
        """Generate a fresh episode.  Returns an initial observation."""
        if seed is None:
            seed = random.randint(0, 2**31)
        self.world = generate(difficulty=difficulty, seed=seed)
        return self._observation(
            tool_result={"message": "Incident detected. Begin investigation."},
            tick_info={"done": False, "outcome": "running", "tick": 0},
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, tool_name: str, tool_args: dict[str, Any]) -> dict:
        """Execute one tool call, then advance the world by one tick."""
        assert self.world is not None, "call reset() first"

        # validate tool
        if tool_name not in TOOL_HANDLERS:
            return self._observation(
                tool_result={"error": f"unknown tool '{tool_name}'"},
                tick_info={"done": False, "outcome": "running", "tick": self.world.tick},
            )

        # execute tool
        handler = TOOL_HANDLERS[tool_name]
        tool_result = handler(self.world, **tool_args)

        # fault resolution / wrong-action for action tools
        if tool_name in ACTION_TOOLS:
            any_resolved = False
            new_faults: list = []
            for fault in self.world.active_faults:
                if fault.resolved:
                    continue
                if fault.check_resolution(tool_name, tool_args, self.world):
                    fault.resolved = True
                    any_resolved = True
                else:
                    consequence = fault.check_wrong_action(
                        tool_name, tool_args, self.world
                    )
                    if consequence is not None:
                        new_faults.append(consequence)

            for nf in new_faults:
                nf.inject(self.world)
                self.world.active_faults.append(nf)

            # log action
            self.world.action_log.append({
                "tick": self.world.tick,
                "tool": tool_name,
                "args": tool_args,
                "resolved": any_resolved,
            })

            if not any_resolved and not new_faults:
                self.world.wasted_actions += 1

        # advance simulation
        tick_info = tick_world(self.world)

        return self._observation(tool_result=tool_result, tick_info=tick_info)

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------
    def get_tools(self) -> list[dict]:
        return TOOL_SCHEMAS

    def grade(self) -> dict:
        assert self.world is not None
        return grade(self.world)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------
    def _observation(self, tool_result: dict, tick_info: dict) -> dict:
        assert self.world is not None
        return {
            "tool_result": tool_result,
            "tick": tick_info,
            "done": tick_info["done"],
            "services_summary": [
                {"name": s.name, "status": s.status, "health": round(health(s), 3)}
                for s in self.world.services.values()
            ],
        }
