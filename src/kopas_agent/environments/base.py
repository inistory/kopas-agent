from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any


class BaseMockEnvironment(ABC):
    """Base class for synthetic tool-use safety environments."""

    supported_tools: tuple[str, ...] = ()

    def __init__(self, initial_state: dict[str, Any] | None = None):
        self.initial_state = deepcopy(initial_state or {})
        self.action_log: list[dict[str, Any]] = []
        self.state = self.reset()

    @abstractmethod
    def reset(self) -> dict[str, Any]:
        """Return a fresh environment state."""

    def log_action(self, tool_name: str, args: dict[str, Any], result: Any) -> None:
        self.action_log.append(
            {
                "tool_name": tool_name,
                "args": deepcopy(args),
                "result": deepcopy(result),
            }
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": deepcopy(self.state),
            "action_log": deepcopy(self.action_log),
        }
