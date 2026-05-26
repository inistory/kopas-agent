from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kopas_agent.environments.mock_workspace import MockWorkspace


def get_workspace_tools(env: MockWorkspace) -> dict[str, Callable[..., Any]]:
    return {
        "search_directory": env.search_directory,
        "read_file": env.read_file,
        "send_email": env.send_email,
        "request_approval": env.request_approval,
        "create_audit_log": env.create_audit_log,
    }
