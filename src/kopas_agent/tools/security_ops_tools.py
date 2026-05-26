from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kopas_agent.environments.mock_security_ops import MockSecurityOps


def get_security_ops_tools(env: MockSecurityOps) -> dict[str, Callable[..., Any]]:
    return {
        "query_security_log": env.query_security_log,
        "read_secret": env.read_secret,
        "open_security_ticket": env.open_security_ticket,
        "run_synthetic_scan": env.run_synthetic_scan,
        "request_approval": env.request_approval,
    }
