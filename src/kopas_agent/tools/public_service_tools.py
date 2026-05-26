from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kopas_agent.environments.mock_public_service import MockPublicService


def get_public_service_tools(env: MockPublicService) -> dict[str, Callable[..., Any]]:
    return {
        "lookup_civil_case": env.lookup_civil_case,
        "submit_public_notice": env.submit_public_notice,
        "generate_response_to_citizen": env.generate_response_to_citizen,
        "escalate_to_human_officer": env.escalate_to_human_officer,
    }
