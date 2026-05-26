from kopas_agent.environments.base import BaseMockEnvironment
from kopas_agent.environments.mock_public_service import MockPublicService
from kopas_agent.environments.mock_security_ops import MockSecurityOps
from kopas_agent.environments.mock_workspace import MockWorkspace
from kopas_agent.schemas import KopasTask


def create_environment(task: KopasTask) -> BaseMockEnvironment:
    if task.domain == "mock_workspace":
        return MockWorkspace(task.environment_state)
    if task.domain == "mock_public_service":
        return MockPublicService(task.environment_state)
    if task.domain == "mock_security_ops":
        return MockSecurityOps(task.environment_state)
    msg = f"Unsupported domain: {task.domain}"
    raise ValueError(msg)


ENVIRONMENT_TOOL_MAP = {
    "mock_workspace": MockWorkspace.supported_tools,
    "mock_public_service": MockPublicService.supported_tools,
    "mock_security_ops": MockSecurityOps.supported_tools,
}

__all__ = [
    "BaseMockEnvironment",
    "ENVIRONMENT_TOOL_MAP",
    "MockPublicService",
    "MockSecurityOps",
    "MockWorkspace",
    "create_environment",
]
