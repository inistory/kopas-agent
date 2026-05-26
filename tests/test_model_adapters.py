from __future__ import annotations

from kopas_agent.environments import create_environment
from kopas_agent.eval_runner.model_adapters import RuleBasedDummyAdapter
from kopas_agent.schemas import KopasTask
from tests.test_schemas import valid_task_dict


def test_variant_sensitive_adapter_preserves_configured_behavior_metadata() -> None:
    raw = valid_task_dict()
    raw["variant_type"] = "authority_claim"
    task = KopasTask.model_validate(raw)
    env = create_environment(task)
    adapter = RuleBasedDummyAdapter(
        "authority_vulnerable_dummy",
        {"behavior": "authority_vulnerable"},
    )

    result = adapter.run(task, env)

    assert result.metadata["behavior"] == "authority_vulnerable"
    assert result.metadata["action_profile"] == "risky"
