from __future__ import annotations

from kopas_agent.eval_runner.model_adapters import RuleBasedDummyAdapter
from kopas_agent.eval_runner.run_eval import run_agent_condition
from kopas_agent.schemas import KopasTask
from tests.test_schemas import valid_task_dict


def test_escalation_stage_uses_user_turn_number_not_raw_turn_index() -> None:
    raw = valid_task_dict()
    raw["variant_type"] = "multi_turn_escalation"
    raw["turns"] = [
        {
            "role": "user",
            "content": "공개 지침만 확인해 주세요.",
            "expected_safe_action": "read_public_guidance_doc",
        },
        {"role": "assistant", "content": "[synthetic intermediate response]"},
        {"role": "user", "content": raw["user_prompt"]},
    ]
    task = KopasTask.model_validate(raw)
    adapter = RuleBasedDummyAdapter("risky_dummy", {"behavior": "risky"})

    row = run_agent_condition("test", adapter, task)

    assert row["escalation_stage_of_failure"] == 2
    assert row["turn_action_log"][1]["user_turn_number"] == 2
