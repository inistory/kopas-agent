from __future__ import annotations

import json
from pathlib import Path

import yaml

from kopas_agent.eval_runner.run_eval import run_eval
from tests.test_schemas import valid_task_dict


def test_run_eval_writes_chat_and_agent_conditions(tmp_path: Path) -> None:
    dataset = tmp_path / "tasks.jsonl"
    config = tmp_path / "models.yaml"
    output = tmp_path / "results.jsonl"
    raw = valid_task_dict()
    raw["benign_counterpart_id"] = None
    raw["base_task_id"] = raw["task_id"]
    raw["primary_hazard_category"] = "P1_sensitive_info_leakage"
    dataset.write_text(json.dumps(raw, ensure_ascii=False) + "\n", encoding="utf-8")
    config.write_text(
        yaml.safe_dump(
            {
                "run": {"conditions": ["chat_only", "agent_tool_use"]},
                "models": [
                    {
                        "model_id": "authority_vulnerable_dummy",
                        "adapter": "rule_based",
                        "behavior": "authority_vulnerable",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = run_eval(dataset, config, output, run_id="test")

    assert len(rows) == 2
    assert {row["condition"] for row in rows} == {"chat_only", "agent_tool_use"}
    assert all("chat_response_score" in row for row in rows)


def test_run_eval_resume_skips_completed_rows(tmp_path: Path) -> None:
    dataset = tmp_path / "tasks.jsonl"
    config = tmp_path / "models.yaml"
    output = tmp_path / "results.jsonl"
    raw = valid_task_dict()
    raw["benign_counterpart_id"] = None
    raw["base_task_id"] = raw["task_id"]
    raw["primary_hazard_category"] = "P1_sensitive_info_leakage"
    dataset.write_text(json.dumps(raw, ensure_ascii=False) + "\n", encoding="utf-8")
    config.write_text(
        yaml.safe_dump(
            {
                "run": {"conditions": ["chat_only", "agent_tool_use"]},
                "models": [
                    {
                        "model_id": "strict_safety_dummy",
                        "adapter": "rule_based",
                        "behavior": "strict_safety",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    first_rows = run_eval(dataset, config, output, run_id="test")
    resumed_rows = run_eval(dataset, config, output, run_id="test", resume=True)
    written_rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]

    assert len(first_rows) == 2
    assert len(resumed_rows) == 2
    assert len(written_rows) == 2


def test_run_eval_repeats_record_run_index_and_resume(tmp_path: Path) -> None:
    dataset = tmp_path / "tasks.jsonl"
    config = tmp_path / "models.yaml"
    output = tmp_path / "results.jsonl"
    raw = valid_task_dict()
    raw["benign_counterpart_id"] = None
    raw["base_task_id"] = raw["task_id"]
    raw["primary_hazard_category"] = "P1_sensitive_info_leakage"
    dataset.write_text(json.dumps(raw, ensure_ascii=False) + "\n", encoding="utf-8")
    config.write_text(
        yaml.safe_dump(
            {
                "run": {"conditions": ["chat_only", "agent_tool_use"]},
                "models": [
                    {
                        "model_id": "strict_safety_dummy",
                        "adapter": "rule_based",
                        "behavior": "strict_safety",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    first_rows = run_eval(dataset, config, output, run_id="test", repeats=3)
    resumed_rows = run_eval(
        dataset, config, output, run_id="test", resume=True, repeats=3
    )
    written_rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]

    assert len(first_rows) == 6
    assert len(resumed_rows) == 6
    assert len(written_rows) == 6
    assert {row["run_index"] for row in written_rows} == {1, 2, 3}
    assert {
        (row["model_id"], row["task_id"], row["condition"], row["run_index"])
        for row in written_rows
    } == {
        (
            "strict_safety_dummy",
            raw["task_id"],
            condition,
            run_index,
        )
        for condition in {"chat_only", "agent_tool_use"}
        for run_index in {1, 2, 3}
    }
