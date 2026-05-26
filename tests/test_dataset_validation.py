from __future__ import annotations

import json
from pathlib import Path

import pytest

from kopas_agent.dataset_tools.validate_dataset import (
    DatasetValidationError,
    validate_dataset,
    validate_tasks,
)
from kopas_agent.schemas import KopasTask
from tests.test_schemas import valid_task_dict


def test_main_dataset_validates() -> None:
    tasks = validate_dataset([Path("datasets/kopas_pragmatic_main.jsonl")])
    assert len(tasks) == 624
    assert len({task.intent_family_id for task in tasks}) == 52
    assert {task.variant_type for task in tasks} >= {
        "direct_unsafe",
        "honorific_polite",
        "indirect_request",
        "authority_claim",
        "urgency_pressure",
        "responsibility_transfer",
        "benign_counterpart",
    }


def test_duplicate_task_id_detected() -> None:
    raw = valid_task_dict()
    task = KopasTask.model_validate(raw)
    errors = validate_tasks([task, task])
    assert any("duplicate task_id" in error for error in errors)


def test_missing_counterpart_detected() -> None:
    raw = valid_task_dict()
    task = KopasTask.model_validate(raw)
    errors = validate_tasks([task])
    assert any("missing benign_counterpart_id" in error for error in errors)


def test_primary_hazard_must_be_one_of_task_hazards() -> None:
    raw = valid_task_dict()
    raw["primary_hazard_category"] = "P3_cybersecurity_permission_bypass"
    raw["benign_counterpart_id"] = None
    task = KopasTask.model_validate(raw)
    errors = validate_tasks([task])
    assert any("primary_hazard_category" in error for error in errors)


def test_cli_loader_reports_validation_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    bad = valid_task_dict()
    bad["available_tools"] = ["unsupported_tool"]
    bad["benign_counterpart_id"] = None
    path.write_text(json.dumps(bad, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        validate_dataset([path])
