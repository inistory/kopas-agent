from __future__ import annotations

import json
from pathlib import Path

import pytest

from kopas_agent.environments import create_environment
from kopas_agent.eval_runner.model_adapters import RecordedTraceAdapter
from kopas_agent.schemas import KopasTask
from tests.test_schemas import valid_task_dict


def test_recorded_trace_rejects_unavailable_tool(tmp_path: Path) -> None:
    raw = valid_task_dict()
    task = KopasTask.model_validate(raw)
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "model_id": "recorded",
                "task_id": task.task_id,
                "condition": "agent_tool_use",
                "final_response": "synthetic trace",
                "tool_calls": [{"tool_name": "not_available", "args": {}}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    adapter = RecordedTraceAdapter("recorded", {"trace_path": str(trace_path)})

    with pytest.raises(ValueError, match="unavailable tool"):
        adapter.run(task, create_environment(task))
