from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from kopas_agent.environments import create_environment
from kopas_agent.eval_runner.model_adapters import create_adapter
from kopas_agent.schemas import KopasTask
from kopas_agent.scorers.chat_response_scorer import score_chat_response
from kopas_agent.scorers.response_rubric_scorer import score_response_rubric
from kopas_agent.scorers.state_diff_scorer import StateDiffScore, score_state_diff


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                msg = f"{path}:{line_no}: invalid JSON: {exc}"
                raise ValueError(msg) from exc
    return rows


def safe_json_dumps(row: dict[str, Any]) -> str:
    dumped = json.dumps(row, ensure_ascii=False)
    return dumped.encode("utf-8", errors="replace").decode("utf-8")


def run_eval(
    dataset: Path,
    model_config_path: Path,
    output: Path,
    run_id: str | None = None,
    limit: int | None = None,
    resume: bool = False,
    repeats: int = 1,
    task_ids_file: Path | None = None,
    run_indexes: set[int] | None = None,
    resume_reference: Path | None = None,
    continue_on_error: bool = False,
    retry_provider_errors: bool = False,
) -> list[dict]:
    config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    model_configs = config.get("models", [])
    conditions = config.get("run", {}).get("conditions", ["agent_tool_use"])
    if not model_configs:
        msg = "model config must include at least one model"
        raise ValueError(msg)
    if repeats < 1:
        msg = "repeats must be at least 1"
        raise ValueError(msg)

    run_id = run_id or f"kopas_{datetime.now(UTC).strftime('%Y_%m_%d_%H%M%S')}"
    tasks = [KopasTask.model_validate(row) for row in load_jsonl(dataset)]
    if task_ids_file is not None:
        task_ids = {
            line.strip()
            for line in task_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        tasks = [task for task in tasks if task.task_id in task_ids]
    if limit is not None:
        tasks = tasks[:limit]
    output.parent.mkdir(parents=True, exist_ok=True)

    results = []
    completed_keys: set[tuple[str, str, str, int]] = set()
    completed_sources = []
    if resume_reference is not None and resume_reference.exists():
        completed_sources.append(resume_reference)
    if resume and output.exists():
        completed_sources.append(output)
    for completed_source in completed_sources:
        for row in load_jsonl(completed_source):
            results.append(row)
            if retry_provider_errors and row.get("adapter_metadata", {}).get("provider_error"):
                continue
            run_index = int(row.get("run_index", 1))
            completed_keys.add(
                (
                    str(row.get("model_id", "")),
                    str(row.get("task_id", "")),
                    str(row.get("condition", "")),
                    run_index,
                )
            )

    mode = "a" if resume and output.exists() else "w"
    with output.open(mode, encoding="utf-8") as handle:
        for model_config in model_configs:
            adapter = create_adapter(model_config)
            selected_run_indexes = run_indexes or set(range(1, repeats + 1))
            for run_index in range(1, repeats + 1):
                if run_index not in selected_run_indexes:
                    continue
                for task in tasks:
                    for condition in conditions:
                        key = (
                            str(adapter.model_id),
                            task.task_id,
                            str(condition),
                            run_index,
                        )
                        if key in completed_keys:
                            continue
                        try:
                            if condition == "chat_only":
                                row = run_chat_condition(run_id, adapter, task, run_index)
                            elif condition == "agent_tool_use":
                                row = run_agent_condition(run_id, adapter, task, run_index)
                            else:
                                msg = f"Unsupported eval condition: {condition}"
                                raise ValueError(msg)
                        except Exception as exc:
                            if not continue_on_error:
                                raise
                            row = run_error_condition(
                                run_id,
                                adapter,
                                task,
                                str(condition),
                                run_index,
                                exc,
                            )
                        results.append(row)
                        completed_keys.add(key)
                        handle.write(safe_json_dumps(row) + "\n")
    return results


def common_row(
    run_id: str,
    model_id: str,
    task: KopasTask,
    condition: str,
    run_index: int = 1,
) -> dict[str, Any]:
    task_input = {
        "user_prompt": task.user_prompt,
        "agent_goal": task.agent_goal,
        "available_tools": task.available_tools,
        "environment_state": task.environment_state,
        "safe_behavior": task.safe_behavior,
        "violation_conditions": task.violation_conditions,
        "unsafe_tool_conditions": [
            condition.model_dump() for condition in task.unsafe_tool_conditions
        ],
        "rubric": task.rubric.model_dump(),
        "turns": [turn.model_dump() for turn in task.turns],
    }
    return {
        "run_id": run_id,
        "run_index": run_index,
        "model_id": model_id,
        "condition": condition,
        "task_id": task.task_id,
        "domain": task.domain,
        "hazard_category": task.hazard_category,
        "primary_hazard_category": task.primary_hazard_category or task.hazard_category[0],
        "base_task_id": task.base_task_id or task.task_id,
        "intent_id": task.intent_id or task.task_id,
        "intent_family_id": task.intent_family_id or task.base_task_id or task.task_id,
        "harm_category": (
            task.harm_category or task.primary_hazard_category or task.hazard_category[0]
        ),
        "base_intent": task.base_intent or task.user_prompt,
        "pragmatic_form": task.pragmatic_form or task.variant_type,
        "is_benign": task.is_benign or task.variant_type == "benign_counterpart",
        "source_dataset": task.source_dataset,
        "source_split": task.source_split,
        "source_item_id": task.source_item_id,
        "source_category": task.source_category,
        "source_name": task.source_name,
        "source_target_functions": task.source_target_functions,
        "source_grading_function": task.source_grading_function,
        "abstract_intent": task.abstract_intent,
        "localized_neutral_prompt_ko": task.localized_neutral_prompt_ko,
        "variant_type": task.variant_type,
        "turn_count": len(task.turns) if task.turns else 1,
        "user_prompt": task.user_prompt,
        "input": task_input,
    }


def run_chat_condition(
    run_id: str,
    adapter,
    task: KopasTask,
    run_index: int = 1,
) -> dict[str, Any]:
    agent_result = adapter.run_chat(task)
    chat_score = score_chat_response(task, agent_result.final_response)
    state_diff_score = StateDiffScore(material_violation=False, tool_misuse=False)
    rubric_score = score_response_rubric(
        task=task,
        model_final_response=agent_result.final_response,
        action_log=[],
        state_diff_score=state_diff_score,
    )
    row = common_row(run_id, adapter.model_id, task, "chat_only", run_index)
    row.update(
        {
            "final_response": agent_result.final_response,
            "action_log": [],
            "turn_action_log": [],
            "escalation_stage_of_failure": None,
            "state_diff_score": state_diff_score.model_dump(),
            "chat_response_score": chat_score.model_dump(),
            "rubric_score": rubric_score.model_dump(),
            "adapter_metadata": agent_result.metadata,
        }
    )
    return row


def run_agent_condition(
    run_id: str,
    adapter,
    task: KopasTask,
    run_index: int = 1,
) -> dict[str, Any]:
    env = create_environment(task)
    before = env.snapshot()
    final_result = None
    turn_action_log = []
    escalation_stage_of_failure = None

    if task.turns:
        last_seen_actions = 0
        user_turn_number = 0
        for turn_index, turn in enumerate(task.turns):
            if turn.role != "user":
                continue
            user_turn_number += 1
            final_result = adapter.run_agent_turn(task, env, turn_index)
            current = env.snapshot()
            incremental_actions = current["action_log"][last_seen_actions:]
            last_seen_actions = len(current["action_log"])
            turn_action_log.append(
                {
                    "turn_index": turn_index,
                    "user_turn_number": user_turn_number,
                    "user_content": turn.content,
                    "actions": incremental_actions,
                    "assistant_response": final_result.final_response,
                }
            )
            interim_score = score_state_diff(task, before, current)
            if interim_score.material_violation and escalation_stage_of_failure is None:
                escalation_stage_of_failure = user_turn_number
    else:
        final_result = adapter.run(task, env)

    if final_result is None:
        final_result = adapter.run(task, env)

    after = env.snapshot()
    state_diff_score = score_state_diff(task, before, after)
    rubric_score = score_response_rubric(
        task=task,
        model_final_response=final_result.final_response,
        action_log=after["action_log"],
        state_diff_score=state_diff_score,
    )
    chat_score = score_chat_response(task, final_result.final_response)
    row = common_row(run_id, adapter.model_id, task, "agent_tool_use", run_index)
    row.update(
        {
            "final_response": final_result.final_response,
            "action_log": after["action_log"],
            "turn_action_log": turn_action_log,
            "escalation_stage_of_failure": escalation_stage_of_failure,
            "state_diff_score": state_diff_score.model_dump(),
            "chat_response_score": chat_score.model_dump(),
            "rubric_score": rubric_score.model_dump(),
            "adapter_metadata": final_result.metadata,
        }
    )
    return row


def run_error_condition(
    run_id: str,
    adapter,
    task: KopasTask,
    condition: str,
    run_index: int,
    exc: Exception,
) -> dict[str, Any]:
    final_response = "[PROVIDER_OR_ADAPTER_ERROR]"
    state_diff_score = StateDiffScore(material_violation=False, tool_misuse=False)
    chat_score = score_chat_response(task, final_response)
    rubric_score = score_response_rubric(
        task=task,
        model_final_response=final_response,
        action_log=[],
        state_diff_score=state_diff_score,
    )
    row = common_row(run_id, adapter.model_id, task, condition, run_index)
    row.update(
        {
            "final_response": final_response,
            "action_log": [],
            "turn_action_log": [],
            "escalation_stage_of_failure": None,
            "state_diff_score": state_diff_score.model_dump(),
            "chat_response_score": chat_score.model_dump(),
            "rubric_score": rubric_score.model_dump(),
            "adapter_metadata": {
                "adapter": getattr(adapter, "adapter_label", adapter.__class__.__name__),
                "model": getattr(adapter, "model_name", adapter.model_id),
                "condition": condition,
                "provider_error": True,
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            },
        }
    )
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run KoPAS-Agent local MVP evaluation.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N tasks. Useful for paid API smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Append to an existing output file and skip completed "
            "(model_id, task_id, condition, run_index) rows."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "Repeat each (model_id, task_id, condition) evaluation N times. "
            "Each row records a 1-based run_index."
        ),
    )
    parser.add_argument(
        "--task-ids-file",
        type=Path,
        default=None,
        help="Evaluate only task IDs listed one per line in this file.",
    )
    parser.add_argument(
        "--run-indexes",
        default=None,
        help="Comma-separated run indexes to execute, e.g. 3 or 1,3.",
    )
    parser.add_argument(
        "--resume-reference",
        type=Path,
        default=None,
        help=(
            "Read completed keys from this JSONL file without writing to it. "
            "Useful for safe parallel shard outputs."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write an error row and continue when a model/provider request fails.",
    )
    parser.add_argument(
        "--retry-provider-errors",
        action="store_true",
        help=(
            "When resuming, do not count existing provider_error rows as completed. "
            "Use with --continue-on-error to retry provider-blocked or failed rows."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_indexes = None
    if args.run_indexes:
        run_indexes = {int(part.strip()) for part in args.run_indexes.split(",") if part.strip()}
    results = run_eval(
        args.dataset,
        args.model_config,
        args.output,
        args.run_id,
        args.limit,
        resume=args.resume,
        repeats=args.repeats,
        task_ids_file=args.task_ids_file,
        run_indexes=run_indexes,
        resume_reference=args.resume_reference,
        continue_on_error=args.continue_on_error,
        retry_provider_errors=args.retry_provider_errors,
    )
    print(f"Wrote {len(results)} result rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
