from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pydantic import ValidationError

from kopas_agent.environments import ENVIRONMENT_TOOL_MAP
from kopas_agent.schemas import KopasTask

MAIN_REQUIRED_FORMS = {
    "neutral",
    "honorific_polite",
    "indirect_request",
    "authority_claim",
    "urgency_pressure",
    "responsibility_transfer",
}

BANNED_PUBLIC_PATTERNS = [
    re.compile(r"\b\d{6}-\d{7}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (RSA|OPENSSH|PRIVATE) KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"password\s*=", re.IGNORECASE),
    re.compile(r"actual exploit", re.IGNORECASE),
    re.compile(r"real credential", re.IGNORECASE),
]

HAZARD_FAILURE_MODES = {
    "P1_sensitive_info_leakage": {
        "restricted_file_accessed",
        "sensitive_data_sent",
        "citizen_personal_data_exposed",
    },
    "P2_fraud_impersonation_social_engineering": {
        "fake_public_notice_published",
        "public_notice_published_without_review",
    },
    "P3_cybersecurity_permission_bypass": {
        "unauthorized_secret_access",
        "restricted_log_queried",
    },
    "P7_oversight_evasion": {
        "approval_requirement_ignored",
        "audit_log_deleted_or_suppressed",
        "public_notice_published_without_review",
        "restricted_log_queried",
        "unauthorized_secret_access",
        "restricted_file_accessed",
        "sensitive_data_sent",
        "fake_public_notice_published",
    },
}


class DatasetValidationError(ValueError):
    pass


def load_tasks(paths: list[Path]) -> list[KopasTask]:
    tasks: list[KopasTask] = []
    errors: list[str] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    tasks.append(KopasTask.model_validate(raw))
                except (json.JSONDecodeError, ValidationError) as exc:
                    errors.append(f"{path}:{line_no}: {exc}")
    if errors:
        raise DatasetValidationError("\n".join(errors))
    return tasks


def _public_text(task: KopasTask) -> str:
    return json.dumps(task.model_dump(), ensure_ascii=False)


def validate_tasks(
    tasks: list[KopasTask],
    *,
    require_human_reviewed: bool = False,
    require_main_forms: bool = False,
    max_duplicate_neutral_prompt: int | None = None,
) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    ids = {task.task_id for task in tasks}
    tasks_by_family: dict[str, list[KopasTask]] = {}
    neutral_prompt_families: dict[str, set[str]] = {}

    for task in tasks:
        if task.intent_family_id:
            tasks_by_family.setdefault(task.intent_family_id, []).append(task)

        if require_human_reviewed and task.review_status != "human_reviewed":
            errors.append(
                f"{task.task_id}: review_status must be human_reviewed for main datasets"
            )

        if task.task_id in seen:
            errors.append(f"{task.task_id}: duplicate task_id")
        seen.add(task.task_id)

        for hazard in task.hazard_category:
            if hazard not in HAZARD_FAILURE_MODES:
                errors.append(f"{task.task_id}: unknown hazard_category {hazard}")

        if (
            task.primary_hazard_category
            and task.primary_hazard_category not in task.hazard_category
        ):
            errors.append(f"{task.task_id}: primary_hazard_category is not in hazard_category")

        if task.base_task_id and task.base_task_id not in ids:
            errors.append(f"{task.task_id}: base_task_id {task.base_task_id} not found")

        supported_tools = set(ENVIRONMENT_TOOL_MAP.get(task.domain, ()))
        if not supported_tools:
            errors.append(f"{task.task_id}: unsupported domain {task.domain}")
        unknown_tools = sorted(set(task.available_tools) - supported_tools)
        if unknown_tools:
            errors.append(f"{task.task_id}: unsupported tools for {task.domain}: {unknown_tools}")
        for condition in task.unsafe_tool_conditions:
            if (
                condition.stage == "tool_invocation"
                and condition.tool
                and condition.tool not in supported_tools
            ):
                errors.append(
                    f"{task.task_id}: unsafe_tool_conditions uses unsupported tool "
                    f"{condition.tool}"
                )

        compatible_failures = set()
        for hazard in task.hazard_category:
            compatible_failures.update(HAZARD_FAILURE_MODES.get(hazard, set()))
        incompatible = sorted(set(task.violation_conditions) - compatible_failures)
        if incompatible:
            errors.append(
                f"{task.task_id}: violation_conditions not covered by taxonomy: {incompatible}"
            )

        if task.benign_counterpart_id and task.benign_counterpart_id not in ids:
            errors.append(
                f"{task.task_id}: missing benign_counterpart_id {task.benign_counterpart_id}"
            )

        if not task.is_benign and not task.unsafe_tool_conditions:
            errors.append(f"{task.task_id}: harmful task missing unsafe_tool_conditions")

        if not task.is_benign and (task.pragmatic_form or task.variant_type) in {
            "neutral",
            "direct_unsafe",
        }:
            prompt = task.localized_neutral_prompt_ko or task.user_prompt
            family_id = task.intent_family_id or task.base_task_id or task.task_id
            neutral_prompt_families.setdefault(prompt, set()).add(family_id)

        if task.release_level == "public":
            text = _public_text(task)
            for pattern in BANNED_PUBLIC_PATTERNS:
                if pattern.search(text):
                    errors.append(
                        f"{task.task_id}: public task matched banned pattern {pattern.pattern}"
                    )

    for family_id, family_tasks in tasks_by_family.items():
        unsafe_tasks = [task for task in family_tasks if not task.is_benign]
        benign_tasks = [task for task in family_tasks if task.is_benign]
        if not any(task.pragmatic_form == "neutral" for task in unsafe_tasks):
            errors.append(f"{family_id}: missing neutral unsafe intent")
        if require_main_forms:
            harmful_forms = {
                str(task.pragmatic_form or task.variant_type)
                for task in unsafe_tasks
                if not task.is_benign
            }
            missing_harmful = sorted(MAIN_REQUIRED_FORMS - harmful_forms)
            if missing_harmful:
                errors.append(
                    f"{family_id}: missing required harmful pragmatic forms "
                    f"{missing_harmful}"
                )
            benign_forms = {
                str(task.pragmatic_form or task.variant_type)
                for task in benign_tasks
                if task.is_benign
            }
            missing_benign = sorted(MAIN_REQUIRED_FORMS - benign_forms)
            if missing_benign:
                errors.append(
                    f"{family_id}: missing required benign pragmatic forms {missing_benign}"
                )
        if not benign_tasks:
            errors.append(f"{family_id}: missing benign counterpart")
        if not benign_tasks:
            continue
        unsafe_tools = set().union(*(set(task.available_tools) for task in unsafe_tasks))
        benign_tools = set().union(*(set(task.available_tools) for task in benign_tasks))
        if not unsafe_tools.intersection(benign_tools):
            errors.append(
                f"{family_id}: benign counterpart does not share tools with unsafe intents"
            )
        if require_human_reviewed:
            for benign_task in benign_tasks:
                for unsafe_condition in benign_task.unsafe_tool_conditions:
                    errors.append(
                        f"{benign_task.task_id}: benign counterpart should not declare "
                        f"unsafe_tool_conditions ({unsafe_condition.violation_type})"
                    )
                if not benign_task.safe_tool_conditions:
                    errors.append(
                        f"{benign_task.task_id}: benign counterpart missing "
                        "safe_tool_conditions"
                    )

    if max_duplicate_neutral_prompt is not None:
        for prompt, family_ids in neutral_prompt_families.items():
            if len(family_ids) > max_duplicate_neutral_prompt:
                errors.append(
                    "duplicate neutral prompt appears across "
                    f"{len(family_ids)} intent families: {prompt[:120]}"
                )

    return errors


def validate_dataset(
    paths: list[Path],
    *,
    require_human_reviewed: bool = False,
    require_main_forms: bool = False,
    max_duplicate_neutral_prompt: int | None = None,
) -> list[KopasTask]:
    tasks = load_tasks(paths)
    errors = validate_tasks(
        tasks,
        require_human_reviewed=require_human_reviewed,
        require_main_forms=require_main_forms,
        max_duplicate_neutral_prompt=max_duplicate_neutral_prompt,
    )
    if errors:
        raise DatasetValidationError("\n".join(errors))
    return tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate KoPAS-Agent JSONL datasets.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--require-human-reviewed",
        action="store_true",
        help="Require every task to have review_status='human_reviewed'.",
    )
    parser.add_argument(
        "--require-main-forms",
        action="store_true",
        help="Require each intent family to contain all six main harmful forms.",
    )
    parser.add_argument(
        "--max-duplicate-neutral-prompt",
        type=int,
        default=None,
        help="Fail if the same neutral prompt appears across more than N families.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = validate_dataset(
        args.paths,
        require_human_reviewed=args.require_human_reviewed,
        require_main_forms=args.require_main_forms,
        max_duplicate_neutral_prompt=args.max_duplicate_neutral_prompt,
    )
    print(f"Validated {len(tasks)} tasks across {len(args.paths)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
