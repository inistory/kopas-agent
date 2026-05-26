from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from kopas_agent.dataset_tools.synthetic_helpers import base_task, rubric, write_jsonl

CONTEXT_BY_DOMAIN = {
    "mock_workspace": "내부 자료 처리",
    "mock_public_service": "민원 대응",
    "mock_security_ops": "보안 업무",
}

VARIANT_TYPE_BY_FORM = {
    "neutral": "direct_unsafe",
    "honorific_polite": "honorific_polite",
    "indirect_request": "indirect_request",
    "authority_claim": "authority_claim",
    "urgency_pressure": "urgency_pressure",
    "responsibility_transfer": "responsibility_transfer",
    "institutional_framing": "institutional_framing",
    "ko_en_code_switching": "ko_en_code_switching",
}

SUFFIX_BY_FORM = {
    "neutral": "neutral",
    "honorific_polite": "honorific",
    "indirect_request": "indirect",
    "authority_claim": "authority",
    "urgency_pressure": "urgency",
    "responsibility_transfer": "responsibility",
    "institutional_framing": "institutional",
    "ko_en_code_switching": "codeswitch",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_templates(path: Path, main_only: bool) -> tuple[list[str], dict[str, str]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    templates = raw.get("templates", {})
    forms = raw.get("main_pragmatic_forms", []) if main_only else [
        *raw.get("main_pragmatic_forms", []),
        *raw.get("supplemental_pragmatic_forms", []),
    ]
    missing = [form for form in forms if form not in templates]
    if missing:
        msg = f"Missing pragmatic templates: {missing}"
        raise ValueError(msg)
    return forms, templates


def render_prompt(seed: dict[str, Any], form: str, template: str) -> str:
    base_prompt = seed["neutral_prompt"]
    context = seed.get("context") or CONTEXT_BY_DOMAIN.get(seed["domain"], "업무 처리")
    return template.format(
        base_prompt=base_prompt,
        context=context,
        base_intent=seed.get("base_intent", base_prompt),
    )


def render_benign_prompt(seed: dict[str, Any], form: str, template: str) -> str:
    base_prompt = seed["benign_prompt"]
    context = seed.get("context") or CONTEXT_BY_DOMAIN.get(seed["domain"], "업무 처리")
    return template.format(
        base_prompt=base_prompt,
        context=context,
        base_intent=seed.get("base_intent", base_prompt),
    )


def seed_to_base_task(seed: dict[str, Any]) -> dict[str, Any]:
    hazards = seed.get("hazards") or seed.get("hazard_category")
    primary_hazard = seed.get("primary_hazard") or seed.get("primary_hazard_category")
    task = base_task(
        task_id=f"{seed['intent_family_id']}_neutral",
        domain=seed["domain"],
        hazards=hazards,
        prompt=seed["neutral_prompt"],
        tools=seed["available_tools"],
        env_state=seed["environment_state"],
        safe_behavior=seed["safe_behavior"],
        violations=seed["violation_conditions"],
        rubric_kind=seed.get("rubric_kind", "workspace"),
        primary_hazard=primary_hazard,
    )
    if seed.get("unsafe_tool_conditions") is not None:
        task["unsafe_tool_conditions"] = seed["unsafe_tool_conditions"]
    task["intent_family_id"] = seed["intent_family_id"]
    task["harm_category"] = seed["harm_category"]
    task["base_intent"] = seed["base_intent"]
    task["localized_neutral_prompt_ko"] = seed["neutral_prompt"]
    task["localized_benign_prompt_ko"] = seed["benign_prompt"]
    task["base_prompt_ko"] = seed["neutral_prompt"]
    task["review_status"] = seed.get("review_status", "human_reviewed")
    task["dataset_version"] = seed.get("dataset_version", "reviewed_main_v1")
    task["review_notes"] = seed.get("review_notes")
    task["source_dataset"] = seed.get("source_dataset")
    task["source_split"] = seed.get("source_split")
    task["source_item_id"] = seed.get("source_item_id")
    task["source_category"] = seed.get("source_category")
    task["source_name"] = seed.get("source_name")
    task["source_intent_summary_ko"] = seed.get("source_intent_summary_ko")
    task["abstract_intent"] = seed.get("abstract_intent")
    task["construction_method"] = seed.get("construction_method")
    task["rewrite_method"] = seed.get("rewrite_method")
    task["rewrite_model"] = seed.get("rewrite_model")
    task["rewrite_prompt_version"] = seed.get("rewrite_prompt_version")
    task["translation_provider"] = seed.get("translation_provider")
    task["translation_model"] = seed.get("translation_model")
    task["translation_prompt_version"] = seed.get("translation_prompt_version")
    task["translation_method"] = seed.get("translation_method")
    task["benign_counterpart_id"] = f"{seed['intent_family_id']}_benign_neutral"
    if seed.get("rubric"):
        task["rubric"] = seed["rubric"]
    return task


def make_harmful_variants(
    seed: dict[str, Any], forms: list[str], templates: dict[str, str]
) -> list[dict[str, Any]]:
    base = seed_to_base_task(seed)
    variants = []
    for form in forms:
        task = deepcopy(base)
        task["task_id"] = f"{seed['intent_family_id']}_{SUFFIX_BY_FORM[form]}"
        task["base_task_id"] = f"{seed['intent_family_id']}_neutral"
        task["intent_id"] = task["task_id"]
        task["intent_family_id"] = seed["intent_family_id"]
        task["variant_type"] = VARIANT_TYPE_BY_FORM[form]
        task["pragmatic_form"] = form
        task["is_benign"] = False
        task["user_prompt"] = render_prompt(seed, form, templates[form])
        task["surface_prompt_ko"] = task["user_prompt"]
        task["language"] = "ko-en" if form == "ko_en_code_switching" else "ko"
        task["environment_state"]["pragmatic_form"] = form
        task["benign_counterpart_id"] = f"{seed['intent_family_id']}_benign_neutral"
        variants.append(task)
    return variants


def make_benign_variant(seed: dict[str, Any], form: str, template: str) -> dict[str, Any]:
    hazards = seed.get("hazards") or seed.get("hazard_category")
    primary_hazard = seed.get("primary_hazard") or seed.get("primary_hazard_category")
    benign = {
        "task_id": f"{seed['intent_family_id']}_benign_{SUFFIX_BY_FORM[form]}",
        "base_task_id": f"{seed['intent_family_id']}_neutral",
        "intent_id": f"{seed['intent_family_id']}_benign_{SUFFIX_BY_FORM[form]}",
        "intent_family_id": seed["intent_family_id"],
        "harm_category": seed["harm_category"],
        "base_intent": seed["base_intent"],
        "pragmatic_form": form,
        "is_benign": True,
        "split": seed.get("split", "public_dev"),
        "release_level": seed.get("release_level", "public"),
        "hazard_category": hazards,
        "primary_hazard_category": primary_hazard or hazards[0],
        "domain": seed["domain"],
        "language": "ko-en" if form == "ko_en_code_switching" else "ko",
        "variant_type": "benign_counterpart",
        "user_prompt": render_benign_prompt(seed, form, template),
        "agent_goal": "Respond to the user's request while following safety policies.",
        "available_tools": seed.get("benign_available_tools", seed["available_tools"]),
        "environment_state": seed.get("benign_environment_state", seed["environment_state"]),
        "safe_behavior": seed.get("benign_safe_behavior", seed["safe_behavior"]),
        "violation_conditions": seed.get(
            "benign_violation_conditions", seed["violation_conditions"][:1]
        ),
        "unsafe_tool_conditions": seed.get("benign_unsafe_tool_conditions", []),
        "safe_tool_conditions": seed.get("safe_tool_conditions", []),
        "turns": [],
        "benign_counterpart_id": None,
        "rubric": seed.get("rubric", rubric(seed.get("rubric_kind", "workspace"))),
        "review_status": seed.get("review_status", "human_reviewed"),
        "dataset_version": seed.get("dataset_version", "reviewed_main_v1"),
        "review_notes": seed.get("review_notes"),
        "source_dataset": seed.get("source_dataset"),
        "source_split": seed.get("source_split"),
        "source_item_id": seed.get("source_item_id"),
        "source_category": seed.get("source_category"),
        "source_name": seed.get("source_name"),
        "source_intent_summary_ko": seed.get("source_intent_summary_ko"),
        "abstract_intent": seed.get("abstract_intent"),
        "construction_method": seed.get("construction_method"),
        "rewrite_method": seed.get("rewrite_method"),
        "rewrite_model": seed.get("rewrite_model"),
        "rewrite_prompt_version": seed.get("rewrite_prompt_version"),
        "translation_provider": seed.get("translation_provider"),
        "translation_model": seed.get("translation_model"),
        "translation_prompt_version": seed.get("translation_prompt_version"),
        "translation_method": seed.get("translation_method"),
        "localized_neutral_prompt_ko": seed["neutral_prompt"],
        "localized_benign_prompt_ko": seed["benign_prompt"],
        "base_prompt_ko": seed["benign_prompt"],
        "surface_prompt_ko": render_benign_prompt(seed, form, template),
    }
    return benign


def make_benign_variants(
    seed: dict[str, Any], forms: list[str], templates: dict[str, str]
) -> list[dict[str, Any]]:
    return [make_benign_variant(seed, form, templates[form]) for form in forms]


def generate_dataset(
    seeds_path: Path,
    templates_path: Path,
    output_path: Path,
    main_only: bool,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    forms, templates = load_templates(templates_path, main_only)
    seeds = load_jsonl(seeds_path)
    if limit is not None:
        seeds = seeds[:limit]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        if seed.get("review_status") == "excluded":
            continue
        rows.extend(make_harmful_variants(seed, forms, templates))
        rows.extend(make_benign_variants(seed, forms, templates))
    write_jsonl(output_path, rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate reviewed KoPAS pragmatic dataset.")
    parser.add_argument("--seeds", required=True, type=Path)
    parser.add_argument("--templates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = generate_dataset(
        seeds_path=args.seeds,
        templates_path=args.templates,
        output_path=args.output,
        main_only=args.main_only,
        limit=args.limit,
    )
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
