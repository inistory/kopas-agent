from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from kopas_agent.scorers.metrics import (
    MAIN_PRAGMATIC_FORMS,
    PRAGMATIC_PRESSURE_VARIANTS,
    agentic_gap,
    compute_metric_summary,
)

METRIC_FIELDS = [
    "n",
    "MVR",
    "SWVR",
    "TMR",
    "SCR",
    "benign_tool_success_rate",
    "benign_abort_rate",
    "ORR",
    "SGR",
    "HRR",
    "unsafe_response_rate",
    "unsafe_tool_invocation_rate",
    "unsafe_completion_rate",
    "unsafe_state_violation_rate",
    "unsafe_behavior_rate",
    "response_action_mismatch_rate",
    "RR",
    "SER",
    "KPV_delta",
    "paired_KPV_delta",
    "response_delta",
    "behavior_invocation_delta",
    "behavior_completion_delta",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _secondary_output_path(output: Path, default_name: str) -> Path:
    if output.name == "mvp_metrics.csv":
        return output.parent / default_name

    prefix = output.stem
    if prefix.endswith("_metrics"):
        prefix = prefix[: -len("_metrics")]

    names = {
        "mvp_metrics_by_hazard.csv": f"{prefix}_metrics_by_hazard.csv",
        "mvp_metrics_by_primary_hazard.csv": f"{prefix}_metrics_by_primary_hazard.csv",
        "mvp_metrics_by_variant.csv": f"{prefix}_metrics_by_variant.csv",
        "paired_variant_effects.csv": f"{prefix}_paired_variant_effects.csv",
        "agentic_gap_table.csv": f"{prefix}_agentic_gap_table.csv",
        "paper_response_vs_behavior.csv": f"{prefix}_paper_response_vs_behavior.csv",
        "paper_benign_by_form.csv": f"{prefix}_paper_benign_by_form.csv",
        "paper_pragmatic_variant_effects.csv": f"{prefix}_paper_pragmatic_variant_effects.csv",
        "paper_response_action_mismatch.csv": f"{prefix}_paper_response_action_mismatch.csv",
        "paper_by_harm_category.csv": f"{prefix}_paper_by_harm_category.csv",
        "paper_main_delta_by_model.csv": f"{prefix}_paper_main_delta_by_model.csv",
        "paper_delta_by_family.csv": f"{prefix}_paper_delta_by_family.csv",
    }
    return output.parent / names[default_name]


def _round_metrics(summary: dict[str, float]) -> dict[str, Any]:
    rounded = {}
    for key, value in summary.items():
        if key == "n":
            rounded[key] = int(value)
        else:
            rounded[key] = round(value, 6)
    return rounded


def group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        values = row.get(key)
        if isinstance(values, list):
            for value in values:
                grouped[value].append(row)
        else:
            grouped[str(values)].append(row)
    return grouped


def group_by_fields(
    rows: list[dict[str, Any]], keys: list[str]
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "")) for key in keys)].append(row)
    return grouped


def build_metric_tables(
    rows: list[dict[str, Any]],
    model_config: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    _ = model_config
    model_table = []
    for (model_id, condition), model_rows in sorted(
        group_by_fields(rows, ["model_id", "condition"]).items()
    ):
        summary = _round_metrics(compute_metric_summary(model_rows))
        summary["Model"] = model_id
        summary["Condition"] = condition
        model_table.append(summary)

    hazard_table = []
    for (condition, hazard), hazard_rows in sorted(
        _group_multilabel_by_condition(rows, "hazard_category").items()
    ):
        summary = _round_metrics(compute_metric_summary(hazard_rows))
        hazard_table.append({"Condition": condition, "Hazard": hazard, **summary})

    primary_hazard_table = []
    for (condition, hazard), hazard_rows in sorted(
        group_by_fields(rows, ["condition", "primary_hazard_category"]).items()
    ):
        summary = _round_metrics(compute_metric_summary(hazard_rows))
        primary_hazard_table.append({"Condition": condition, "Primary Hazard": hazard, **summary})

    variant_table = []
    direct_by_model = {}
    for (model_id, condition), model_rows in group_by_fields(
        rows, ["model_id", "condition"]
    ).items():
        direct_rows = [row for row in model_rows if row.get("variant_type") == "direct_unsafe"]
        direct_by_model[(model_id, condition)] = compute_metric_summary(direct_rows)["MVR"]
    for (condition, variant), variant_rows in sorted(
        group_by_fields(rows, ["condition", "variant_type"]).items()
    ):
        summary = _round_metrics(compute_metric_summary(variant_rows))
        model_deltas = []
        for model_id, model_rows in group_rows(variant_rows, "model_id").items():
            model_deltas.append(
                compute_metric_summary(model_rows)["MVR"]
                - direct_by_model.get((model_id, condition), 0.0)
            )
        delta = sum(model_deltas) / len(model_deltas) if model_deltas else 0.0
        variant_table.append(
            {
                "Condition": condition,
                "Variant": variant,
                "delta_over_direct": round(delta, 6),
                **summary,
            }
        )

    paired_variant_effect_table = build_paired_variant_effect_table(rows)

    gap_table = []
    by_model_condition = {(row["Model"], row["Condition"]): row for row in model_table}
    for model_id in sorted({row["Model"] for row in model_table}):
        chat_row = by_model_condition.get((model_id, "chat_only"), {})
        agent_row = by_model_condition.get((model_id, "agent_tool_use"), {})
        chat_rate = float(chat_row.get("HRR", 0.0))
        agent_mvr = float(agent_row.get("MVR", 0.0))
        gap_table.append(
            {
                "Model": model_id,
                "HRR_chat": chat_rate,
                "MVR_agent": agent_mvr,
                "Agentic Gap": round(agentic_gap(agent_mvr, chat_rate), 6),
            }
        )

    return {
        "model": model_table,
        "hazard": hazard_table,
        "primary_hazard": primary_hazard_table,
        "variant": variant_table,
        "paired_variant_effect": paired_variant_effect_table,
        "agentic_gap": gap_table,
        "paper_response_vs_behavior": build_paper_response_vs_behavior_table(rows),
        "paper_benign_by_form": build_paper_benign_by_form_table(rows),
        "paper_response_action_mismatch": build_paper_mismatch_table(rows),
        "paper_by_harm_category": build_paper_by_harm_category_table(rows),
        "paper_main_delta_by_model": build_paper_main_delta_by_model(rows),
        "paper_delta_by_family": build_paper_delta_by_family(rows),
    }


def _group_multilabel_by_condition(
    rows: list[dict[str, Any]], key: str
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        condition = str(row.get("condition", ""))
        for value in row.get(key, []):
            grouped[(condition, str(value))].append(row)
    return grouped


def build_paired_variant_effect_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = group_by_fields(_paper_rows(rows), ["model_id", "intent_family_id"])
    deltas: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for (model_id, _family_id), family_rows in grouped.items():
        response_neutral = _metric_for_family_form(
            family_rows, "chat_only", "neutral", "unsafe_response_rate"
        )
        invocation_neutral = _metric_for_family_form(
            family_rows, "agent_tool_use", "neutral", "unsafe_tool_invocation_rate"
        )
        completion_neutral = _metric_for_family_form(
            family_rows, "agent_tool_use", "neutral", "unsafe_completion_rate"
        )
        if (
            response_neutral is None
            or invocation_neutral is None
            or completion_neutral is None
        ):
            continue
        for form in sorted(PRAGMATIC_PRESSURE_VARIANTS):
            response_value = _metric_for_family_form(
                family_rows, "chat_only", form, "unsafe_response_rate"
            )
            invocation_value = _metric_for_family_form(
                family_rows, "agent_tool_use", form, "unsafe_tool_invocation_rate"
            )
            completion_value = _metric_for_family_form(
                family_rows, "agent_tool_use", form, "unsafe_completion_rate"
            )
            if response_value is None or invocation_value is None or completion_value is None:
                continue
            deltas[(model_id, form)].append(
                {
                    "response_delta": response_value - response_neutral,
                    "behavior_invocation_delta": invocation_value - invocation_neutral,
                    "behavior_completion_delta": completion_value - completion_neutral,
                }
            )

    table = []
    for (model_id, form), values in sorted(deltas.items()):
        n = len(values)
        table.append(
            {
                "Model": model_id,
                "Pragmatic Form": form,
                "n": n,
                "response_unsafe_delta_over_neutral": round(
                    _mean(value["response_delta"] for value in values), 6
                ),
                "behavior_invocation_delta_over_neutral": round(
                    _mean(value["behavior_invocation_delta"] for value in values), 6
                ),
                "behavior_completion_delta_over_neutral": round(
                    _mean(value["behavior_completion_delta"] for value in values), 6
                ),
            }
        )
    return table


def _paper_form(row: dict[str, Any]) -> str:
    form = row.get("pragmatic_form") or row.get("variant_type", "")
    if form == "direct_unsafe":
        return "neutral"
    return str(form)


def _paper_harm_category(row: dict[str, Any]) -> str:
    return str(row.get("harm_category") or row.get("primary_hazard_category") or "")


def _paper_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if not row.get("is_benign")
        and _paper_form(row) in {"neutral", *MAIN_PRAGMATIC_FORMS}
    ]


def build_paper_response_vs_behavior_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    grouped = group_by_fields(_paper_rows(rows), ["model_id", "pragmatic_form"])
    for (model_id, form), form_rows in sorted(grouped.items()):
        normalized_form = form or _paper_form(form_rows[0])
        if normalized_form == "direct_unsafe":
            normalized_form = "neutral"
        chat_rows = [row for row in form_rows if row.get("condition") == "chat_only"]
        agent_rows = [row for row in form_rows if row.get("condition") == "agent_tool_use"]
        chat_summary = compute_metric_summary(chat_rows)
        agent_summary = compute_metric_summary(agent_rows)
        table.append(
            {
                "Model": model_id,
                "Pragmatic Form": normalized_form,
                "n_response": int(chat_summary["n"]),
                "n_behavior": int(agent_summary["n"]),
                "unsafe_response_rate": round(chat_summary["unsafe_response_rate"], 6),
                "unsafe_tool_invocation_rate": round(
                    agent_summary["unsafe_tool_invocation_rate"], 6
                ),
                "unsafe_completion_rate": round(agent_summary["unsafe_completion_rate"], 6),
                "unsafe_state_violation_rate": round(
                    agent_summary["unsafe_state_violation_rate"], 6
                ),
                "response_action_mismatch_rate": round(
                    agent_summary["response_action_mismatch_rate"], 6
                ),
            }
        )
    return table


def build_paper_benign_by_form_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    benign_rows = [row for row in rows if row.get("is_benign")]
    for (model_id, form, condition), form_rows in sorted(
        group_by_fields(benign_rows, ["model_id", "pragmatic_form", "condition"]).items()
    ):
        summary = compute_metric_summary(form_rows)
        table.append(
            {
                "Model": model_id,
                "Pragmatic Form": form,
                "Condition": condition,
                "n": int(summary["n"]),
                "benign_tool_success_rate": round(
                    summary["benign_tool_success_rate"], 6
                ),
                "safe_completion_rate": round(summary["SCR"], 6),
                "over_refusal_rate": round(summary["ORR"], 6),
                "benign_abort_rate": round(summary["benign_abort_rate"], 6),
            }
        )
    return table


def build_paper_mismatch_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    agent_rows = [row for row in _paper_rows(rows) if row.get("condition") == "agent_tool_use"]
    for (model_id, form), form_rows in sorted(
        group_by_fields(agent_rows, ["model_id", "pragmatic_form"]).items()
    ):
        summary = compute_metric_summary(form_rows)
        table.append(
            {
                "Model": model_id,
                "Pragmatic Form": form or _paper_form(form_rows[0]),
                "n": int(summary["n"]),
                "response_action_mismatch_rate": round(
                    summary["response_action_mismatch_rate"], 6
                ),
                "unsafe_tool_invocation_rate": round(
                    summary["unsafe_tool_invocation_rate"], 6
                ),
                "unsafe_completion_rate": round(summary["unsafe_completion_rate"], 6),
                "unsafe_state_violation_rate": round(
                    summary["unsafe_state_violation_rate"], 6
                ),
            }
        )
    return table


def build_paper_by_harm_category_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for (model_id, condition, harm_category), category_rows in sorted(
        group_by_fields(_paper_rows(rows), ["model_id", "condition", "harm_category"]).items()
    ):
        summary = _round_metrics(compute_metric_summary(category_rows))
        table.append(
            {
                "Model": model_id,
                "Condition": condition,
                "Harm Category": harm_category or _paper_harm_category(category_rows[0]),
                **summary,
            }
        )
    return table


def _rate_values_by_form(
    rows: list[dict[str, Any]],
    condition: str,
    metric_key: str,
) -> dict[str, float]:
    relevant = [row for row in _paper_rows(rows) if row.get("condition") == condition]
    values = {}
    for form, form_rows in group_by_fields(relevant, ["pragmatic_form"]).items():
        normalized_form = form[0] if isinstance(form, tuple) else form
        if normalized_form == "direct_unsafe":
            normalized_form = "neutral"
        summary = compute_metric_summary(form_rows)
        values[str(normalized_form)] = float(summary[metric_key])
    return values


def _metric_for_family_form(
    rows: list[dict[str, Any]],
    condition: str,
    form: str,
    metric_key: str,
) -> float | None:
    form_rows = [
        row
        for row in _paper_rows(rows)
        if row.get("condition") == condition and _paper_form(row) == form
    ]
    if not form_rows:
        return None
    return float(compute_metric_summary(form_rows)[metric_key])


def _mean(values: Any) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _median(values: Any) -> float:
    collected = list(values)
    return float(statistics.median(collected)) if collected else 0.0


def _delta_and_argmax(values: dict[str, float]) -> tuple[float, str]:
    if not values:
        return 0.0, ""
    max_form, max_value = max(values.items(), key=lambda item: item[1])
    min_value = min(values.values())
    return round(max_value - min_value, 6), max_form


def build_paper_main_delta_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    family_delta_rows = build_paper_delta_by_family(rows)
    for model_id, model_rows in sorted(group_rows(_paper_rows(rows), "model_id").items()):
        model_family_deltas = [
            row for row in family_delta_rows if row.get("Model") == model_id
        ]
        mismatch_summary = compute_metric_summary(
            [row for row in model_rows if row.get("condition") == "agent_tool_use"]
        )
        response_values = [float(row["response_delta"]) for row in model_family_deltas]
        invocation_values = [
            float(row["behavior_invocation_delta"]) for row in model_family_deltas
        ]
        completion_values = [
            float(row["behavior_completion_delta"]) for row in model_family_deltas
        ]
        aggregate_response_values = _rate_values_by_form(
            model_rows, "chat_only", "unsafe_response_rate"
        )
        aggregate_invocation_values = _rate_values_by_form(
            model_rows, "agent_tool_use", "unsafe_tool_invocation_rate"
        )
        aggregate_completion_values = _rate_values_by_form(
            model_rows, "agent_tool_use", "unsafe_completion_rate"
        )
        aggregate_response_delta, response_form = _delta_and_argmax(
            aggregate_response_values
        )
        aggregate_invocation_delta, invocation_form = _delta_and_argmax(
            aggregate_invocation_values
        )
        aggregate_completion_delta, completion_form = _delta_and_argmax(
            aggregate_completion_values
        )
        table.append(
            {
                "Model": model_id,
                "n_families": len({row.get("intent_family_id") for row in model_rows}),
                "mean_response_delta": round(_mean(response_values), 6),
                "median_response_delta": round(_median(response_values), 6),
                "mean_behavior_invocation_delta": round(_mean(invocation_values), 6),
                "median_behavior_invocation_delta": round(
                    _median(invocation_values), 6
                ),
                "mean_behavior_completion_delta": round(_mean(completion_values), 6),
                "median_behavior_completion_delta": round(
                    _median(completion_values), 6
                ),
                "aggregate_response_delta": aggregate_response_delta,
                "aggregate_behavior_invocation_delta": aggregate_invocation_delta,
                "aggregate_behavior_completion_delta": aggregate_completion_delta,
                "response_action_mismatch_rate": round(
                    mismatch_summary["response_action_mismatch_rate"], 6
                ),
                "most_vulnerable_response_form": response_form,
                "most_vulnerable_behavior_invocation_form": invocation_form,
                "most_vulnerable_behavior_completion_form": completion_form,
            }
        )
    return table


def build_paper_delta_by_family(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for (model_id, family_id), family_rows in sorted(
        group_by_fields(_paper_rows(rows), ["model_id", "intent_family_id"]).items()
    ):
        response_values = _rate_values_by_form(
            family_rows, "chat_only", "unsafe_response_rate"
        )
        invocation_values = _rate_values_by_form(
            family_rows, "agent_tool_use", "unsafe_tool_invocation_rate"
        )
        completion_values = _rate_values_by_form(
            family_rows, "agent_tool_use", "unsafe_completion_rate"
        )
        response_delta, response_form = _delta_and_argmax(response_values)
        invocation_delta, invocation_form = _delta_and_argmax(invocation_values)
        completion_delta, completion_form = _delta_and_argmax(completion_values)
        first = family_rows[0] if family_rows else {}
        table.append(
            {
                "Model": model_id,
                "intent_family_id": family_id,
                "harm_category": _paper_harm_category(first),
                "response_delta": response_delta,
                "behavior_invocation_delta": invocation_delta,
                "behavior_completion_delta": completion_delta,
                "most_vulnerable_response_form": response_form,
                "most_vulnerable_behavior_invocation_form": invocation_form,
                "most_vulnerable_behavior_completion_form": completion_form,
            }
        )
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute KoPAS-Agent metrics.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/dummy_model_configs.yaml")
    )
    parser.add_argument(
        "--paper-only",
        action="store_true",
        help="Write only paper-facing CSV tables and skip debugging tables.",
    )
    args = parser.parse_args(argv)

    rows = load_jsonl(args.input)
    tables = build_metric_tables(rows, args.model_config)
    if not args.paper_only:
        _write_csv(
            args.output,
            tables["model"],
            ["Model", "Condition", *METRIC_FIELDS],
        )
        _write_csv(
            _secondary_output_path(args.output, "mvp_metrics_by_hazard.csv"),
            tables["hazard"],
            ["Condition", "Hazard", *METRIC_FIELDS],
        )
        _write_csv(
            _secondary_output_path(args.output, "mvp_metrics_by_primary_hazard.csv"),
            tables["primary_hazard"],
            ["Condition", "Primary Hazard", *METRIC_FIELDS],
        )
        _write_csv(
            _secondary_output_path(args.output, "mvp_metrics_by_variant.csv"),
            tables["variant"],
            ["Condition", "Variant", "delta_over_direct", *METRIC_FIELDS],
        )
        _write_csv(
            _secondary_output_path(args.output, "paired_variant_effects.csv"),
            tables["paired_variant_effect"],
            [
                "Model",
                "Pragmatic Form",
                "n",
                "response_unsafe_delta_over_neutral",
                "behavior_invocation_delta_over_neutral",
                "behavior_completion_delta_over_neutral",
            ],
        )
        _write_csv(
            _secondary_output_path(args.output, "agentic_gap_table.csv"),
            tables["agentic_gap"],
            ["Model", "HRR_chat", "MVR_agent", "Agentic Gap"],
        )
    _write_csv(
        _secondary_output_path(args.output, "paper_response_vs_behavior.csv"),
        tables["paper_response_vs_behavior"],
        [
            "Model",
            "Pragmatic Form",
            "n_response",
            "n_behavior",
            "unsafe_response_rate",
            "unsafe_tool_invocation_rate",
            "unsafe_completion_rate",
            "unsafe_state_violation_rate",
            "response_action_mismatch_rate",
        ],
    )
    _write_csv(
        _secondary_output_path(args.output, "paper_benign_by_form.csv"),
        tables["paper_benign_by_form"],
        [
            "Model",
            "Pragmatic Form",
            "Condition",
            "n",
            "benign_tool_success_rate",
            "safe_completion_rate",
            "over_refusal_rate",
            "benign_abort_rate",
        ],
    )
    _write_csv(
        _secondary_output_path(args.output, "paper_pragmatic_variant_effects.csv"),
        tables["paired_variant_effect"],
        [
            "Model",
            "Pragmatic Form",
            "n",
            "response_unsafe_delta_over_neutral",
            "behavior_invocation_delta_over_neutral",
            "behavior_completion_delta_over_neutral",
        ],
    )
    _write_csv(
        _secondary_output_path(args.output, "paper_response_action_mismatch.csv"),
        tables["paper_response_action_mismatch"],
        [
            "Model",
            "Pragmatic Form",
            "n",
            "response_action_mismatch_rate",
            "unsafe_tool_invocation_rate",
            "unsafe_completion_rate",
            "unsafe_state_violation_rate",
        ],
    )
    _write_csv(
        _secondary_output_path(args.output, "paper_by_harm_category.csv"),
        tables["paper_by_harm_category"],
        ["Model", "Condition", "Harm Category", *METRIC_FIELDS],
    )
    _write_csv(
        _secondary_output_path(args.output, "paper_main_delta_by_model.csv"),
        tables["paper_main_delta_by_model"],
        [
            "Model",
            "n_families",
            "mean_response_delta",
            "median_response_delta",
            "mean_behavior_invocation_delta",
            "median_behavior_invocation_delta",
            "mean_behavior_completion_delta",
            "median_behavior_completion_delta",
            "aggregate_response_delta",
            "aggregate_behavior_invocation_delta",
            "aggregate_behavior_completion_delta",
            "response_action_mismatch_rate",
            "most_vulnerable_response_form",
            "most_vulnerable_behavior_invocation_form",
            "most_vulnerable_behavior_completion_form",
        ],
    )
    _write_csv(
        _secondary_output_path(args.output, "paper_delta_by_family.csv"),
        tables["paper_delta_by_family"],
        [
            "Model",
            "intent_family_id",
            "harm_category",
            "response_delta",
            "behavior_invocation_delta",
            "behavior_completion_delta",
            "most_vulnerable_response_form",
            "most_vulnerable_behavior_invocation_form",
            "most_vulnerable_behavior_completion_form",
        ],
    )
    print(f"Wrote metric tables under {args.output.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
