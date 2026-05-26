from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_dataset_by_task_id(path: Path) -> dict[str, dict[str, Any]]:
    return {row["task_id"]: row for row in load_jsonl(path)}


def input_block(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_prompt": task.get("user_prompt", ""),
        "agent_goal": task.get("agent_goal", ""),
        "available_tools": task.get("available_tools", []),
        "environment_state": task.get("environment_state", {}),
        "safe_behavior": task.get("safe_behavior", []),
        "violation_conditions": task.get("violation_conditions", []),
        "rubric": task.get("rubric", {}),
        "turns": task.get("turns", []),
    }


def enrich_rows(
    rows: list[dict[str, Any]], dataset_by_task_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        copied = dict(row)
        task = dataset_by_task_id.get(str(row.get("task_id", "")), {})
        if task:
            copied.setdefault("user_prompt", task.get("user_prompt", ""))
            copied.setdefault("input", input_block(task))
        else:
            copied.setdefault("user_prompt", copied.get("input", {}).get("user_prompt", ""))
            copied.setdefault("input", copied.get("input", {}))
        enriched.append(copied)
    return enriched


def readable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted = []
    for row in rows:
        state = row.get("state_diff_score", {})
        chat = row.get("chat_response_score", {})
        rubric = row.get("rubric_score", {})
        adapter = row.get("adapter_metadata", {})
        inp = row.get("input", {})
        actions = row.get("action_log", [])
        raw_responses = adapter.get("raw_responses")
        if raw_responses is None and "raw_response" in adapter:
            raw_responses = [adapter["raw_response"]]
        formatted.append(
            {
                "model_id": row.get("model_id", ""),
                "condition": row.get("condition", ""),
                "task_id": row.get("task_id", ""),
                "domain": row.get("domain", ""),
                "variant_type": row.get("variant_type", ""),
                "user_prompt": row.get("user_prompt") or inp.get("user_prompt", ""),
                "available_tools": ", ".join(inp.get("available_tools", [])),
                "environment_state": json.dumps(
                    inp.get("environment_state", {}), ensure_ascii=False
                ),
                "safe_behavior": ", ".join(inp.get("safe_behavior", [])),
                "violation_conditions": ", ".join(inp.get("violation_conditions", [])),
                "final_response": row.get("final_response", ""),
                "tool_sequence": " > ".join(action.get("tool_name", "") for action in actions),
                "action_log": json.dumps(actions, ensure_ascii=False),
                "material_violation": state.get("material_violation", False),
                "tool_misuse": state.get("tool_misuse", False),
                "violation_types": ", ".join(state.get("violation_types", [])),
                "severity_score": state.get("severity_score", 0),
                "rubric_score": rubric.get("rubric_score", ""),
                "safe_completion": rubric.get("safe_completion", ""),
                "refusal": chat.get("refusal", ""),
                "harmful_response": chat.get("harmful_response", ""),
                "parse_error_count": len(adapter.get("parse_errors", [])),
                "raw_responses": json.dumps(raw_responses or [], ensure_ascii=False),
            }
        )
    return formatted


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = ["# KoPAS-Agent Result Review", ""]
    blocks.extend(_summary_blocks(rows))
    blocks.extend(_task_overview_blocks(rows))
    blocks.extend(_task_detail_blocks(rows))
    path.write_text("\n".join(blocks), encoding="utf-8")


def _summary_blocks(rows: list[dict[str, Any]]) -> list[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)

    lines = [
        "## Run Summary",
        "",
        "| condition | n | MVR | tool misuse | parse errors |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition, condition_rows in sorted(grouped.items()):
        n = len(condition_rows)
        mvr = _rate(row["material_violation"] for row in condition_rows)
        tmr = _rate(row["tool_misuse"] for row in condition_rows)
        parse_errors = sum(int(row["parse_error_count"]) for row in condition_rows)
        lines.append(f"| `{condition}` | {n} | {mvr:.3f} | {tmr:.3f} | {parse_errors} |")
    lines.append("")
    return lines


def _task_overview_blocks(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Task Overview",
        "",
        "| task | input | chat result | agent result | agent tools |",
        "|---|---|---|---|---|",
    ]
    for task_id, task_rows in _group_by_task(rows).items():
        by_condition = {row["condition"]: row for row in task_rows}
        first = task_rows[0]
        chat = by_condition.get("chat_only")
        agent = by_condition.get("agent_tool_use")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{task_id}`",
                    _md_cell(_clip(first["user_prompt"], 80)),
                    _md_cell(_status_cell(chat)),
                    _md_cell(_status_cell(agent)),
                    _md_cell(agent["tool_sequence"] if agent else ""),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _task_detail_blocks(rows: list[dict[str, Any]]) -> list[str]:
    blocks = ["## Task Details", ""]
    for index, (task_id, task_rows) in enumerate(_group_by_task(rows).items(), start=1):
        first = task_rows[0]
        by_condition = {row["condition"]: row for row in task_rows}
        blocks.extend(
            [
                f"### {index}. `{task_id}`",
                "",
                f"- model: `{first['model_id']}`",
                f"- domain: `{first['domain']}`",
                f"- variant: `{first['variant_type']}`",
                f"- available tools: `{first['available_tools']}`",
                f"- violation conditions: `{first['violation_conditions']}`",
                "",
                "**Input**",
                "",
                first["user_prompt"],
                "",
                "**Score Comparison**",
                "",
                (
                    "| condition | MVR | tool misuse | severity | safe completion | refusal | "
                    "harmful text | tools |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for condition in ("chat_only", "agent_tool_use"):
            row = by_condition.get(condition)
            if not row:
                continue
            blocks.append(
                "| "
                + " | ".join(
                    [
                        f"`{condition}`",
                        _bool(row["material_violation"]),
                        _bool(row["tool_misuse"]),
                        str(row["severity_score"]),
                        _bool(row["safe_completion"]),
                        _bool(row["refusal"]),
                        _bool(row["harmful_response"]),
                        _md_cell(row["tool_sequence"] or "-"),
                    ]
                )
                + " |"
            )
        blocks.append("")

        for condition in ("chat_only", "agent_tool_use"):
            row = by_condition.get(condition)
            if not row:
                continue
            blocks.extend(_condition_detail_blocks(condition, row))
    return blocks


def _condition_detail_blocks(condition: str, row: dict[str, Any]) -> list[str]:
    action_log = _json_pretty(row["action_log"])
    raw_responses = _json_pretty(row["raw_responses"])
    return [
        f"#### `{condition}` Output",
        "",
        row["final_response"] or "(empty)",
        "",
        f"#### `{condition}` Tool Calls",
        "",
        _tool_call_table(row["action_log"]),
        "",
        f"<details><summary>`{condition}` raw JSON details</summary>",
        "",
        "Action log:",
        "",
        f"```json\n{action_log}\n```",
        "",
        "Raw model responses:",
        "",
        f"```json\n{raw_responses}\n```",
        "",
        "</details>",
        "",
    ]


def _tool_call_table(action_log_json: str) -> str:
    try:
        actions = json.loads(action_log_json)
    except json.JSONDecodeError:
        actions = []
    if not actions:
        return "_No tool calls._"

    lines = [
        "| # | tool | args | result |",
        "|---:|---|---|---|",
    ]
    for index, action in enumerate(actions, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"`{action.get('tool_name', '')}`",
                    _md_cell(json.dumps(action.get("args", {}), ensure_ascii=False)),
                    _md_cell(json.dumps(action.get("result", {}), ensure_ascii=False)),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _group_by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["task_id"], []).append(row)
    return grouped


def _rate(values: Any) -> float:
    data = [bool(value) for value in values]
    return sum(data) / len(data) if data else 0.0


def _status_cell(row: dict[str, Any] | None) -> str:
    if not row:
        return "-"
    if row["material_violation"]:
        return f"MVR YES, severity {row['severity_score']}"
    if row["tool_misuse"]:
        return f"tool misuse, severity {row['severity_score']}"
    return "MVR no"


def _bool(value: Any) -> str:
    return "YES" if str(value).lower() == "true" or value is True else "no"


def _clip(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else f"{clean[: limit - 1]}..."


def _md_cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", "<br>")


def _json_pretty(text: str) -> str:
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return text


def default_output_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}{suffix}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add task input fields and readable views.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--jsonl-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)

    rows = load_jsonl(args.input)
    enriched = enrich_rows(rows, load_dataset_by_task_id(args.dataset))
    readable = readable_rows(enriched)

    jsonl_output = args.jsonl_output or args.input
    csv_output = args.csv_output or default_output_path(args.input, "_readable.csv")
    markdown_output = args.markdown_output or default_output_path(args.input, "_readable.md")

    write_jsonl(jsonl_output, enriched)
    write_csv(csv_output, readable)
    write_markdown(markdown_output, readable)

    print(f"Wrote enriched JSONL to {jsonl_output}")
    print(f"Wrote readable CSV to {csv_output}")
    print(f"Wrote readable Markdown to {markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
