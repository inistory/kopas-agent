from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReleaseLevel(str, Enum):
    public = "public"
    controlled = "controlled"
    private = "private"
    redacted = "redacted"


class Split(str, Enum):
    public_practice = "public_practice"
    public_dev = "public_dev"
    private_test = "private_test"
    human_audit = "human_audit"


class Domain(str, Enum):
    mock_workspace = "mock_workspace"
    mock_public_service = "mock_public_service"
    mock_security_ops = "mock_security_ops"


class VariantType(str, Enum):
    direct_unsafe = "direct_unsafe"
    honorific_polite = "honorific_polite"
    indirect_request = "indirect_request"
    authority_claim = "authority_claim"
    urgency_pressure = "urgency_pressure"
    responsibility_transfer = "responsibility_transfer"
    institutional_framing = "institutional_framing"
    research_or_test_framing = "research_or_test_framing"
    ko_en_code_switching = "ko_en_code_switching"
    multi_turn_escalation = "multi_turn_escalation"
    benign_counterpart = "benign_counterpart"


class EvalCondition(str, Enum):
    chat_only = "chat_only"
    agent_tool_use = "agent_tool_use"


class ReviewStatus(str, Enum):
    auto_draft = "auto_draft"
    human_reviewed = "human_reviewed"
    excluded = "excluded"


class Rubric(BaseModel):
    material_violation: str
    safe_completion: str
    over_refusal: str


class ConversationTurn(BaseModel):
    role: str
    content: str
    expected_safe_action: str | None = None


class UnsafeToolCondition(BaseModel):
    stage: str
    violation_type: str
    tool: str | None = None
    state_key: str | None = None
    arguments_must_include: dict[str, Any] = Field(default_factory=dict)
    must_include: dict[str, Any] | str | list[str] | None = None

    @field_validator("stage")
    @classmethod
    def validate_stage(cls, value: str) -> str:
        if value not in {"tool_invocation", "environment_state"}:
            msg = f"Unsupported unsafe condition stage: {value}"
            raise ValueError(msg)
        return value


class SafeToolCondition(BaseModel):
    tool: str
    success_type: str
    arguments_must_include: dict[str, Any] = Field(default_factory=dict)
    state_key: str | None = None
    must_include: dict[str, Any] | str | list[str] | None = None


class KopasTask(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    task_id: str = Field(min_length=1)
    split: Split
    release_level: ReleaseLevel
    hazard_category: list[str] = Field(min_length=1)
    domain: Domain
    language: str = "ko"
    variant_type: VariantType
    user_prompt: str = Field(min_length=1)
    agent_goal: str = Field(min_length=1)
    available_tools: list[str] = Field(min_length=1)
    environment_state: dict[str, Any] = Field(default_factory=dict)
    safe_behavior: list[str] = Field(min_length=1)
    violation_conditions: list[str] = Field(min_length=1)
    primary_hazard_category: str | None = None
    base_task_id: str | None = None
    intent_id: str | None = None
    intent_family_id: str | None = None
    harm_category: str | None = None
    base_intent: str | None = None
    pragmatic_form: str | None = None
    is_benign: bool = False
    source_dataset: str | None = None
    source_split: str | None = None
    source_item_id: str | None = None
    source_category: str | None = None
    source_name: str | None = None
    source_target_functions: list[str] = Field(default_factory=list)
    source_grading_function: str | None = None
    source_intent_summary_ko: str | None = None
    abstract_intent: str | None = None
    localized_neutral_prompt_ko: str | None = None
    localized_benign_prompt_ko: str | None = None
    base_prompt_ko: str | None = None
    surface_prompt_ko: str | None = None
    construction_method: str | None = None
    rewrite_method: str | None = None
    rewrite_model: str | None = None
    rewrite_prompt_version: str | None = None
    translation_provider: str | None = None
    translation_model: str | None = None
    translation_prompt_version: str | None = None
    translation_method: str | None = None
    review_status: ReviewStatus = ReviewStatus.auto_draft
    dataset_version: str | None = None
    review_notes: str | None = None
    unsafe_tool_conditions: list[UnsafeToolCondition] = Field(default_factory=list)
    safe_tool_conditions: list[SafeToolCondition] = Field(default_factory=list)
    turns: list[ConversationTurn] = Field(default_factory=list)
    benign_counterpart_id: str | None = None
    rubric: Rubric

    @field_validator("hazard_category")
    @classmethod
    def validate_hazard_names(cls, values: list[str]) -> list[str]:
        for value in values:
            if not value.startswith(("P1_", "P2_", "P3_", "P7_")):
                msg = f"Unsupported MVP hazard category: {value}"
                raise ValueError(msg)
        return values

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        if value not in {"ko", "ko-en", "en"}:
            msg = f"Unsupported language: {value}"
            raise ValueError(msg)
        return value


class ActionRecord(BaseModel):
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None


class RubricScore(BaseModel):
    rubric_score: int = Field(ge=0, le=4)
    safe_completion: bool
    over_refusal: bool
    safety_grounded: bool
    rationale: str


class ChatResponseScore(BaseModel):
    harmful_response: bool
    refusal: bool
    safe_explanation: bool
    over_refusal: bool
    rationale: str


class TaskFamily(BaseModel):
    base_task_id: str
    task_ids: list[str] = Field(default_factory=list)
