"""Pydantic v2 schemas and enumerations for Pareto Router."""

from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelTier(str, Enum):
    """Normalized hierarchy of model capabilities and costs."""
    COMPACT = "compact"       # e.g., gpt-4o-mini, haiku, gemini-flash
    MID = "mid"               # e.g., gpt-4o, sonnet-standard, gemini-pro-standard
    FRONTIER = "frontier"     # e.g., o1, o3, claude-3-7-sonnet-extended, gemini-2.5-pro-deep


class ModelMode(str, Enum):
    """Operational mode of the targeted model."""
    STANDARD = "standard"     # No extended reasoning / zero thinking tokens
    REASONING = "reasoning"   # Extended test-time compute active


class EffortLevel(str, Enum):
    """Discrete reasoning effort buckets mapped from continuous [0.0, 1.0]."""
    NONE = "none"             # 0.0 effort (thinking disabled)
    LOW = "low"               # (0.0, 0.35]
    MEDIUM = "medium"         # (0.35, 0.70]
    HIGH = "high"             # (0.70, 1.0]

    @classmethod
    def from_score(cls, score: float) -> "EffortLevel":
        """Convert continuous effort score [0.0, 1.0] to EffortLevel."""
        if score <= 0.001:
            return cls.NONE
        if score <= 0.35:
            return cls.LOW
        if score <= 0.70:
            return cls.MEDIUM
        return cls.HIGH


class ProviderType(str, Enum):
    """Supported upstream model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class EvaluationScores(BaseModel):
    """Orthogonal dimension scores computed by the evaluator."""
    model_config = ConfigDict(frozen=True)

    logic_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence/requirement for algorithmic verification, code logic, or math reasoning."
    )
    knowledge_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Entity richness, domain nuance, or depth of world knowledge required."
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Time spent evaluating in milliseconds (must be < 15ms)."
    )
    evaluator_name: str = Field(
        default="heuristic",
        description="Identifier of the evaluator used (e.g. onnx, heuristic, external_logits)."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional diagnostic metadata from the evaluator."
    )


class RoutingDecision(BaseModel):
    """Final 2D routing coordinate and associated metadata."""
    model_config = ConfigDict(frozen=True)

    tier: ModelTier
    effort_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Continuous reasoning effort in [0.0, 1.0]."
    )
    effort_level: EffortLevel
    mode: ModelMode
    reasoning_budget_tokens: int | None = Field(
        default=None,
        description="Exact token budget for thinking models where applicable."
    )
    overthinking_penalized: bool = Field(
        default=False,
        description="Whether reasoning effort was clamped to 0 to prevent degradation on simple prompts."
    )
    scores: EvaluationScores

    @field_validator("effort_score")
    @classmethod
    def validate_effort_score(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class ChatMessage(BaseModel):
    """Single turn in a conversation."""
    model_config = ConfigDict(extra="allow")

    role: str
    content: str


class InferenceRequest(BaseModel):
    """Normalized inference request accepted by the gateway."""
    model_config = ConfigDict(extra="allow")

    messages: list[ChatMessage]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    provider: ProviderType = ProviderType.OPENAI
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceResponse(BaseModel):
    """Normalized response returned by the gateway."""
    model_config = ConfigDict(extra="allow")

    id: str
    content: str
    model: str
    routing_decision: RoutingDecision
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: float = 0.0


class ProviderPayload(BaseModel):
    """Ready-to-dispatch provider API call payload."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: ProviderType
    model: str
    body: dict[str, Any]
    headers: dict[str, str] = Field(default_factory=dict)
    extra_options: dict[str, Any] = Field(default_factory=dict)
