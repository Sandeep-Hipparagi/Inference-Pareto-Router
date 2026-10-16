"""Bidirectional mapping between normalized Pareto coordinates and provider payloads."""

from dataclasses import dataclass
from typing import Any
from pareto_router.core.schemas import (
    EffortLevel,
    ModelMode,
    ModelTier,
    ProviderPayload,
    ProviderType,
    RoutingDecision,
)


@dataclass(frozen=True)
class ProviderModelCatalog:
    """Catalog of model identifiers categorized by tier and operational mode."""
    # OpenAI
    openai_compact_standard: str = "gpt-4o-mini"
    openai_compact_reasoning: str = "o3-mini"
    openai_mid_standard: str = "gpt-4o"
    openai_mid_reasoning: str = "o3-mini"
    openai_frontier_standard: str = "gpt-4o"
    openai_frontier_reasoning: str = "o1"

    # Anthropic
    anthropic_compact_standard: str = "claude-3-5-haiku-20241022"
    anthropic_compact_reasoning: str = "claude-3-7-sonnet-20250219"
    anthropic_mid_standard: str = "claude-3-5-sonnet-20241022"
    anthropic_mid_reasoning: str = "claude-3-7-sonnet-20250219"
    anthropic_frontier_standard: str = "claude-3-5-sonnet-20241022"
    anthropic_frontier_reasoning: str = "claude-3-7-sonnet-20250219"

    # Gemini
    gemini_compact_standard: str = "gemini-2.5-flash"
    gemini_compact_reasoning: str = "gemini-2.5-flash"
    gemini_mid_standard: str = "gemini-2.5-flash"
    gemini_mid_reasoning: str = "gemini-2.5-pro"
    gemini_frontier_standard: str = "gemini-2.5-pro"
    gemini_frontier_reasoning: str = "gemini-2.5-pro"


DEFAULT_CATALOG = ProviderModelCatalog()


class ProviderNormalizer:
    """Translates between 2D Pareto coordinates (tier, effort) and vendor-specific payloads."""

    def __init__(self, catalog: ProviderModelCatalog = DEFAULT_CATALOG) -> None:
        self.catalog = catalog

    # --- Resolution Helpers ---

    def resolve_model_name(
        self,
        provider: ProviderType,
        tier: ModelTier,
        mode: ModelMode
    ) -> str:
        """Resolve concrete vendor model string for a given tier and mode."""
        if provider == ProviderType.OPENAI:
            if tier == ModelTier.COMPACT:
                return (
                    self.catalog.openai_compact_reasoning
                    if mode == ModelMode.REASONING
                    else self.catalog.openai_compact_standard
                )
            elif tier == ModelTier.MID:
                return (
                    self.catalog.openai_mid_reasoning
                    if mode == ModelMode.REASONING
                    else self.catalog.openai_mid_standard
                )
            else:
                return (
                    self.catalog.openai_frontier_reasoning
                    if mode == ModelMode.REASONING
                    else self.catalog.openai_frontier_standard
                )

        elif provider == ProviderType.ANTHROPIC:
            if tier == ModelTier.COMPACT:
                return (
                    self.catalog.anthropic_compact_reasoning
                    if mode == ModelMode.REASONING
                    else self.catalog.anthropic_compact_standard
                )
            elif tier == ModelTier.MID:
                return (
                    self.catalog.anthropic_mid_reasoning
                    if mode == ModelMode.REASONING
                    else self.catalog.anthropic_mid_standard
                )
            else:
                return (
                    self.catalog.anthropic_frontier_reasoning
                    if mode == ModelMode.REASONING
                    else self.catalog.anthropic_frontier_standard
                )

        elif provider == ProviderType.GEMINI:
            if tier == ModelTier.COMPACT:
                return (
                    self.catalog.gemini_compact_reasoning
                    if mode == ModelMode.REASONING
                    else self.catalog.gemini_compact_standard
                )
            elif tier == ModelTier.MID:
                return (
                    self.catalog.gemini_mid_reasoning
                    if mode == ModelMode.REASONING
                    else self.catalog.gemini_mid_standard
                )
            else:
                return (
                    self.catalog.gemini_frontier_reasoning
                    if mode == ModelMode.REASONING
                    else self.catalog.gemini_frontier_standard
                )

        raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def map_effort_to_openai_level(effort_score: float) -> str:
        """Map effort score in [0.0, 1.0] to OpenAI reasoning_effort string."""
        if effort_score <= 0.35:
            return "low"
        elif effort_score <= 0.70:
            return "medium"
        else:
            return "high"

    @staticmethod
    def map_effort_to_anthropic_tokens(
        effort_score: float,
        min_budget: int = 1024,
        max_budget: int = 32000
    ) -> int:
        """Map effort score in (0.0, 1.0] to integer token budget for Anthropic."""
        clamped = max(0.01, min(1.0, effort_score))
        budget = int(min_budget + clamped * (max_budget - min_budget))
        return max(min_budget, budget)

    @staticmethod
    def map_effort_to_gemini_tokens(
        effort_score: float,
        min_budget: int = 1024,
        max_budget: int = 24576
    ) -> int:
        """Map effort score in (0.0, 1.0] to integer thinking budget for Gemini."""
        clamped = max(0.01, min(1.0, effort_score))
        budget = int(min_budget + clamped * (max_budget - min_budget))
        return max(min_budget, budget)

    # --- Forward Mapping ---

    def to_provider_payload(
        self,
        decision: RoutingDecision,
        provider: ProviderType,
        messages: list[dict[str, Any]],
        base_body: dict[str, Any] | None = None
    ) -> ProviderPayload:
        """Construct exact provider-specific payload according to RoutingDecision."""
        body = dict(base_body) if base_body else {}
        body["messages"] = messages

        model_name = self.resolve_model_name(provider, decision.tier, decision.mode)
        body["model"] = model_name

        if provider == ProviderType.OPENAI:
            if decision.mode == ModelMode.REASONING:
                effort_str = self.map_effort_to_openai_level(decision.effort_score)
                body["reasoning_effort"] = effort_str
                # Reasoning models o1/o3-mini require temperature to be omitted or 1
                if "temperature" in body:
                    body.pop("temperature", None)
                # Remap max_tokens to max_completion_tokens if present
                if "max_tokens" in body:
                    body["max_completion_tokens"] = body.pop("max_tokens")
            else:
                body.pop("reasoning_effort", None)

        elif provider == ProviderType.ANTHROPIC:
            if decision.mode == ModelMode.REASONING and decision.effort_score > 0.0:
                budget = self.map_effort_to_anthropic_tokens(decision.effort_score)
                body["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget,
                }
                # Anthropic API constraint: thinking requires temperature=1.0
                body["temperature"] = 1.0
                # max_tokens must be strictly greater than budget_tokens
                current_max = body.get("max_tokens", 0)
                body["max_tokens"] = max(current_max, budget + 4096)
            else:
                # Standard mode: thinking disabled or omitted
                body.pop("thinking", None)

        elif provider == ProviderType.GEMINI:
            if decision.mode == ModelMode.REASONING and decision.effort_score > 0.0:
                gemini_budget = self.map_effort_to_gemini_tokens(decision.effort_score)
                body["thinking_config"] = {
                    "thinking_budget": gemini_budget
                }
            else:
                body["thinking_config"] = {
                    "thinking_budget": 0
                }

        return ProviderPayload(
            provider=provider,
            model=model_name,
            body=body,
        )

    # --- Backward Mapping (Denormalization) ---

    def denormalize_from_payload(
        self,
        payload_body: dict[str, Any],
        provider: ProviderType
    ) -> tuple[ModelTier, float, EffortLevel, ModelMode]:
        """Infer (ModelTier, effort_score, EffortLevel, ModelMode) from raw payload body."""
        model = str(payload_body.get("model", "")).lower()

        if provider == ProviderType.OPENAI:
            reasoning_effort = payload_body.get("reasoning_effort")
            is_reasoning_model = any(k in model for k in ["o1", "o3", "o4"])

            if reasoning_effort or is_reasoning_model:
                mode = ModelMode.REASONING
                effort_str = str(reasoning_effort).lower() if reasoning_effort else "medium"
                if effort_str == "low":
                    effort_score = 0.25
                elif effort_str == "medium":
                    effort_score = 0.50
                elif effort_str == "high":
                    effort_score = 0.85
                else:
                    effort_score = 0.50
            else:
                mode = ModelMode.STANDARD
                effort_score = 0.0

            # Determine tier
            if any(k in model for k in ["mini"]):
                tier = ModelTier.COMPACT
            elif any(k in model for k in ["o1", "o3", "chatgpt-4o-latest"]):
                tier = ModelTier.FRONTIER
            else:
                tier = ModelTier.MID

        elif provider == ProviderType.ANTHROPIC:
            thinking = payload_body.get("thinking")
            if isinstance(thinking, dict) and thinking.get("type") == "enabled":
                mode = ModelMode.REASONING
                budget = thinking.get("budget_tokens", 1024)
                # Reverse normalize budget: min 1024, max 32000
                clamped_budget = max(1024, min(32000, budget))
                effort_score = round((clamped_budget - 1024) / (32000 - 1024), 2)
                effort_score = max(0.05, effort_score)
            else:
                mode = ModelMode.STANDARD
                effort_score = 0.0

            # Determine tier
            if "haiku" in model:
                tier = ModelTier.COMPACT
            elif "opus" in model or "claude-3-7" in model:
                tier = ModelTier.FRONTIER if mode == ModelMode.REASONING else ModelTier.MID
            else:
                tier = ModelTier.MID

        elif provider == ProviderType.GEMINI:
            thinking_config = payload_body.get("thinking_config")
            budget = 0
            if isinstance(thinking_config, dict):
                budget = thinking_config.get("thinking_budget", 0)

            if budget > 0:
                mode = ModelMode.REASONING
                clamped_budget = max(1024, min(24576, budget))
                effort_score = round((clamped_budget - 1024) / (24576 - 1024), 2)
                effort_score = max(0.05, effort_score)
            else:
                mode = ModelMode.STANDARD
                effort_score = 0.0

            # Determine tier
            if "flash" in model:
                tier = ModelTier.COMPACT
            elif "pro" in model:
                tier = ModelTier.FRONTIER if mode == ModelMode.REASONING else ModelTier.MID
            else:
                tier = ModelTier.MID

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        effort_level = EffortLevel.from_score(effort_score)
        return tier, effort_score, effort_level, mode
