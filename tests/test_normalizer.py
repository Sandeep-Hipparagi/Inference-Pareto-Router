"""Unit tests for bidirectional provider normalization."""

import pytest
from pareto_router.core.normalizer import ProviderNormalizer
from pareto_router.core.schemas import (
    EffortLevel,
    EvaluationScores,
    ModelMode,
    ModelTier,
    ProviderType,
    RoutingDecision,
)


def make_decision(
    tier: ModelTier,
    effort: float,
    mode: ModelMode | None = None
) -> RoutingDecision:
    actual_mode = mode or (ModelMode.REASONING if effort > 0.0 else ModelMode.STANDARD)
    return RoutingDecision(
        tier=tier,
        effort_score=effort,
        effort_level=EffortLevel.from_score(effort),
        mode=actual_mode,
        reasoning_budget_tokens=int(1024 + effort * 30000) if effort > 0 else None,
        overthinking_penalized=False,
        scores=EvaluationScores(
            logic_score=effort,
            knowledge_score=0.5,
            latency_ms=2.1,
            evaluator_name="test"
        )
    )


class TestOpenAINormalization:
    def setup_method(self) -> None:
        self.normalizer = ProviderNormalizer()

    def test_compact_standard(self) -> None:
        decision = make_decision(ModelTier.COMPACT, 0.0)
        messages = [{"role": "user", "content": "Hello"}]
        payload = self.normalizer.to_provider_payload(
            decision, ProviderType.OPENAI, messages, {"temperature": 0.7, "max_tokens": 100}
        )

        assert payload.model == "gpt-4o-mini"
        assert "reasoning_effort" not in payload.body
        assert payload.body["temperature"] == 0.7
        assert payload.body["max_tokens"] == 100

    def test_reasoning_effort_levels(self) -> None:
        # Low effort
        d_low = make_decision(ModelTier.COMPACT, 0.2)
        p_low = self.normalizer.to_provider_payload(
            d_low, ProviderType.OPENAI, [{"role": "user", "content": "solve"}], {"temperature": 0.7, "max_tokens": 200}
        )
        assert p_low.model == "o3-mini"
        assert p_low.body.get("reasoning_effort") == "low"
        assert "temperature" not in p_low.body  # stripped for reasoning model
        assert p_low.body.get("max_completion_tokens") == 200

        # Medium effort
        d_med = make_decision(ModelTier.MID, 0.5)
        p_med = self.normalizer.to_provider_payload(
            d_med, ProviderType.OPENAI, [{"role": "user", "content": "solve"}]
        )
        assert p_med.body.get("reasoning_effort") == "medium"

        # High effort
        d_high = make_decision(ModelTier.FRONTIER, 0.85)
        p_high = self.normalizer.to_provider_payload(
            d_high, ProviderType.OPENAI, [{"role": "user", "content": "solve"}]
        )
        assert p_high.model == "o1"
        assert p_high.body.get("reasoning_effort") == "high"


class TestAnthropicNormalization:
    def setup_method(self) -> None:
        self.normalizer = ProviderNormalizer()

    def test_compact_standard_no_thinking(self) -> None:
        decision = make_decision(ModelTier.COMPACT, 0.0)
        payload = self.normalizer.to_provider_payload(
            decision, ProviderType.ANTHROPIC, [{"role": "user", "content": "hi"}]
        )
        assert payload.model == "claude-3-5-haiku-20241022"
        assert "thinking" not in payload.body

    def test_reasoning_with_thinking_budget(self) -> None:
        decision = make_decision(ModelTier.FRONTIER, 0.5)
        payload = self.normalizer.to_provider_payload(
            decision,
            ProviderType.ANTHROPIC,
            [{"role": "user", "content": "prove"}],
            {"max_tokens": 2000}
        )
        assert payload.model == "claude-3-7-sonnet-20250219"
        assert "thinking" in payload.body
        assert payload.body["thinking"]["type"] == "enabled"
        budget = payload.body["thinking"]["budget_tokens"]
        assert budget >= 1024
        # temperature must be 1.0 for thinking
        assert payload.body["temperature"] == 1.0
        # max_tokens must be strictly greater than budget
        assert payload.body["max_tokens"] > budget

    def test_boundary_effort_values(self) -> None:
        # Min boundary
        budget_min = self.normalizer.map_effort_to_anthropic_tokens(0.01)
        assert budget_min >= 1024

        # Max boundary
        budget_max = self.normalizer.map_effort_to_anthropic_tokens(1.0)
        assert budget_max == 32000


class TestGeminiNormalization:
    def setup_method(self) -> None:
        self.normalizer = ProviderNormalizer()

    def test_gemini_standard(self) -> None:
        decision = make_decision(ModelTier.COMPACT, 0.0)
        payload = self.normalizer.to_provider_payload(
            decision, ProviderType.GEMINI, [{"role": "user", "content": "summarize"}]
        )
        assert payload.model == "gemini-2.5-flash"
        assert payload.body["thinking_config"]["thinking_budget"] == 0

    def test_gemini_reasoning(self) -> None:
        decision = make_decision(ModelTier.FRONTIER, 0.8)
        payload = self.normalizer.to_provider_payload(
            decision, ProviderType.GEMINI, [{"role": "user", "content": "complex math"}]
        )
        assert payload.model == "gemini-2.5-pro"
        assert payload.body["thinking_config"]["thinking_budget"] >= 1024


class TestBidirectionalDenormalization:
    def setup_method(self) -> None:
        self.normalizer = ProviderNormalizer()

    def test_denormalize_openai(self) -> None:
        body = {
            "model": "o3-mini",
            "reasoning_effort": "high"
        }
        tier, effort_score, effort_level, mode = self.normalizer.denormalize_from_payload(
            body, ProviderType.OPENAI
        )
        assert tier == ModelTier.COMPACT
        assert effort_level == EffortLevel.HIGH
        assert mode == ModelMode.REASONING
        assert effort_score >= 0.70

    def test_denormalize_anthropic(self) -> None:
        body = {
            "model": "claude-3-7-sonnet-20250219",
            "thinking": {"type": "enabled", "budget_tokens": 16512}
        }
        tier, effort_score, effort_level, mode = self.normalizer.denormalize_from_payload(
            body, ProviderType.ANTHROPIC
        )
        assert tier == ModelTier.FRONTIER
        assert mode == ModelMode.REASONING
        assert 0.45 <= effort_score <= 0.55
        assert effort_level == EffortLevel.MEDIUM

    def test_denormalize_gemini(self) -> None:
        body = {
            "model": "gemini-2.5-flash",
            "thinking_config": {"thinking_budget": 0}
        }
        tier, effort_score, effort_level, mode = self.normalizer.denormalize_from_payload(
            body, ProviderType.GEMINI
        )
        assert tier == ModelTier.COMPACT
        assert mode == ModelMode.STANDARD
        assert effort_score == 0.0
        assert effort_level == EffortLevel.NONE
