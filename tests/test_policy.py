"""Unit tests for 2D Pareto Policy and Overthinking Penalty."""

import pytest
from pareto_router.core.policy import ParetoPolicy
from pareto_router.core.schemas import (
    EffortLevel,
    EvaluationScores,
    ModelMode,
    ModelTier,
)


class TestParetoPolicy:
    def setup_method(self) -> None:
        self.policy = ParetoPolicy()

    def test_overthinking_penalty_on_greeting(self) -> None:
        scores = EvaluationScores(
            logic_score=0.05,
            knowledge_score=0.10,
            latency_ms=0.3,
            metadata={"is_low_entropy": True, "is_greeting": True}
        )
        decision = self.policy.evaluate_coordinates(scores)

        assert decision.tier == ModelTier.COMPACT
        assert decision.effort_score == 0.0
        assert decision.effort_level == EffortLevel.NONE
        assert decision.mode == ModelMode.STANDARD
        assert decision.overthinking_penalized is True
        assert decision.reasoning_budget_tokens is None

    def test_overthinking_penalty_on_aesthetic_edit(self) -> None:
        scores = EvaluationScores(
            logic_score=0.10,
            knowledge_score=0.25,
            latency_ms=0.4,
            metadata={"is_low_entropy": True, "is_aesthetic": True}
        )
        decision = self.policy.evaluate_coordinates(scores)

        assert decision.effort_score == 0.0
        assert decision.mode == ModelMode.STANDARD
        assert decision.overthinking_penalized is True

    def test_deep_knowledge_standard_frontier_no_overthinking(self) -> None:
        # A deep essay on philosophy/history requires high knowledge, but minimal algorithmic logic
        scores = EvaluationScores(
            logic_score=0.12,
            knowledge_score=0.85,
            latency_ms=0.5,
            metadata={}
        )
        decision = self.policy.evaluate_coordinates(scores)

        # Should be Frontier model (e.g. Claude Sonnet / GPT-4o), but ZERO reasoning effort!
        assert decision.tier == ModelTier.FRONTIER
        assert decision.effort_score == 0.0
        assert decision.effort_level == EffortLevel.NONE
        assert decision.mode == ModelMode.STANDARD
        assert decision.reasoning_budget_tokens is None

    def test_high_logic_algorithmic_prompt(self) -> None:
        # Complex dynamic programming or regex verification
        scores = EvaluationScores(
            logic_score=0.90,
            knowledge_score=0.20,
            latency_ms=0.4,
            metadata={}
        )
        decision = self.policy.evaluate_coordinates(scores)

        assert decision.tier == ModelTier.FRONTIER
        assert decision.effort_score >= 0.70
        assert decision.effort_level == EffortLevel.HIGH
        assert decision.mode == ModelMode.REASONING
        assert decision.reasoning_budget_tokens is not None
        assert decision.reasoning_budget_tokens > 20000

    def test_dual_high_frontier_problem(self) -> None:
        # Distributed consensus protocol formal proof (high logic + high knowledge)
        scores = EvaluationScores(
            logic_score=0.85,
            knowledge_score=0.80,
            latency_ms=0.6,
            metadata={}
        )
        decision = self.policy.evaluate_coordinates(scores)

        assert decision.tier == ModelTier.FRONTIER
        assert decision.effort_score >= 0.80
        assert decision.effort_level == EffortLevel.HIGH
        assert decision.mode == ModelMode.REASONING

    def test_mid_tier_moderate_problem(self) -> None:
        scores = EvaluationScores(
            logic_score=0.45,
            knowledge_score=0.40,
            latency_ms=0.4,
            metadata={}
        )
        decision = self.policy.evaluate_coordinates(scores)

        assert decision.tier == ModelTier.MID
        assert decision.effort_level in (EffortLevel.LOW, EffortLevel.MEDIUM)
        assert decision.mode == ModelMode.REASONING
