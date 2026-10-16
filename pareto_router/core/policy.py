"""2D Pareto Policy Matrix with Overthinking Penalty clamping."""

from typing import Any
from pareto_router.core.schemas import (
    EffortLevel,
    EvaluationScores,
    ModelMode,
    ModelTier,
    RoutingDecision,
)


class ParetoPolicy:
    """Computes the optimal (ModelTier, ReasoningEffort) coordinate in the 2D Pareto space."""

    def __init__(
        self,
        logic_threshold_low: float = 0.15,
        logic_threshold_high: float = 0.75,
        knowledge_threshold_low: float = 0.35,
        knowledge_threshold_high: float = 0.65,
        overthinking_threshold: float = 0.25,
    ) -> None:
        self.logic_threshold_low = logic_threshold_low
        self.logic_threshold_high = logic_threshold_high
        self.knowledge_threshold_low = knowledge_threshold_low
        self.knowledge_threshold_high = knowledge_threshold_high
        self.overthinking_threshold = overthinking_threshold

    def evaluate_coordinates(
        self,
        scores: EvaluationScores
    ) -> RoutingDecision:
        """Map 2D (logic, knowledge) scores to a definitive RoutingDecision."""
        logic = scores.logic_score
        knowledge = scores.knowledge_score
        metadata = scores.metadata or {}

        # 1. Overthinking Penalty Detection
        # Triggers for low-entropy prompts (greetings, aesthetic edits, trivial translations)
        is_low_entropy = bool(
            metadata.get("is_low_entropy", False)
            or metadata.get("is_greeting", False)
            or metadata.get("is_aesthetic", False)
        )

        overthinking_penalized = False
        if is_low_entropy and logic < self.overthinking_threshold:
            effort_score = 0.0
            overthinking_penalized = True
        elif logic <= self.logic_threshold_low:
            # Low verifiability/logic requirement: clamp reasoning effort to 0.0
            effort_score = 0.0
            if is_low_entropy:
                overthinking_penalized = True
        else:
            # Scale continuous effort score in (0.0, 1.0]
            raw_effort = (logic - self.logic_threshold_low) / (1.0 - self.logic_threshold_low)
            # Nuance boost if problem has both high logic and high knowledge
            if knowledge > self.knowledge_threshold_high and logic > 0.50:
                raw_effort = min(1.0, raw_effort + 0.10)
            effort_score = round(max(0.05, min(1.0, raw_effort)), 3)

        # 2. Discrete Effort Level & Mode
        effort_level = EffortLevel.from_score(effort_score)
        mode = ModelMode.REASONING if effort_score > 0.0 else ModelMode.STANDARD

        # 3. Model Tier Selection (2D Frontier)
        # Frontier tier: deep knowledge (>0.65) OR very high logic (>0.75) OR combined high complexity
        if (
            knowledge >= self.knowledge_threshold_high
            or logic >= self.logic_threshold_high
            or (knowledge >= 0.50 and logic >= 0.50)
        ):
            tier = ModelTier.FRONTIER
        # Mid tier: moderate complexity in either dimension
        elif (
            knowledge >= self.knowledge_threshold_low
            or logic >= self.logic_threshold_low
        ):
            tier = ModelTier.MID
        # Compact tier: simple tasks, low logic, low knowledge
        else:
            tier = ModelTier.COMPACT

        # 4. Token Budget Allocation for Reasoning Mode
        reasoning_budget_tokens: int | None = None
        if mode == ModelMode.REASONING:
            reasoning_budget_tokens = int(1024 + effort_score * (32000 - 1024))

        return RoutingDecision(
            tier=tier,
            effort_score=effort_score,
            effort_level=effort_level,
            mode=mode,
            reasoning_budget_tokens=reasoning_budget_tokens,
            overthinking_penalized=overthinking_penalized,
            scores=scores,
        )
