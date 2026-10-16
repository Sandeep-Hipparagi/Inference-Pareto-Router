"""2D Pareto Policy Matrix with Overthinking Penalty clamping and strategy presets."""

from typing import Any
from pareto_router.core.schemas import (
    EffortLevel,
    EvaluationScores,
    ModelMode,
    ModelTier,
    RoutingDecision,
    RoutingStrategy,
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
        default_strategy: RoutingStrategy = RoutingStrategy.PARETO_BALANCED,
    ) -> None:
        self.logic_threshold_low = logic_threshold_low
        self.logic_threshold_high = logic_threshold_high
        self.knowledge_threshold_low = knowledge_threshold_low
        self.knowledge_threshold_high = knowledge_threshold_high
        self.overthinking_threshold = overthinking_threshold
        self.default_strategy = default_strategy

    def evaluate_coordinates(
        self,
        scores: EvaluationScores,
        strategy: RoutingStrategy | str | None = None,
    ) -> RoutingDecision:
        """Map 2D (logic, knowledge) scores to a definitive RoutingDecision."""
        strat = RoutingStrategy(strategy) if strategy else self.default_strategy

        # Adjust thresholds based on strategy preset
        if strat == RoutingStrategy.PARETO_COST:
            t_logic_low = 0.25
            t_logic_high = 0.85
            t_know_low = 0.45
            t_know_high = 0.75
            t_overthinking = 0.35
        elif strat == RoutingStrategy.PARETO_QUALITY:
            t_logic_low = 0.10
            t_logic_high = 0.60
            t_know_low = 0.25
            t_know_high = 0.50
            t_overthinking = 0.15
        else:  # PARETO_BALANCED
            t_logic_low = self.logic_threshold_low
            t_logic_high = self.logic_threshold_high
            t_know_low = self.knowledge_threshold_low
            t_know_high = self.knowledge_threshold_high
            t_overthinking = self.overthinking_threshold

        logic = scores.logic_score
        knowledge = scores.knowledge_score
        metadata = scores.metadata or {}

        # 1. Overthinking Penalty Detection
        is_low_entropy = bool(
            metadata.get("is_low_entropy", False)
            or metadata.get("is_greeting", False)
            or metadata.get("is_aesthetic", False)
        )

        overthinking_penalized = False
        if is_low_entropy and logic < t_overthinking:
            effort_score = 0.0
            overthinking_penalized = True
        elif logic <= t_logic_low:
            # Low verifiability/logic requirement: clamp reasoning effort to 0.0
            effort_score = 0.0
            if is_low_entropy:
                overthinking_penalized = True
        else:
            # Scale continuous effort score in (0.0, 1.0]
            raw_effort = (logic - t_logic_low) / (1.0 - t_logic_low)
            # Boost if problem has both high logic and high knowledge
            if knowledge > t_know_high and logic > 0.50:
                raw_effort = min(1.0, raw_effort + 0.10)
            if strat == RoutingStrategy.PARETO_QUALITY:
                raw_effort = min(1.0, raw_effort + 0.10)
            elif strat == RoutingStrategy.PARETO_COST:
                raw_effort = max(0.05, raw_effort - 0.10)
            effort_score = round(max(0.05, min(1.0, raw_effort)), 3)

        # 2. Discrete Effort Level & Mode
        effort_level = EffortLevel.from_score(effort_score)
        mode = ModelMode.REASONING if effort_score > 0.0 else ModelMode.STANDARD

        # 3. Model Tier Selection (2D Frontier)
        if (
            knowledge >= t_know_high
            or logic >= t_logic_high
            or (knowledge >= 0.50 and logic >= 0.50)
        ):
            tier = ModelTier.FRONTIER
        elif (
            knowledge >= t_know_low
            or logic >= t_logic_low
        ):
            tier = ModelTier.MID
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
