"""Abstract Base Class and common utilities for prompt evaluators."""

import time
from abc import ABC, abstractmethod
from typing import Any
from pareto_router.core.schemas import EvaluationScores


class BaseEvaluator(ABC):
    """Abstract interface for sub-15ms dual-axis prompt evaluation."""

    @abstractmethod
    def _compute_scores(
        self,
        text: str,
        context: list[dict[str, str]] | None = None
    ) -> tuple[float, float, dict[str, Any]]:
        """Compute (logic_score, knowledge_score, metadata) synchronously.
        
        Must execute in under 15ms.
        """
        pass

    def evaluate_sync(
        self,
        text: str,
        context: list[dict[str, str]] | None = None
    ) -> EvaluationScores:
        """Synchronously evaluate text and measure precise latency."""
        t0 = time.perf_counter_ns()
        logic, knowledge, meta = self._compute_scores(text, context)
        t1 = time.perf_counter_ns()
        latency_ms = (t1 - t0) / 1_000_000.0

        return EvaluationScores(
            logic_score=max(0.0, min(1.0, float(logic))),
            knowledge_score=max(0.0, min(1.0, float(knowledge))),
            latency_ms=round(latency_ms, 3),
            evaluator_name=self.__class__.__name__,
            metadata=meta,
        )

    async def evaluate(
        self,
        text: str,
        context: list[dict[str, str]] | None = None
    ) -> EvaluationScores:
        """Async entrypoint for non-blocking evaluation pipeline."""
        return self.evaluate_sync(text, context)
