"""Speculative escalation cascade engine with programmatic verification."""

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Coroutine
from pareto_router.core.normalizer import ProviderNormalizer
from pareto_router.core.schemas import (
    EffortLevel,
    EvaluationScores,
    InferenceResponse,
    ModelMode,
    ModelTier,
    ProviderType,
    RoutingDecision,
)
from pareto_router.gateway.verifiers import BaseVerifier


@dataclass
class CascadeAttempt:
    """Record of an execution attempt in the cascade."""
    attempt_number: int
    tier: ModelTier
    effort_score: float
    model: str
    status: str  # "success", "validation_failed", "provider_error"
    error_message: str | None = None
    latency_ms: float = 0.0


@dataclass
class CascadeResult:
    """Outcome of a speculative cascade run."""
    response: InferenceResponse
    attempts: list[CascadeAttempt] = field(default_factory=list)
    escalated: bool = False
    final_tier: ModelTier = ModelTier.COMPACT
    total_latency_ms: float = 0.0


# Type for inference execution callable: (RoutingDecision, ProviderPayload) -> InferenceResponse
InferenceExecutor = Callable[[RoutingDecision, dict[str, Any]], Coroutine[Any, Any, InferenceResponse]]


class SpeculativeCascade:
    """Executes requests with fast speculative models, escalating on verification failures."""

    def __init__(
        self,
        normalizer: ProviderNormalizer | None = None,
        escalation_tier: ModelTier = ModelTier.FRONTIER,
        escalation_effort: float = 0.85,
    ) -> None:
        self.normalizer = normalizer or ProviderNormalizer()
        self.escalation_tier = escalation_tier
        self.escalation_effort = escalation_effort

    def build_escalated_decision(
        self,
        original_decision: RoutingDecision,
        error_context: str
    ) -> RoutingDecision:
        """Construct escalated (Frontier, High Effort) decision."""
        meta = dict(original_decision.scores.metadata)
        meta["escalated_from_tier"] = original_decision.tier.value
        meta["escalation_reason"] = error_context

        escalated_scores = EvaluationScores(
            logic_score=max(original_decision.scores.logic_score, 0.90),
            knowledge_score=max(original_decision.scores.knowledge_score, 0.70),
            latency_ms=original_decision.scores.latency_ms,
            evaluator_name=original_decision.scores.evaluator_name,
            metadata=meta,
        )

        return RoutingDecision(
            tier=self.escalation_tier,
            effort_score=self.escalation_effort,
            effort_level=EffortLevel.from_score(self.escalation_effort),
            mode=ModelMode.REASONING,
            reasoning_budget_tokens=int(1024 + self.escalation_effort * (32000 - 1024)),
            overthinking_penalized=False,
            scores=escalated_scores,
        )

    async def execute(
        self,
        initial_decision: RoutingDecision,
        provider: ProviderType,
        messages: list[dict[str, Any]],
        executor: InferenceExecutor,
        verifiers: list[BaseVerifier] | None = None,
        base_body: dict[str, Any] | None = None,
    ) -> CascadeResult:
        """Run speculative execution with automatic escalation on failure."""
        t0 = time.perf_counter_ns()
        attempts: list[CascadeAttempt] = []
        active_decision = initial_decision

        # Attempt 1: Optimal routed decision
        payload1 = self.normalizer.to_provider_payload(
            active_decision, provider, messages, base_body
        )

        t_attempt0 = time.perf_counter_ns()
        try:
            resp = await executor(active_decision, payload1.body)
            t_attempt1 = time.perf_counter_ns()
            attempt_latency = (t_attempt1 - t_attempt0) / 1_000_000.0

            # Run programmatic verifiers
            validation_error: str | None = None
            if verifiers:
                for v in verifiers:
                    valid, err = v.verify(resp.content)
                    if not valid:
                        validation_error = err
                        break

            if validation_error is None:
                attempts.append(CascadeAttempt(
                    attempt_number=1,
                    tier=active_decision.tier,
                    effort_score=active_decision.effort_score,
                    model=payload1.model,
                    status="success",
                    latency_ms=round(attempt_latency, 2),
                ))
                t1 = time.perf_counter_ns()
                total_latency = (t1 - t0) / 1_000_000.0
                return CascadeResult(
                    response=resp,
                    attempts=attempts,
                    escalated=False,
                    final_tier=active_decision.tier,
                    total_latency_ms=round(total_latency, 2),
                )
            else:
                attempts.append(CascadeAttempt(
                    attempt_number=1,
                    tier=active_decision.tier,
                    effort_score=active_decision.effort_score,
                    model=payload1.model,
                    status="validation_failed",
                    error_message=validation_error,
                    latency_ms=round(attempt_latency, 2),
                ))
                escalation_reason = f"Verification failed: {validation_error}"

        except Exception as e:
            t_attempt1 = time.perf_counter_ns()
            attempt_latency = (t_attempt1 - t_attempt0) / 1_000_000.0
            attempts.append(CascadeAttempt(
                attempt_number=1,
                tier=active_decision.tier,
                effort_score=active_decision.effort_score,
                model=payload1.model,
                status="provider_error",
                error_message=str(e),
                latency_ms=round(attempt_latency, 2),
            ))
            escalation_reason = f"Provider call failed: {e}"

        # Attempt 2: Speculative Escalation to (Frontier, High Effort)
        escalated_decision = self.build_escalated_decision(active_decision, escalation_reason)
        payload2 = self.normalizer.to_provider_payload(
            escalated_decision, provider, messages, base_body
        )

        t_esc0 = time.perf_counter_ns()
        resp2 = await executor(escalated_decision, payload2.body)
        t_esc1 = time.perf_counter_ns()
        esc_latency = (t_esc1 - t_esc0) / 1_000_000.0

        attempts.append(CascadeAttempt(
            attempt_number=2,
            tier=escalated_decision.tier,
            effort_score=escalated_decision.effort_score,
            model=payload2.model,
            status="success",
            latency_ms=round(esc_latency, 2),
        ))

        t1 = time.perf_counter_ns()
        total_latency = (t1 - t0) / 1_000_000.0

        return CascadeResult(
            response=resp2,
            attempts=attempts,
            escalated=True,
            final_tier=escalated_decision.tier,
            total_latency_ms=round(total_latency, 2),
        )
