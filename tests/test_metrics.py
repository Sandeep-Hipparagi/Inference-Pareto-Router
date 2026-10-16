"""Unit tests for Prometheus and OpenTelemetry-ready metrics exporter."""

import pytest
from pareto_router.core.schemas import (
    EffortLevel,
    EvaluationScores,
    ModelMode,
    ModelTier,
    ProviderType,
    RoutingDecision,
    RoutingStrategy,
)
from pareto_router.gateway.client import ParetoRouter
from pareto_router.telemetry.metrics import PrometheusMetricsExporter


class TestPrometheusMetricsExporter:
    def setup_method(self) -> None:
        self.exporter = PrometheusMetricsExporter()

    def test_record_and_prometheus_text_export(self) -> None:
        scores = EvaluationScores(
            logic_score=0.1,
            knowledge_score=0.2,
            latency_ms=0.45,
            evaluator_name="HeuristicEvaluator",
        )
        decision = RoutingDecision(
            tier=ModelTier.COMPACT,
            effort_score=0.0,
            effort_level=EffortLevel.NONE,
            mode=ModelMode.STANDARD,
            overthinking_penalized=True,
            scores=scores,
        )

        self.exporter.record_decision(
            decision,
            ProviderType.OPENAI,
            RoutingStrategy.PARETO_BALANCED
        )

        text = self.exporter.export_prometheus_text()

        # Check standard Prometheus exposition headers
        assert "# HELP pareto_router_tier_dispatch_total" in text
        assert "# TYPE pareto_router_tier_dispatch_total counter"
        assert 'pareto_router_tier_dispatch_total{provider="openai",strategy="pareto_balanced",tier="compact"} 1' in text

        assert "# HELP pareto_router_effort_level_total" in text
        assert 'pareto_router_effort_level_total{effort_level="none",provider="openai"} 1' in text

        assert "# HELP pareto_router_overthinking_penalties_total" in text
        assert 'pareto_router_overthinking_penalties_total{provider="openai"} 1' in text

        assert "# HELP pareto_router_overhead_seconds" in text
        assert '# TYPE pareto_router_overhead_seconds histogram' in text
        assert 'pareto_router_overhead_seconds_count{evaluator="HeuristicEvaluator"} 1' in text

    def test_metrics_summary_dict(self) -> None:
        scores = EvaluationScores(
            logic_score=0.9,
            knowledge_score=0.8,
            latency_ms=0.85,
            evaluator_name="HeuristicEvaluator",
        )
        decision = RoutingDecision(
            tier=ModelTier.FRONTIER,
            effort_score=0.9,
            effort_level=EffortLevel.HIGH,
            mode=ModelMode.REASONING,
            reasoning_budget_tokens=25000,
            overthinking_penalized=False,
            scores=scores,
        )

        self.exporter.record_decision(decision, ProviderType.ANTHROPIC, RoutingStrategy.PARETO_QUALITY)

        summary = self.exporter.get_summary()
        assert summary["total_routed_requests"] == 1
        assert summary["overthinking_penalties_clamped"] == 0
        assert summary["average_evaluation_overhead_ms"] > 0
        assert "frontier:anthropic" in summary["tier_dispatches"]
        assert "high:anthropic" in summary["effort_dispatches"]

    def test_router_integration_with_metrics(self) -> None:
        router = ParetoRouter(enable_metrics=True)
        assert router.metrics is not None

        router.chat.completions.create(
            messages=[{"role": "user", "content": "Hello world!"}]
        )
        router.chat.completions.create(
            messages=[{"role": "user", "content": "Implement a binary search tree in Rust"}]
        )

        summary = router.metrics.get_summary()
        assert summary["total_routed_requests"] == 2
        prometheus_text = router.metrics.export_prometheus_text()
        assert "pareto_router_tier_dispatch_total" in prometheus_text
