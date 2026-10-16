"""Prometheus and OpenTelemetry-ready metrics exporter for Pareto Router."""

from collections import defaultdict
import threading
from typing import Any
from pareto_router.core.schemas import (
    EffortLevel,
    ModelTier,
    ProviderType,
    RoutingDecision,
    RoutingStrategy,
)


class PrometheusMetricsExporter:
    """Thread-safe zero-dependency Prometheus metrics accumulator and text exporter."""

    HISTOGRAM_BUCKETS = (0.0002, 0.0005, 0.001, 0.002, 0.005, 0.010, 0.015, float("inf"))

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Counters: key -> count
        # (tier, provider, strategy) -> count
        self._tier_dispatches: dict[tuple[str, str, str], int] = defaultdict(int)
        # (effort_level, provider) -> count
        self._effort_dispatches: dict[tuple[str, str], int] = defaultdict(int)
        # provider -> count
        self._overthinking_penalties: dict[str, int] = defaultdict(int)

        # Histogram: evaluator -> {bucket: count}, sum, total_count
        self._hist_counts: dict[str, dict[float, int]] = defaultdict(lambda: defaultdict(int))
        self._hist_sums: dict[str, float] = defaultdict(float)
        self._hist_totals: dict[str, int] = defaultdict(int)

    def record_decision(
        self,
        decision: RoutingDecision,
        provider: ProviderType | str,
        strategy: RoutingStrategy | str = RoutingStrategy.PARETO_BALANCED,
    ) -> None:
        """Record telemetry for a completed routing decision."""
        prov_str = provider.value if isinstance(provider, ProviderType) else str(provider)
        strat_str = strategy.value if isinstance(strategy, RoutingStrategy) else str(strategy)
        tier_str = decision.tier.value if isinstance(decision.tier, ModelTier) else str(decision.tier)
        effort_str = (
            decision.effort_level.value
            if isinstance(decision.effort_level, EffortLevel)
            else str(decision.effort_level)
        )
        evaluator_name = decision.scores.evaluator_name or "HeuristicEvaluator"
        overhead_sec = decision.scores.latency_ms / 1000.0

        with self._lock:
            # 1. Tier dispatch counter
            self._tier_dispatches[(tier_str, prov_str, strat_str)] += 1

            # 2. Effort level counter
            self._effort_dispatches[(effort_str, prov_str)] += 1

            # 3. Overthinking penalty counter
            if decision.overthinking_penalized:
                self._overthinking_penalties[prov_str] += 1

            # 4. Latency overhead histogram
            self._hist_totals[evaluator_name] += 1
            self._hist_sums[evaluator_name] += overhead_sec
            for b in self.HISTOGRAM_BUCKETS:
                if overhead_sec <= b:
                    self._hist_counts[evaluator_name][b] += 1

    def export_prometheus_text(self) -> str:
        """Format metrics strictly according to Prometheus text exposition format (0.0.4)."""
        lines: list[str] = []

        with self._lock:
            # 1. Tier Dispatches
            lines.append("# HELP pareto_router_tier_dispatch_total Total requests routed per model tier, provider, and strategy.")
            lines.append("# TYPE pareto_router_tier_dispatch_total counter")
            for (tier, prov, strat), count in sorted(self._tier_dispatches.items()):
                lines.append(f'pareto_router_tier_dispatch_total{{provider="{prov}",strategy="{strat}",tier="{tier}"}} {count}')

            # 2. Effort Level Dispatches
            lines.append("# HELP pareto_router_effort_level_total Total requests routed per reasoning effort level and provider.")
            lines.append("# TYPE pareto_router_effort_level_total counter")
            for (effort, prov), count in sorted(self._effort_dispatches.items()):
                lines.append(f'pareto_router_effort_level_total{{effort_level="{effort}",provider="{prov}"}} {count}')

            # 3. Overthinking Penalties
            lines.append("# HELP pareto_router_overthinking_penalties_total Total requests where reasoning effort was clamped to 0.0 to prevent token degradation.")
            lines.append("# TYPE pareto_router_overthinking_penalties_total counter")
            for prov, count in sorted(self._overthinking_penalties.items()):
                lines.append(f'pareto_router_overthinking_penalties_total{{provider="{prov}"}} {count}')

            # 4. Latency Overhead Histogram
            lines.append("# HELP pareto_router_overhead_seconds Router evaluation latency overhead in seconds.")
            lines.append("# TYPE pareto_router_overhead_seconds histogram")
            for eval_name in sorted(self._hist_totals.keys()):
                cum = 0
                for b in self.HISTOGRAM_BUCKETS:
                    cum += self._hist_counts[eval_name][b]
                    le_str = "+Inf" if b == float("inf") else f"{b:.4f}"
                    lines.append(f'pareto_router_overhead_seconds_bucket{{evaluator="{eval_name}",le="{le_str}"}} {cum}')
                lines.append(f'pareto_router_overhead_seconds_sum{{evaluator="{eval_name}"}} {self._hist_sums[eval_name]:.6f}')
                lines.append(f'pareto_router_overhead_seconds_count{{evaluator="{eval_name}"}} {self._hist_totals[eval_name]}')

        return "\n".join(lines) + "\n"

    def get_summary(self) -> dict[str, Any]:
        """Return structured dictionary summary for JSON monitoring endpoints or logs."""
        with self._lock:
            total_requests = sum(self._tier_dispatches.values())
            overthinking_total = sum(self._overthinking_penalties.values())

            tier_breakdown = {f"{t}:{p}": c for (t, p, s), c in self._tier_dispatches.items()}
            effort_breakdown = {f"{e}:{p}": c for (e, p), c in self._effort_dispatches.items()}

            avg_overhead_ms = 0.0
            total_overhead_calls = sum(self._hist_totals.values())
            if total_overhead_calls > 0:
                avg_overhead_ms = (sum(self._hist_sums.values()) / total_overhead_calls) * 1000.0

            return {
                "total_routed_requests": total_requests,
                "overthinking_penalties_clamped": overthinking_total,
                "average_evaluation_overhead_ms": round(avg_overhead_ms, 3),
                "tier_dispatches": tier_breakdown,
                "effort_dispatches": effort_breakdown,
            }

    def reset(self) -> None:
        """Reset all in-memory metric counters."""
        with self._lock:
            self._tier_dispatches.clear()
            self._effort_dispatches.clear()
            self._overthinking_penalties.clear()
            self._hist_counts.clear()
            self._hist_sums.clear()
            self._hist_totals.clear()
