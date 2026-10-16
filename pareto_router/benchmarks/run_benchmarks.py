"""Benchmark runner evaluating 100 diverse scenarios for latency, cost, and route distribution."""

import json
from pathlib import Path
import time
from typing import Any
import numpy as np

from pareto_router.benchmarks.scenarios import SCENARIOS, BenchmarkScenario
from pareto_router.core.policy import ParetoPolicy
from pareto_router.core.schemas import EffortLevel, ModelTier
from pareto_router.evaluator.heuristic import HeuristicEvaluator


def run_benchmark_suite() -> dict[str, Any]:
    evaluator = HeuristicEvaluator()
    policy = ParetoPolicy()

    latencies_ms: list[float] = []
    tier_counts: dict[str, int] = {t.value: 0 for t in ModelTier}
    effort_counts: dict[str, int] = {e.value: 0 for e in EffortLevel}
    overthinking_penalties = 0
    correct_overthinking_detections = 0
    total_overthinking_candidates = sum(1 for s in SCENARIOS if s.is_overthinking_candidate)

    # Cost proxy calculation (relative to 100% Frontier High-Reasoning baseline)
    # Frontier High-Reasoning cost weight: 100.0 (high token multiplier + frontier price)
    # Frontier Standard: 25.0
    # Mid Reasoning: 20.0
    # Mid Standard: 8.0
    # Compact Reasoning: 3.0
    # Compact Standard: 1.0
    baseline_cost = len(SCENARIOS) * 100.0
    routed_cost = 0.0

    # Latency proxy (seconds taken by upstream inference)
    # Baseline: 15.0s per high-reasoning turn
    baseline_inference_time_s = len(SCENARIOS) * 15.0
    routed_inference_time_s = 0.0

    print("=" * 78)
    print(" INFERENCE-PARETO-ROUTER: 100-SCENARIO SYNTHETIC BENCHMARK")
    print("=" * 78)

    results_detail = []

    for sc in SCENARIOS:
        t0 = time.perf_counter_ns()
        scores = evaluator.evaluate_sync(sc.prompt)
        decision = policy.evaluate_coordinates(scores)
        t1 = time.perf_counter_ns()

        lat_ms = (t1 - t0) / 1_000_000.0
        latencies_ms.append(lat_ms)

        tier_counts[decision.tier.value] += 1
        effort_counts[decision.effort_level.value] += 1

        if decision.overthinking_penalized:
            overthinking_penalties += 1
            if sc.is_overthinking_candidate:
                correct_overthinking_detections += 1

        # Calculate cost weight for this route
        if decision.tier == ModelTier.FRONTIER:
            c_weight = 100.0 if decision.effort_level in (EffortLevel.MEDIUM, EffortLevel.HIGH) else 25.0
            inf_time = 14.0 if decision.effort_level in (EffortLevel.MEDIUM, EffortLevel.HIGH) else 2.5
        elif decision.tier == ModelTier.MID:
            c_weight = 20.0 if decision.effort_level != EffortLevel.NONE else 8.0
            inf_time = 6.0 if decision.effort_level != EffortLevel.NONE else 1.5
        else:
            c_weight = 3.0 if decision.effort_level != EffortLevel.NONE else 1.0
            inf_time = 3.0 if decision.effort_level != EffortLevel.NONE else 0.4

        routed_cost += c_weight
        routed_inference_time_s += inf_time

        results_detail.append({
            "id": sc.id,
            "category": sc.category,
            "tier": decision.tier.value,
            "effort": decision.effort_level.value,
            "logic_score": scores.logic_score,
            "knowledge_score": scores.knowledge_score,
            "latency_ms": lat_ms,
            "overthinking_penalized": decision.overthinking_penalized,
        })

    lat_arr = np.array(latencies_ms)
    cost_savings_pct = ((baseline_cost - routed_cost) / baseline_cost) * 100.0
    latency_savings_pct = ((baseline_inference_time_s - routed_inference_time_s) / baseline_inference_time_s) * 100.0

    summary = {
        "total_scenarios": len(SCENARIOS),
        "latency_metrics_ms": {
            "mean": round(float(np.mean(lat_arr)), 3),
            "median_p50": round(float(np.percentile(lat_arr, 50)), 3),
            "p90": round(float(np.percentile(lat_arr, 90)), 3),
            "p95": round(float(np.percentile(lat_arr, 95)), 3),
            "p99": round(float(np.percentile(lat_arr, 99)), 3),
            "max": round(float(np.max(lat_arr)), 3),
            "budget_threshold_ms": 15.0,
            "budget_passed": bool(np.max(lat_arr) < 15.0),
        },
        "tier_distribution": tier_counts,
        "effort_distribution": effort_counts,
        "overthinking_penalty": {
            "total_activations": overthinking_penalties,
            "expected_candidates": total_overthinking_candidates,
            "accuracy_pct": round((correct_overthinking_detections / total_overthinking_candidates) * 100.0, 1),
        },
        "efficiency_gains": {
            "estimated_cost_savings_pct": round(cost_savings_pct, 1),
            "estimated_latency_savings_pct": round(latency_savings_pct, 1),
            "baseline_model": "100% Frontier High-Reasoning (o1 / Claude 3.7 Thinking)",
        }
    }

    print("\n--- 1. ROUTING OVERHEAD LATENCY (<15ms Budget) ---")
    print(f"  P50 (Median)   : {summary['latency_metrics_ms']['median_p50']} ms")
    print(f"  P90            : {summary['latency_metrics_ms']['p90']} ms")
    print(f"  P95            : {summary['latency_metrics_ms']['p95']} ms")
    print(f"  P99            : {summary['latency_metrics_ms']['p99']} ms")
    print(f"  Max Latency    : {summary['latency_metrics_ms']['max']} ms")
    print(f"  Status         : {'PASSED (Strictly < 15ms)' if summary['latency_metrics_ms']['budget_passed'] else 'FAILED'}")

    print("\n--- 2. 2D ROUTE DISTRIBUTION ---")
    print("  Tier Breakdown:")
    for tier, count in tier_counts.items():
        print(f"    - {tier.upper():<10}: {count:>3} ({count / len(SCENARIOS) * 100:.1f}%)")
    print("  Effort Breakdown:")
    for effort, count in effort_counts.items():
        print(f"    - {effort.upper():<10}: {count:>3} ({count / len(SCENARIOS) * 100:.1f}%)")

    print("\n--- 3. OVERTHINKING PENALTY ---")
    print(f"  Activations    : {overthinking_penalties} triggers")
    print(f"  Target Accuracy: {summary['overthinking_penalty']['accuracy_pct']}% of aesthetic/greeting prompts clamped to 0 effort")

    print("\n--- 4. ESTIMATED SAVINGS VS ALL-FRONTIER REASONING BASELINE ---")
    print(f"  Estimated Cost Reduction   : {summary['efficiency_gains']['estimated_cost_savings_pct']}%")
    print(f"  Estimated Latency Reduction: {summary['efficiency_gains']['estimated_latency_savings_pct']}%")
    print("=" * 78)

    return summary


if __name__ == "__main__":
    report = run_benchmark_suite()
    out_file = Path("benchmark_results.json")
    out_file.write_text(json.dumps(report, indent=2))
    print(f"\nWrote full benchmark report to: {out_file.resolve()}\n")
