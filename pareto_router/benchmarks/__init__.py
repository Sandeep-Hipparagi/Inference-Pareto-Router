"""Benchmarks package."""

from pareto_router.benchmarks.run_benchmarks import run_benchmark_suite
from pareto_router.benchmarks.scenarios import SCENARIOS, BenchmarkScenario

__all__ = ["BenchmarkScenario", "SCENARIOS", "run_benchmark_suite"]
