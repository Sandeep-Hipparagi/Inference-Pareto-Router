"""Unit tests for Evaluator engines, latency benchmarks, and fallbacks."""

import pytest
from pareto_router.evaluator.decision_model import DecisionModelClient
from pareto_router.evaluator.heuristic import HeuristicEvaluator
from pareto_router.evaluator.onnx_evaluator import ONNXEvaluator


class TestHeuristicEvaluator:
    def setup_method(self) -> None:
        self.evaluator = HeuristicEvaluator()

    def test_greeting_triage(self) -> None:
        scores = self.evaluator.evaluate_sync("Hello there! Good morning.")
        assert scores.logic_score < 0.15
        assert scores.metadata.get("is_low_entropy") is True
        assert scores.metadata.get("is_greeting") is True
        assert scores.latency_ms < 15.0  # Must be well under 15ms

    def test_aesthetic_rewrite_triage(self) -> None:
        scores = self.evaluator.evaluate_sync("Can you make this email sound more professional and polite?")
        assert scores.logic_score < 0.20
        assert scores.metadata.get("is_low_entropy") is True
        assert scores.metadata.get("is_aesthetic") is True
        assert scores.latency_ms < 15.0

    def test_code_logic_triage(self) -> None:
        prompt = (
            "Write a Python function to detect a race condition or deadlock between two threads "
            "using mutex and semaphores with O(1) lock acquisition complexity."
        )
        scores = self.evaluator.evaluate_sync(prompt)
        assert scores.logic_score > 0.60
        assert scores.metadata.get("is_low_entropy") is False
        assert scores.latency_ms < 15.0

    def test_math_and_regex_triage(self) -> None:
        prompt = "Provide a regular expression that validates RFC 5322 email addresses and prove its correctness."
        scores = self.evaluator.evaluate_sync(prompt)
        assert scores.logic_score > 0.45
        assert scores.latency_ms < 15.0

    def test_knowledge_nuance_triage(self) -> None:
        prompt = (
            "Explain the philosophical epistemology of Kant's Critique of Pure Reason and its "
            "subsequent influence on 19th-century German idealism and metaphysics."
        )
        scores = self.evaluator.evaluate_sync(prompt)
        assert scores.knowledge_score > 0.50
        assert scores.logic_score < 0.35
        assert scores.latency_ms < 15.0

    def test_sub_15ms_latency_budget(self) -> None:
        # Benchmark 50 consecutive runs to ensure max latency < 15ms
        long_prompt = (
            "Given an undirected graph with N vertices and M edges, devise a dynamic programming "
            "algorithm to compute the chromatic number. Analyze the time complexity O(2^n * n) "
            "and discuss how this relates to NP-completeness and polynomial reductions."
        )
        latencies = []
        for _ in range(50):
            res = self.evaluator.evaluate_sync(long_prompt)
            latencies.append(res.latency_ms)

        max_latency = max(latencies)
        avg_latency = sum(latencies) / len(latencies)
        assert max_latency < 15.0, f"Max latency {max_latency}ms exceeded 15ms budget"
        assert avg_latency < 2.0, f"Average latency {avg_latency}ms should be under 2ms"


class TestONNXEvaluator:
    def test_fallback_without_model(self) -> None:
        evaluator = ONNXEvaluator(model_path=None)
        scores = evaluator.evaluate_sync("def solve_knapsack(weights, values, capacity): return 0")

        assert scores.logic_score > 0.30
        assert scores.metadata.get("onnx_active") is False
        assert scores.latency_ms < 15.0

    def test_nonexistent_model_path_fallback(self) -> None:
        evaluator = ONNXEvaluator(model_path="nonexistent_model.onnx")
        scores = evaluator.evaluate_sync("Explain the Treaty of Westphalia and its sovereignty doctrine.")

        assert scores.knowledge_score > 0.40
        assert scores.metadata.get("onnx_active") is False
        assert scores.latency_ms < 15.0


@pytest.mark.asyncio
class TestDecisionModelClient:
    async def test_fallback_on_unreachable_endpoint(self) -> None:
        # Points to a closed port on localhost with 10ms timeout
        client = DecisionModelClient(
            endpoint_url="http://127.0.0.1:59999/v1/triage",
            timeout_ms=10.0
        )
        scores = await client.evaluate("How do I implement binary search in Rust?")

        assert scores.logic_score > 0.30
        assert scores.latency_ms < 100.0  # Safe recovery without hanging
