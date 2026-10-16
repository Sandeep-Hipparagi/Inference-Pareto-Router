"""Unit tests for Phase 5 refinements: multi-turn context, streaming, pluggable decision endpoints, and strategy presets."""

import os
import pytest
from pareto_router.core.policy import ParetoPolicy
from pareto_router.core.schemas import (
    EffortLevel,
    EvaluationScores,
    ModelTier,
    RoutingStrategy,
)
from pareto_router.evaluator.decision_model import DecisionModelClient
from pareto_router.evaluator.heuristic import HeuristicEvaluator
from pareto_router.gateway.client import ParetoAsyncStream, ParetoRouter, ParetoStream


class TestMultiTurnContextTriage:
    def test_referential_bug_fix_inherits_code_context(self) -> None:
        evaluator = HeuristicEvaluator()
        context = [
            {
                "role": "user",
                "content": "Here is our concurrent worker loop:\n```go\nfunc worker() {\n  mu.Lock()\n  ch <- 1\n  mu.Unlock()\n}\n```"
            },
            {
                "role": "assistant",
                "content": "I see a potential channel deadlock if the buffer is full."
            }
        ]

        # In isolation, "Fix the bug" is only 3 words and has 0 code tokens
        scores = evaluator.evaluate_sync("Fix the bug", context=context)

        # But because it refers to the code in context, logic must be elevated!
        assert scores.logic_score >= 0.70
        assert scores.metadata.get("is_referential_followup") is True
        assert scores.metadata.get("prior_had_code_or_logic") is True
        assert scores.metadata.get("is_low_entropy") is False

        # Verify that policy does NOT clamp it with overthinking penalty
        policy = ParetoPolicy()
        decision = policy.evaluate_coordinates(scores)
        assert decision.overthinking_penalized is False
        assert decision.tier in (ModelTier.MID, ModelTier.FRONTIER)
        assert decision.effort_level in (EffortLevel.MEDIUM, EffortLevel.HIGH)

    def test_isolated_short_text_remains_low_entropy(self) -> None:
        evaluator = HeuristicEvaluator()
        # No code in prior context
        context = [
            {"role": "user", "content": "How's the weather today?"},
            {"role": "assistant", "content": "It looks sunny."}
        ]
        scores = evaluator.evaluate_sync("Thanks a lot!", context=context)
        assert scores.metadata.get("is_low_entropy") is True


class TestStreamingGateway:
    def test_sync_stream_generation(self) -> None:
        router = ParetoRouter()
        stream = router.chat.completions.create(
            messages=[{"role": "user", "content": "Tell me a joke"}],
            stream=True,
        )

        assert isinstance(stream, ParetoStream)
        assert stream.routing_decision is not None

        chunks = list(stream)
        assert len(chunks) > 0
        text = "".join(c.choices[0].delta.content or "" for c in chunks)
        assert len(text) > 0
        # Last chunk has stop finish reason
        assert chunks[-1].choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_async_stream_generation(self) -> None:
        router = ParetoRouter()
        stream = await router.chat.completions.acreate(
            messages=[{"role": "user", "content": "Hello world"}],
            stream=True,
        )

        assert isinstance(stream, ParetoAsyncStream)
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) > 0
        text = "".join(c.choices[0].delta.content or "" for c in chunks)
        assert len(text) > 0


class TestPluggableDecisionModel:
    def test_custom_flags_and_env_vars(self) -> None:
        os.environ["PARETO_ROUTER_DECISION_ENDPOINT"] = "http://127.0.0.1:8099/v1/drex"
        os.environ["PARETO_ROUTER_DECISION_KEY"] = "secret-token-123"

        client = DecisionModelClient(
            engine_type="drex",
            custom_headers={"X-Experiment-Group": "canary"},
            schema_mapping={"logic_key": "custom_logic", "knowledge_key": "custom_knowledge"},
        )

        assert client.endpoint_url == "http://127.0.0.1:8099/v1/drex"
        assert client.api_key == "secret-token-123"
        assert client.engine_type == "drex"
        headers = client._build_headers()
        assert headers.get("Authorization") == "Bearer secret-token-123"
        assert headers.get("X-Experiment-Group") == "canary"

    def test_custom_response_parser(self) -> None:
        client = DecisionModelClient(
            response_parser=lambda data: (0.88, 0.44)
        )
        logic, know = client._parse_response({"any": "data"})
        assert logic == 0.88
        assert know == 0.44


class TestStrategyPresets:
    def setup_method(self) -> None:
        self.policy = ParetoPolicy()

    def test_pareto_cost_vs_quality(self) -> None:
        # Moderate problem (logic 0.40, knowledge 0.40)
        scores = EvaluationScores(
            logic_score=0.40,
            knowledge_score=0.40,
            latency_ms=0.2,
            metadata={}
        )

        dec_cost = self.policy.evaluate_coordinates(scores, strategy=RoutingStrategy.PARETO_COST)
        dec_quality = self.policy.evaluate_coordinates(scores, strategy=RoutingStrategy.PARETO_QUALITY)

        # Pareto cost is more conservative
        assert dec_cost.effort_score < dec_quality.effort_score
        assert dec_cost.tier in (ModelTier.COMPACT, ModelTier.MID)
        # Pareto quality readily upgrades to Frontier or higher effort
        assert dec_quality.tier in (ModelTier.MID, ModelTier.FRONTIER)
        assert dec_quality.effort_score >= dec_cost.effort_score
