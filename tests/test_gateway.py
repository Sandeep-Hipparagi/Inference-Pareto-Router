"""Unit tests for Gateway client, multi-turn sliding window, verifiers, and speculative cascades."""

from typing import Any
import pytest
from pareto_router.core.schemas import (
    EffortLevel,
    EvaluationScores,
    InferenceResponse,
    ModelMode,
    ModelTier,
    ProviderType,
    RoutingDecision,
)
from pareto_router.gateway.cascade import SpeculativeCascade
from pareto_router.gateway.client import ParetoRouter
from pareto_router.gateway.verifiers import (
    JsonSchemaVerifier,
    PythonSyntaxVerifier,
    RegexVerifier,
)


class TestVerifiers:
    def test_json_schema_verifier_valid(self) -> None:
        verifier = JsonSchemaVerifier(required_keys=["name", "age"])
        valid, err = verifier.verify('```json\n{"name": "Alice", "age": 30}\n```')
        assert valid is True
        assert err is None

    def test_json_schema_verifier_missing_keys(self) -> None:
        verifier = JsonSchemaVerifier(required_keys=["name", "role"])
        valid, err = verifier.verify('{"name": "Alice"}')
        assert valid is False
        assert "Missing required JSON keys" in str(err)

    def test_json_schema_verifier_invalid_json(self) -> None:
        verifier = JsonSchemaVerifier()
        valid, err = verifier.verify('{name: "Alice", broken}')
        assert valid is False
        assert "JSON parsing failed" in str(err)

    def test_python_syntax_verifier_valid(self) -> None:
        verifier = PythonSyntaxVerifier()
        code = "```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```"
        valid, err = verifier.verify(code)
        assert valid is True
        assert err is None

    def test_python_syntax_verifier_invalid(self) -> None:
        verifier = PythonSyntaxVerifier()
        code = "def broken_func(:\n    return 0"
        valid, err = verifier.verify(code)
        assert valid is False
        assert "Python SyntaxError" in str(err)

    def test_regex_verifier(self) -> None:
        verifier = RegexVerifier(r"^ANSWER:\s*\d+$")
        valid, err = verifier.verify("ANSWER: 42")
        assert valid is True

        invalid, err = verifier.verify("The answer is forty-two.")
        assert invalid is False


class TestGatewayClient:
    def test_drop_in_completions_create(self) -> None:
        router = ParetoRouter()
        completion = router.chat.completions.create(
            messages=[{"role": "user", "content": "Hi there!"}]
        )
        assert completion.choices[0].message.content is not None
        decision: RoutingDecision = getattr(completion, "routing_decision")
        assert decision.tier == ModelTier.COMPACT
        assert decision.effort_level == EffortLevel.NONE
        assert decision.overthinking_penalized is True

    def test_algorithmic_query_routing(self) -> None:
        router = ParetoRouter()
        completion = router.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Implement an algorithm to detect race conditions in a distributed lock manager using Raft invariants."
                }
            ]
        )
        decision: RoutingDecision = getattr(completion, "routing_decision")
        assert decision.tier in (ModelTier.MID, ModelTier.FRONTIER)
        assert decision.effort_level in (EffortLevel.MEDIUM, EffortLevel.HIGH)

    def test_sliding_window_context_accumulation(self) -> None:
        router = ParetoRouter()
        messages = [
            {"role": "user", "content": "We are architecting a distributed Byzantine fault-tolerant consensus engine."},
            {"role": "assistant", "content": "Understood. Byzantine fault tolerance requires 3f + 1 nodes."},
            {"role": "user", "content": "How do we handle network partitions and prove liveness without deadlocks?"}
        ]
        completion = router.chat.completions.create(messages=messages)
        decision: RoutingDecision = getattr(completion, "routing_decision")
        # Context should elevate both knowledge and logic
        assert decision.scores.knowledge_score >= 0.35
        assert decision.tier in (ModelTier.MID, ModelTier.FRONTIER)
        assert decision.mode == ModelMode.REASONING


@pytest.mark.asyncio
class TestSpeculativeCascade:
    async def test_successful_on_first_try(self) -> None:
        cascade = SpeculativeCascade()

        decision = RoutingDecision(
            tier=ModelTier.COMPACT,
            effort_score=0.1,
            effort_level=EffortLevel.LOW,
            mode=ModelMode.STANDARD,
            scores=EvaluationScores(logic_score=0.2, knowledge_score=0.2, latency_ms=0.5)
        )

        async def _mock_executor(dec: RoutingDecision, body: dict[str, Any]) -> InferenceResponse:
            return InferenceResponse(
                id="test-1",
                content='{"status": "ok"}',
                model=str(body.get("model", "gpt-4o-mini")),
                routing_decision=dec,
            )

        res = await cascade.execute(
            initial_decision=decision,
            provider=ProviderType.OPENAI,
            messages=[{"role": "user", "content": "ping"}],
            executor=_mock_executor,
            verifiers=[JsonSchemaVerifier()],
        )

        assert res.escalated is False
        assert len(res.attempts) == 1
        assert res.attempts[0].status == "success"
        assert res.final_tier == ModelTier.COMPACT

    async def test_escalation_on_verification_failure(self) -> None:
        cascade = SpeculativeCascade()

        initial_decision = RoutingDecision(
            tier=ModelTier.COMPACT,
            effort_score=0.1,
            effort_level=EffortLevel.LOW,
            mode=ModelMode.STANDARD,
            scores=EvaluationScores(logic_score=0.2, knowledge_score=0.2, latency_ms=0.5)
        )

        call_count = 0

        async def _mock_escalating_executor(dec: RoutingDecision, body: dict[str, Any]) -> InferenceResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First attempt (compact) returns broken syntax
                return InferenceResponse(
                    id="test-attempt-1",
                    content="def invalid_syntax(: return",
                    model=str(body.get("model", "gpt-4o-mini")),
                    routing_decision=dec,
                )
            else:
                # Second attempt (frontier) returns valid syntax
                return InferenceResponse(
                    id="test-attempt-2",
                    content="def valid_syntax() -> int:\n    return 42",
                    model=str(body.get("model", "o1")),
                    routing_decision=dec,
                )

        res = await cascade.execute(
            initial_decision=initial_decision,
            provider=ProviderType.OPENAI,
            messages=[{"role": "user", "content": "write code"}],
            executor=_mock_escalating_executor,
            verifiers=[PythonSyntaxVerifier()],
        )

        assert res.escalated is True
        assert len(res.attempts) == 2
        assert res.attempts[0].status == "validation_failed"
        assert res.attempts[0].tier == ModelTier.COMPACT
        assert res.attempts[1].status == "success"
        assert res.attempts[1].tier == ModelTier.FRONTIER
        assert res.final_tier == ModelTier.FRONTIER
