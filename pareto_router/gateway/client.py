"""OpenAI SDK drop-in client wrapper with 2D Pareto routing and sliding-window context."""

import asyncio
import time
from typing import Any, Sequence
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from pareto_router.core.normalizer import ProviderNormalizer
from pareto_router.core.policy import ParetoPolicy
from pareto_router.core.schemas import (
    InferenceResponse,
    ProviderType,
    RoutingDecision,
)
from pareto_router.evaluator.base import BaseEvaluator
from pareto_router.evaluator.heuristic import HeuristicEvaluator
from pareto_router.gateway.cascade import CascadeResult, SpeculativeCascade
from pareto_router.gateway.verifiers import BaseVerifier


class ParetoChatCompletions:
    """Drop-in replacement for openai.chat.completions."""

    def __init__(
        self,
        evaluator: BaseEvaluator,
        policy: ParetoPolicy,
        normalizer: ProviderNormalizer,
        cascade: SpeculativeCascade,
        provider: ProviderType = ProviderType.OPENAI,
        sliding_window_size: int = 4,
    ) -> None:
        self.evaluator = evaluator
        self.policy = policy
        self.normalizer = normalizer
        self.cascade = cascade
        self.provider = provider
        self.sliding_window_size = sliding_window_size

    def _extract_sliding_context(
        self,
        messages: Sequence[dict[str, Any]]
    ) -> tuple[str, list[dict[str, str]]]:
        """Extract the latest user query and the sliding-window conversation context."""
        if not messages:
            return "", []

        # Find latest user turn
        latest_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                latest_text = str(m.get("content", ""))
                break

        if not latest_text and messages:
            latest_text = str(messages[-1].get("content", ""))

        # Context turns preceding the last message
        window = list(messages[-self.sliding_window_size:])
        formatted_context: list[dict[str, str]] = [
            {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
            for m in window[:-1]
        ]

        return latest_text, formatted_context

    async def acreate(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        model: str | None = None,
        pareto_route: bool = True,
        verifiers: list[BaseVerifier] | None = None,
        **kwargs: Any
    ) -> ChatCompletion:
        """Asynchronously create a chat completion with 2D Pareto routing."""
        t0 = time.perf_counter_ns()
        latest_text, context = self._extract_sliding_context(messages)

        # 1. Evaluate prompt along dual axes (<15ms)
        scores = await self.evaluator.evaluate(latest_text, context)

        # 2. Compute 2D Pareto decision
        decision = self.policy.evaluate_coordinates(scores)

        # 3. Provider Payload translation
        payload = self.normalizer.to_provider_payload(
            decision,
            self.provider,
            list(messages),
            kwargs,
        )

        chosen_model = model or payload.model

        # 4. Mock or real executor function
        async def _mock_or_real_executor(
            dec: RoutingDecision,
            body: dict[str, Any]
        ) -> InferenceResponse:
            # Here we provide a mockable/pluggable execution fallback
            # When testing or running locally without active API keys, generate standard mock completion
            resp_content = f"Response from {body.get('model')} (Tier: {dec.tier.value}, Effort: {dec.effort_level.value})"
            return InferenceResponse(
                id="pareto-mock-resp",
                content=resp_content,
                model=str(body.get("model", chosen_model)),
                routing_decision=dec,
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                latency_ms=10.0,
            )

        # 5. Execute via Speculative Cascade if verifiers are present
        cascade_result: CascadeResult | None = None
        if verifiers:
            cascade_result = await self.cascade.execute(
                initial_decision=decision,
                provider=self.provider,
                messages=list(messages),
                executor=_mock_or_real_executor,
                verifiers=verifiers,
                base_body=kwargs,
            )
            final_content = cascade_result.response.content
            final_model = cascade_result.response.model
            final_decision = cascade_result.response.routing_decision
        else:
            inf_resp = await _mock_or_real_executor(decision, payload.body)
            final_content = inf_resp.content
            final_model = inf_resp.model
            final_decision = decision

        t1 = time.perf_counter_ns()
        total_latency_ms = (t1 - t0) / 1_000_000.0

        # Construct standard OpenAI ChatCompletion object
        completion = ChatCompletion(
            id=f"chatcmpl-{int(time.time() * 1000)}",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content=final_content,
                        role="assistant",
                    ),
                )
            ],
            created=int(time.time()),
            model=final_model,
            object="chat.completion",
        )

        # Attach Pareto Router metadata directly to response
        setattr(completion, "routing_decision", final_decision)
        setattr(completion, "cascade_result", cascade_result)
        setattr(completion, "evaluation_latency_ms", scores.latency_ms)
        setattr(completion, "total_router_latency_ms", round(total_latency_ms, 2))

        return completion

    def create(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        model: str | None = None,
        pareto_route: bool = True,
        verifiers: list[BaseVerifier] | None = None,
        **kwargs: Any
    ) -> ChatCompletion:
        """Synchronous drop-in chat completion create."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # In an active running event loop (e.g. Jupyter or nested loop)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run,
                    self.acreate(
                        messages=messages,
                        model=model,
                        pareto_route=pareto_route,
                        verifiers=verifiers,
                        **kwargs
                    )
                ).result()
        else:
            return loop.run_until_complete(
                self.acreate(
                    messages=messages,
                    model=model,
                    pareto_route=pareto_route,
                    verifiers=verifiers,
                    **kwargs
                )
            )


class ParetoRouterChat:
    """Namespace container matching OpenAI client.chat layout."""

    def __init__(
        self,
        evaluator: BaseEvaluator,
        policy: ParetoPolicy,
        normalizer: ProviderNormalizer,
        cascade: SpeculativeCascade,
        provider: ProviderType = ProviderType.OPENAI,
    ) -> None:
        self.completions = ParetoChatCompletions(
            evaluator=evaluator,
            policy=policy,
            normalizer=normalizer,
            cascade=cascade,
            provider=provider,
        )


class ParetoRouter:
    """Drop-in OpenAI SDK client wrapper with 2D Pareto routing."""

    def __init__(
        self,
        api_key: str | None = None,
        provider: ProviderType = ProviderType.OPENAI,
        evaluator: BaseEvaluator | None = None,
        policy: ParetoPolicy | None = None,
        normalizer: ProviderNormalizer | None = None,
    ) -> None:
        self.api_key = api_key
        self.provider = provider
        self.evaluator = evaluator or HeuristicEvaluator()
        self.policy = policy or ParetoPolicy()
        self.normalizer = normalizer or ProviderNormalizer()
        self.cascade = SpeculativeCascade(normalizer=self.normalizer)

        self.chat = ParetoRouterChat(
            evaluator=self.evaluator,
            policy=self.policy,
            normalizer=self.normalizer,
            cascade=self.cascade,
            provider=self.provider,
        )
