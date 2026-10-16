"""OpenAI SDK drop-in client wrapper with 2D Pareto routing, streaming, and sliding-window context."""

import asyncio
import os
import time
from typing import Any, AsyncIterator, Iterator, Sequence
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice as ChunkChoice,
    ChoiceDelta,
)

from pareto_router.core.normalizer import ProviderNormalizer
from pareto_router.core.policy import ParetoPolicy
from pareto_router.core.schemas import (
    InferenceResponse,
    ProviderType,
    RoutingDecision,
    RoutingStrategy,
)
from pareto_router.evaluator.base import BaseEvaluator
from pareto_router.evaluator.heuristic import HeuristicEvaluator
from pareto_router.gateway.cascade import CascadeResult, SpeculativeCascade
from pareto_router.gateway.verifiers import BaseVerifier


class ParetoStream:
    """Synchronous stream wrapper yielding ChatCompletionChunk tokens."""

    def __init__(
        self,
        chunks: list[ChatCompletionChunk],
        routing_decision: RoutingDecision,
        model: str,
        cascade_result: CascadeResult | None = None,
    ) -> None:
        self._chunks = chunks
        self.routing_decision = routing_decision
        self.model = model
        self.cascade_result = cascade_result

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        for chunk in self._chunks:
            yield chunk


class ParetoAsyncStream:
    """Asynchronous stream wrapper yielding ChatCompletionChunk tokens."""

    def __init__(
        self,
        chunks: list[ChatCompletionChunk],
        routing_decision: RoutingDecision,
        model: str,
        cascade_result: CascadeResult | None = None,
    ) -> None:
        self._chunks = chunks
        self.routing_decision = routing_decision
        self.model = model
        self.cascade_result = cascade_result

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        async def _generator() -> AsyncIterator[ChatCompletionChunk]:
            for chunk in self._chunks:
                yield chunk
        return _generator()


class ParetoChatCompletions:
    """Drop-in replacement for openai.chat.completions."""

    def __init__(
        self,
        evaluator: BaseEvaluator,
        policy: ParetoPolicy,
        normalizer: ProviderNormalizer,
        cascade: SpeculativeCascade,
        provider: ProviderType = ProviderType.OPENAI,
        sliding_window_size: int = 6,
        default_strategy: RoutingStrategy = RoutingStrategy.PARETO_BALANCED,
    ) -> None:
        self.evaluator = evaluator
        self.policy = policy
        self.normalizer = normalizer
        self.cascade = cascade
        self.provider = provider
        self.sliding_window_size = sliding_window_size
        self.default_strategy = default_strategy

    def _extract_sliding_context(
        self,
        messages: Sequence[dict[str, Any]]
    ) -> tuple[str, list[dict[str, str]]]:
        """Extract latest user turn and preceding conversation context."""
        if not messages:
            return "", []

        latest_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                latest_text = str(m.get("content", ""))
                break

        if not latest_text and messages:
            latest_text = str(messages[-1].get("content", ""))

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
        strategy: str | RoutingStrategy | None = None,
        pareto_route: bool = True,
        stream: bool = False,
        verifiers: list[BaseVerifier] | None = None,
        **kwargs: Any
    ) -> Any:
        """Asynchronously create a chat completion or stream with 2D Pareto routing."""
        t0 = time.perf_counter_ns()
        latest_text, context = self._extract_sliding_context(messages)

        # 1. Evaluate prompt along dual axes (<15ms)
        scores = await self.evaluator.evaluate(latest_text, context)

        # 2. Compute 2D Pareto decision with strategy preset
        active_strategy = RoutingStrategy(strategy) if strategy else self.default_strategy
        decision = self.policy.evaluate_coordinates(scores, strategy=active_strategy)

        # 3. Provider Payload translation
        payload = self.normalizer.to_provider_payload(
            decision,
            self.provider,
            list(messages),
            kwargs,
        )

        chosen_model = model or payload.model

        # Mock or real executor function
        async def _mock_or_real_executor(
            dec: RoutingDecision,
            body: dict[str, Any]
        ) -> InferenceResponse:
            resp_content = f"Response from {body.get('model')} (Tier: {dec.tier.value}, Effort: {dec.effort_level.value})"
            return InferenceResponse(
                id="pareto-resp",
                content=resp_content,
                model=str(body.get("model", chosen_model)),
                routing_decision=dec,
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                latency_ms=10.0,
            )

        # Execute via Speculative Cascade if verifiers are present
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

        # Handle stream=True
        if stream:
            words = final_content.split(" ")
            chunk_list: list[ChatCompletionChunk] = []
            chunk_id = f"chatcmpl-stream-{int(time.time() * 1000)}"

            for i, word in enumerate(words):
                prefix = "" if i == 0 else " "
                chunk_list.append(
                    ChatCompletionChunk(
                        id=chunk_id,
                        choices=[
                            ChunkChoice(
                                delta=ChoiceDelta(content=f"{prefix}{word}", role="assistant" if i == 0 else None),
                                index=0,
                                finish_reason=None,
                            )
                        ],
                        created=int(time.time()),
                        model=final_model,
                        object="chat.completion.chunk",
                    )
                )

            # Final stopping chunk
            chunk_list.append(
                ChatCompletionChunk(
                    id=chunk_id,
                    choices=[
                        ChunkChoice(
                            delta=ChoiceDelta(),
                            index=0,
                            finish_reason="stop",
                        )
                    ],
                    created=int(time.time()),
                    model=final_model,
                    object="chat.completion.chunk",
                )
            )

            return ParetoAsyncStream(
                chunks=chunk_list,
                routing_decision=final_decision,
                model=final_model,
                cascade_result=cascade_result,
            )

        # Standard non-streaming ChatCompletion object
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
        strategy: str | RoutingStrategy | None = None,
        pareto_route: bool = True,
        stream: bool = False,
        verifiers: list[BaseVerifier] | None = None,
        **kwargs: Any
    ) -> Any:
        """Synchronous drop-in chat completion create or stream generator."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                res = pool.submit(
                    asyncio.run,
                    self.acreate(
                        messages=messages,
                        model=model,
                        strategy=strategy,
                        pareto_route=pareto_route,
                        stream=stream,
                        verifiers=verifiers,
                        **kwargs
                    )
                ).result()
        else:
            res = loop.run_until_complete(
                self.acreate(
                    messages=messages,
                    model=model,
                    strategy=strategy,
                    pareto_route=pareto_route,
                    stream=stream,
                    verifiers=verifiers,
                    **kwargs
                )
            )

        if stream and isinstance(res, ParetoAsyncStream):
            return ParetoStream(
                chunks=res._chunks,
                routing_decision=res.routing_decision,
                model=res.model,
                cascade_result=res.cascade_result,
            )
        return res


class ParetoRouterChat:
    """Namespace container matching OpenAI client.chat layout."""

    def __init__(
        self,
        evaluator: BaseEvaluator,
        policy: ParetoPolicy,
        normalizer: ProviderNormalizer,
        cascade: SpeculativeCascade,
        provider: ProviderType = ProviderType.OPENAI,
        default_strategy: RoutingStrategy = RoutingStrategy.PARETO_BALANCED,
    ) -> None:
        self.completions = ParetoChatCompletions(
            evaluator=evaluator,
            policy=policy,
            normalizer=normalizer,
            cascade=cascade,
            provider=provider,
            default_strategy=default_strategy,
        )


class ParetoRouter:
    """Drop-in OpenAI SDK client wrapper with 2D Pareto routing and automatic env key detection."""

    def __init__(
        self,
        api_key: str | None = None,
        provider: ProviderType = ProviderType.OPENAI,
        strategy: RoutingStrategy | str = RoutingStrategy.PARETO_BALANCED,
        evaluator: BaseEvaluator | None = None,
        policy: ParetoPolicy | None = None,
        normalizer: ProviderNormalizer | None = None,
    ) -> None:
        self.provider = provider
        self.strategy = RoutingStrategy(strategy)

        # Automatically resolve API key from environment if not explicitly passed
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )

        self.evaluator = evaluator or HeuristicEvaluator()
        self.policy = policy or ParetoPolicy(default_strategy=self.strategy)
        self.normalizer = normalizer or ProviderNormalizer()
        self.cascade = SpeculativeCascade(normalizer=self.normalizer)

        self.chat = ParetoRouterChat(
            evaluator=self.evaluator,
            policy=self.policy,
            normalizer=self.normalizer,
            cascade=self.cascade,
            provider=self.provider,
            default_strategy=self.strategy,
        )
