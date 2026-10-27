"""Route one prompt and inspect the full 2D decision.

    python examples/01_route_and_inspect.py

No API key and no network call: the decision is a pure function of the two axis
scores, which is why it costs well under a millisecond.
"""

from pareto_router import (
    HeuristicEvaluator,
    ParetoPolicy,
    ProviderNormalizer,
    ProviderType,
    RoutingStrategy,
)

PROMPT = "Debug this race condition in a Rust mutex."

scores = HeuristicEvaluator().evaluate_sync(PROMPT)
decision = ParetoPolicy().evaluate_coordinates(scores, strategy=RoutingStrategy.PARETO_QUALITY)
payload = ProviderNormalizer().to_provider_payload(
    decision, ProviderType.OPENAI, [{"role": "user", "content": PROMPT}]
)

print(f"prompt            : {PROMPT}")
print(f"logic score       : {scores.logic_score:.3f}")
print(f"knowledge score   : {scores.knowledge_score:.3f}")
print(f"evaluation latency: {scores.latency_ms:.3f} ms")
print(f"tier              : {decision.tier.value}")
print(f"effort            : {decision.effort_level.value} ({decision.effort_score:.3f})")
print(f"mode              : {decision.mode.value}")
print(f"thinking budget   : {decision.reasoning_budget_tokens} tokens")
print(f"overthinking clamp: {decision.overthinking_penalized}")
print(f"dispatch model    : {payload.model}")
print(f"reasoning_effort  : {payload.body.get('reasoning_effort')}")
