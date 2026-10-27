"""Route the same prompt under all three strategy presets.

    python examples/02_compare_strategies.py

Shows how a preset moves the operating point along the frontier without changing
the algorithm: cost trims effort and delays escalation, quality does the reverse.
"""

from pareto_router import HeuristicEvaluator, ParetoPolicy, RoutingStrategy

PROMPTS = [
    "hi there",
    "Summarise this error log in one line.",
    "Prove that sqrt(2) is irrational.",
    "Refactor this 400-line React component and explain the trade-offs.",
]

PROVIDER_LABEL = {
    RoutingStrategy.PARETO_COST: "cost",
    RoutingStrategy.PARETO_BALANCED: "balanced",
    RoutingStrategy.PARETO_QUALITY: "quality",
}

evaluator = HeuristicEvaluator()
policy = ParetoPolicy()

header = f"{'prompt':52s} " + " ".join(f"{n:>20s}" for n in ("cost", "balanced", "quality"))
print(header)
print("-" * len(header))

for prompt in PROMPTS:
    scores = evaluator.evaluate_sync(prompt)
    cells = []
    for strategy in (RoutingStrategy.PARETO_COST, RoutingStrategy.PARETO_BALANCED, RoutingStrategy.PARETO_QUALITY):
        decision = policy.evaluate_coordinates(scores, strategy=strategy)
        cells.append(f"{decision.tier.value}/{decision.effort_level.value}")
    label = prompt if len(prompt) <= 50 else prompt[:47] + "..."
    print(f"{label:52s} " + " ".join(f"{c:>20s}" for c in cells))
