"""CLI entrypoint for Inference-Pareto-Router."""

import argparse
import sys
from pareto_router.core.normalizer import ProviderNormalizer
from pareto_router.core.policy import ParetoPolicy
from pareto_router.core.schemas import ProviderType
from pareto_router.evaluator.heuristic import HeuristicEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference-Pareto-Router: 2D Model Tier & Reasoning Effort Routing Engine"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        type=str,
        help="Prompt text to evaluate and route (reads from interactive input if omitted)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini"],
        default="openai",
        help="Target LLM provider (default: openai)",
    )

    args = parser.parse_args()

    prompt = args.prompt
    if not prompt:
        if sys.stdin.isatty():
            print("Enter prompt to evaluate (press Enter): ")
            prompt = input("> ")
        else:
            prompt = sys.stdin.read().strip()

    if not prompt:
        print("Error: No prompt provided.")
        sys.exit(1)

    evaluator = HeuristicEvaluator()
    policy = ParetoPolicy()
    normalizer = ProviderNormalizer()

    scores = evaluator.evaluate_sync(prompt)
    decision = policy.evaluate_coordinates(scores)
    provider_type = ProviderType(args.provider)
    payload = normalizer.to_provider_payload(
        decision, provider_type, [{"role": "user", "content": prompt}]
    )

    print("\n" + "=" * 60)
    print(" INFERENCE-PARETO-ROUTER: 2D TRIAGE DECISION")
    print("=" * 60)
    print(f" Prompt                : {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f" Evaluation Latency    : {scores.latency_ms:.3f} ms (<15ms budget)")
    print(f" Logic Score           : {scores.logic_score:.3f}")
    print(f" Knowledge/Nuance Score: {scores.knowledge_score:.3f}")
    print("-" * 60)
    print(f" Selected Model Tier   : {decision.tier.value.upper()}")
    print(f" Reasoning Mode        : {decision.mode.value.upper()}")
    print(f" Reasoning Effort      : {decision.effort_level.value.upper()} (score: {decision.effort_score:.2f})")
    print(f" Overthinking Penalty  : {'ACTIVATED (Clamped to 0)' if decision.overthinking_penalized else 'Inactive'}")
    if decision.reasoning_budget_tokens:
        print(f" Thinking Token Budget : {decision.reasoning_budget_tokens} tokens")
    print("-" * 60)
    print(f" Target Provider       : {provider_type.value.upper()}")
    print(f" Dispatched Model      : {payload.model}")
    print(" Payload Extras        : ")
    for k, v in payload.body.items():
        if k not in ("messages", "model"):
            print(f"   * {k}: {v}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
