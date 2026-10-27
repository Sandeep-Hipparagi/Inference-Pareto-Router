"""Turn a routing decision into a real provider call.

    python examples/03_dispatch_payload.py            # dry run, prints the request
    python examples/03_dispatch_payload.py --live     # actually calls the provider

This is the supported production path in 0.1.x: the router decides, and you
dispatch the payload with the client of your choice.
"""

import json
import os
import sys

import httpx

from pareto_router import HeuristicEvaluator, ParetoPolicy, ProviderNormalizer, ProviderType

PROMPT = "Write a SQL query that finds the second-highest salary per department."
ENDPOINTS = {
    ProviderType.OPENAI: "https://api.openai.com/v1/chat/completions",
    ProviderType.ANTHROPIC: "https://api.anthropic.com/v1/messages",
    ProviderType.GEMINI: "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
}

scores = HeuristicEvaluator().evaluate_sync(PROMPT)
decision = ParetoPolicy().evaluate_coordinates(scores)
payload = ProviderNormalizer().to_provider_payload(
    decision, ProviderType.OPENAI, [{"role": "user", "content": PROMPT}]
)

url = ENDPOINTS[payload.provider]
headers = {"Content-Type": "application/json", **payload.headers}

print(f"routed to : {decision.tier.value} / {decision.effort_level.value}")
print(f"model     : {payload.model}")
print(f"POST      : {url}")
print("body      :", json.dumps(payload.body, indent=2)[:400])

if "--live" not in sys.argv:
    print("\n(dry run - pass --live and set OPENAI_API_KEY to actually dispatch)")
    raise SystemExit(0)

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("OPENAI_API_KEY is not set")

response = httpx.post(url, headers={**headers, "Authorization": f"Bearer {api_key}"},
                      json=payload.body, timeout=120.0)
response.raise_for_status()
print("\nresponse:", response.json()["choices"][0]["message"]["content"][:400])
