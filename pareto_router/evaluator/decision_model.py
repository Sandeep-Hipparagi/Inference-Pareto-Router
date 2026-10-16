"""External zero-token decision model client adapter (e.g., Zev/Jev logit endpoints)."""

import time
from typing import Any
import httpx

from pareto_router.evaluator.base import BaseEvaluator
from pareto_router.evaluator.heuristic import HeuristicEvaluator


class DecisionModelClient(BaseEvaluator):
    """Adapter for remote zero-token categorical logit routing endpoints."""

    def __init__(
        self,
        endpoint_url: str = "http://localhost:8080/v1/triage",
        api_key: str | None = None,
        timeout_ms: float = 12.0,
        fallback_evaluator: BaseEvaluator | None = None,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout_sec = timeout_ms / 1000.0
        self.fallback = fallback_evaluator or HeuristicEvaluator()

    def _compute_scores(
        self,
        text: str,
        context: list[dict[str, str]] | None = None
    ) -> tuple[float, float, dict[str, Any]]:
        """Synchronously query remote logit endpoint with timeout fallback."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "text": text,
            "context": context or [],
        }

        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                res = client.post(self.endpoint_url, json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    logic = float(data.get("logic_score", 0.5))
                    knowledge = float(data.get("knowledge_score", 0.5))
                    meta = {
                        "remote_endpoint": self.endpoint_url,
                        "engine": "remote_zero_token_decision_model",
                        "status_code": res.status_code,
                    }
                    return logic, knowledge, meta
        except Exception as e:
            # Automatic fallback to local heuristic
            pass

        logic, knowledge, meta = self.fallback._compute_scores(text, context)
        meta["engine"] = "remote_timeout_fallback"
        return logic, knowledge, meta

    async def evaluate(
        self,
        text: str,
        context: list[dict[str, str]] | None = None
    ) -> Any:
        """Asynchronously query remote logit endpoint with non-blocking httpx."""
        t0 = time.perf_counter_ns()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "text": text,
            "context": context or [],
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                res = await client.post(self.endpoint_url, json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    t1 = time.perf_counter_ns()
                    latency_ms = (t1 - t0) / 1_000_000.0
                    logic = float(data.get("logic_score", 0.5))
                    knowledge = float(data.get("knowledge_score", 0.5))
                    meta = {
                        "remote_endpoint": self.endpoint_url,
                        "engine": "remote_zero_token_decision_model",
                        "status_code": res.status_code,
                    }
                    from pareto_router.core.schemas import EvaluationScores
                    return EvaluationScores(
                        logic_score=max(0.0, min(1.0, logic)),
                        knowledge_score=max(0.0, min(1.0, knowledge)),
                        latency_ms=round(latency_ms, 3),
                        evaluator_name=self.__class__.__name__,
                        metadata=meta,
                    )
        except Exception:
            pass

        # Fallback to local synchronous evaluator
        return self.fallback.evaluate_sync(text, context)
