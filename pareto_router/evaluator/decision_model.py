"""External zero-token decision model client adapter (e.g., Drex/Jev logit endpoints)."""

import os
import time
from typing import Any, Callable
import httpx

from pareto_router.core.schemas import EvaluationScores
from pareto_router.evaluator.base import BaseEvaluator
from pareto_router.evaluator.heuristic import HeuristicEvaluator

RequestFormatter = Callable[[str, list[dict[str, str]] | None], dict[str, Any]]
ResponseParser = Callable[[dict[str, Any]], tuple[float, float]]


class DecisionModelClient(BaseEvaluator):
    """Pluggable adapter for external zero-token categorical logit routing endpoints (Drex, Jev, custom)."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        api_key: str | None = None,
        timeout_ms: float = 12.0,
        engine_type: str = "categorical_logit",
        custom_headers: dict[str, str] | None = None,
        schema_mapping: dict[str, str] | None = None,
        request_formatter: RequestFormatter | None = None,
        response_parser: ResponseParser | None = None,
        fallback_evaluator: BaseEvaluator | None = None,
    ) -> None:
        self.endpoint_url = (
            endpoint_url
            or os.environ.get("PARETO_ROUTER_DECISION_ENDPOINT")
            or "http://localhost:8080/v1/triage"
        )
        self.api_key = api_key or os.environ.get("PARETO_ROUTER_DECISION_KEY")
        self.timeout_sec = timeout_ms / 1000.0
        self.engine_type = engine_type
        self.custom_headers = custom_headers or {}
        self.schema_mapping = schema_mapping or {
            "logic_key": "logic_score",
            "knowledge_key": "knowledge_score",
        }
        self.request_formatter = request_formatter
        self.response_parser = response_parser
        self.fallback = fallback_evaluator or HeuristicEvaluator()

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.custom_headers)
        return headers

    def _build_payload(self, text: str, context: list[dict[str, str]] | None) -> dict[str, Any]:
        if self.request_formatter:
            return self.request_formatter(text, context)
        return {
            "text": text,
            "context": context or [],
            "engine": self.engine_type,
        }

    def _parse_response(self, data: dict[str, Any]) -> tuple[float, float]:
        if self.response_parser:
            return self.response_parser(data)

        # Use schema mapping
        logic_k = self.schema_mapping.get("logic_key", "logic_score")
        know_k = self.schema_mapping.get("knowledge_key", "knowledge_score")

        logic = float(data.get(logic_k, 0.5))
        knowledge = float(data.get(know_k, 0.5))
        return logic, knowledge

    def _compute_scores(
        self,
        text: str,
        context: list[dict[str, str]] | None = None
    ) -> tuple[float, float, dict[str, Any]]:
        """Synchronously query remote logit endpoint with timeout fallback."""
        headers = self._build_headers()
        payload = self._build_payload(text, context)

        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                res = client.post(self.endpoint_url, json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    logic, knowledge = self._parse_response(data)
                    meta = {
                        "remote_endpoint": self.endpoint_url,
                        "engine": f"remote_{self.engine_type}",
                        "status_code": res.status_code,
                    }
                    return logic, knowledge, meta
        except Exception:
            pass

        logic, knowledge, meta = self.fallback._compute_scores(text, context)
        meta["engine"] = f"{self.engine_type}_timeout_fallback"
        return logic, knowledge, meta

    async def evaluate(
        self,
        text: str,
        context: list[dict[str, str]] | None = None
    ) -> EvaluationScores:
        """Asynchronously query remote logit endpoint with non-blocking httpx."""
        t0 = time.perf_counter_ns()
        headers = self._build_headers()
        payload = self._build_payload(text, context)

        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                res = await client.post(self.endpoint_url, json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    t1 = time.perf_counter_ns()
                    latency_ms = (t1 - t0) / 1_000_000.0
                    logic, knowledge = self._parse_response(data)
                    meta = {
                        "remote_endpoint": self.endpoint_url,
                        "engine": f"remote_{self.engine_type}",
                        "status_code": res.status_code,
                    }
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
