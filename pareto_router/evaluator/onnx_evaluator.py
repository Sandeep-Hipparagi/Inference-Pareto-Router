"""ONNX Runtime dual-axis prompt scoring engine with zero-weight heuristic fallback."""

import os
from pathlib import Path
from typing import Any
import numpy as np

from pareto_router.evaluator.base import BaseEvaluator
from pareto_router.evaluator.heuristic import HeuristicEvaluator


class ONNXEvaluator(BaseEvaluator):
    """Local ONNX Runtime evaluator for sub-15ms dual-axis prompt classification."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        fallback_evaluator: BaseEvaluator | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.fallback = fallback_evaluator or HeuristicEvaluator()
        self.session: Any = None
        self.is_onnx_loaded = False

        if self.model_path and self.model_path.exists():
            self._init_onnx_session()

    def _init_onnx_session(self) -> None:
        """Initialize onnxruntime InferenceSession with CPU optimizations."""
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.is_onnx_loaded = True
        except Exception:
            # Fall back gracefully if ONNX file is corrupted or unsupported
            self.session = None
            self.is_onnx_loaded = False

    def _simple_tokenize(self, text: str, max_length: int = 128) -> np.ndarray:
        """Fast character/subword token encoder for ONNX input vectors."""
        # Fast ASCII token hashing for lightweight classifier graphs
        tokens = [min(255, ord(c)) for c in text[:max_length]]
        if len(tokens) < max_length:
            tokens.extend([0] * (max_length - len(tokens)))
        return np.array([tokens], dtype=np.int64)

    def _compute_scores(
        self,
        text: str,
        context: list[dict[str, str]] | None = None
    ) -> tuple[float, float, dict[str, Any]]:
        if not self.is_onnx_loaded or self.session is None:
            # Seamless fallback to sub-millisecond heuristic
            logic, knowledge, meta = self.fallback._compute_scores(text, context)
            meta["onnx_active"] = False
            meta["engine"] = "heuristic_fallback"
            return logic, knowledge, meta

        try:
            # Run ONNX inference
            input_name = self.session.get_inputs()[0].name
            input_data = self._simple_tokenize(text)
            outputs = self.session.run(None, {input_name: input_data})

            logits = outputs[0][0]
            # Sigmoid activation to [0, 1]
            probs = 1.0 / (1.0 + np.exp(-logits))
            logic_score = float(probs[0])
            knowledge_score = float(probs[1]) if len(probs) > 1 else 0.5

            meta = {
                "onnx_active": True,
                "engine": "onnx_runtime",
                "model_path": str(self.model_path),
            }
            return round(logic_score, 3), round(knowledge_score, 3), meta
        except Exception as e:
            # Fall back if runtime inference encounters unexpected error
            logic, knowledge, meta = self.fallback._compute_scores(text, context)
            meta["onnx_active"] = False
            meta["engine"] = f"heuristic_error_fallback: {e}"
            return logic, knowledge, meta
