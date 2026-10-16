"""Inference-Pareto-Router: 2D Model Tier & Reasoning Effort Routing Engine."""

from pareto_router.core.normalizer import ProviderModelCatalog, ProviderNormalizer
from pareto_router.core.policy import ParetoPolicy
from pareto_router.core.schemas import (
    ChatMessage,
    EffortLevel,
    EvaluationScores,
    InferenceRequest,
    InferenceResponse,
    ModelMode,
    ModelTier,
    ProviderPayload,
    ProviderType,
    RoutingDecision,
)
from pareto_router.evaluator.base import BaseEvaluator
from pareto_router.evaluator.decision_model import DecisionModelClient
from pareto_router.evaluator.heuristic import HeuristicEvaluator
from pareto_router.evaluator.onnx_evaluator import ONNXEvaluator
from pareto_router.gateway.cascade import CascadeAttempt, CascadeResult, SpeculativeCascade
from pareto_router.gateway.client import ParetoRouter
from pareto_router.gateway.verifiers import (
    BaseVerifier,
    CustomVerifier,
    JsonSchemaVerifier,
    PythonSyntaxVerifier,
    RegexVerifier,
)

__version__ = "0.1.0"

__all__ = [
    # Core Schemas
    "ChatMessage",
    "EffortLevel",
    "EvaluationScores",
    "InferenceRequest",
    "InferenceResponse",
    "ModelMode",
    "ModelTier",
    "ProviderPayload",
    "ProviderType",
    "RoutingDecision",
    # Core Logic
    "ParetoPolicy",
    "ProviderModelCatalog",
    "ProviderNormalizer",
    # Evaluators
    "BaseEvaluator",
    "DecisionModelClient",
    "HeuristicEvaluator",
    "ONNXEvaluator",
    # Gateway & Cascades
    "BaseVerifier",
    "CascadeAttempt",
    "CascadeResult",
    "CustomVerifier",
    "JsonSchemaVerifier",
    "ParetoRouter",
    "PythonSyntaxVerifier",
    "RegexVerifier",
    "SpeculativeCascade",
]
