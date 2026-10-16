"""Gateway package providing drop-in client wrappers, speculative cascades, and verifiers."""

from pareto_router.gateway.cascade import CascadeAttempt, CascadeResult, SpeculativeCascade
from pareto_router.gateway.client import ParetoRouter
from pareto_router.gateway.verifiers import (
    BaseVerifier,
    CustomVerifier,
    JsonSchemaVerifier,
    PythonSyntaxVerifier,
    RegexVerifier,
)

__all__ = [
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
