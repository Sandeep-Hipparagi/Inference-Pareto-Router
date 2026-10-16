"""Programmatic verifiers for speculative escalation."""

import ast
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Callable


class BaseVerifier(ABC):
    """Abstract base class for output verification."""

    @abstractmethod
    def verify(self, content: str) -> tuple[bool, str | None]:
        """Verify content. Returns (is_valid, optional_error_message)."""
        pass


class JsonSchemaVerifier(BaseVerifier):
    """Verifies that the content contains valid JSON conforming to expected keys/structure."""

    def __init__(self, required_keys: list[str] | None = None) -> None:
        self.required_keys = required_keys or []

    def verify(self, content: str) -> tuple[bool, str | None]:
        # Extract potential JSON block
        json_str = content.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        try:
            parsed = json.loads(json_str)
            if not isinstance(parsed, dict) and not isinstance(parsed, list):
                return False, f"Expected JSON object or array, got {type(parsed).__name__}"

            if isinstance(parsed, dict) and self.required_keys:
                missing = [k for k in self.required_keys if k not in parsed]
                if missing:
                    return False, f"Missing required JSON keys: {missing}"

            return True, None
        except Exception as e:
            return False, f"JSON parsing failed: {e}"


class PythonSyntaxVerifier(BaseVerifier):
    """Verifies that generated Python code parses into a valid Abstract Syntax Tree."""

    def verify(self, content: str) -> tuple[bool, str | None]:
        code = content.strip()
        # Extract code from markdown fences if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```py" in code:
            code = code.split("```py")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Python SyntaxError at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"AST verification failed: {e}"


class RegexVerifier(BaseVerifier):
    """Verifies that the output matches a specified regular expression."""

    def __init__(self, pattern: str | re.Pattern[str]) -> None:
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern

    def verify(self, content: str) -> tuple[bool, str | None]:
        if self.pattern.search(content):
            return True, None
        return False, f"Content failed to match regex pattern: {self.pattern.pattern}"


class CustomVerifier(BaseVerifier):
    """Wraps a custom validation callable."""

    def __init__(self, validator_fn: Callable[[str], bool | tuple[bool, str | None]]) -> None:
        self.validator_fn = validator_fn

    def verify(self, content: str) -> tuple[bool, str | None]:
        try:
            res = self.validator_fn(content)
            if isinstance(res, tuple):
                return res
            return bool(res), None if res else "Custom validation failed"
        except Exception as e:
            return False, f"Custom validator exception: {e}"
