"""
Base classes for AI providers in the Adversarial Stylometry system.

Defines the abstract interface that all providers must implement,
plus the TransformationResult dataclass for standardized results.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TransformationResult:
    """Result of an AI code transformation."""
    original_code: str
    modified_code: str
    prompt_used: str
    provider_name: str
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    success: bool = True
    error: Optional[str] = None


# System prompt used by all providers to enforce output format
CODE_TRANSFORM_SYSTEM_PROMPT = (
    "You are a code transformation assistant. You will be given Python source code "
    "and a transformation instruction. Apply the transformation and return ONLY the "
    "complete modified Python source code wrapped in ```python fences. "
    "Do not include any explanation, commentary, or additional text outside the code fences. "
    "The code must be valid, runnable Python that preserves the original functionality."
)


def extract_code_from_response(response_text: str) -> str:
    """
    Extract Python code from an AI response.

    Tries to find code in ```python fences first, then ``` fences,
    then falls back to the raw response.

    Args:
        response_text: Raw text response from the AI provider.

    Returns:
        Extracted code string.
    """
    # Try ```python ... ``` first
    match = re.search(r'```python\s*\n(.*?)```', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic ``` ... ```
    match = re.search(r'```\s*\n(.*?)```', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: return stripped response (might be bare code)
    return response_text.strip()


class AIProvider(ABC):
    """Abstract base class for AI providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short provider name (e.g. 'ollama', 'anthropic', 'openai')."""
        ...

    @abstractmethod
    def transform_code(self, original_code: str, prompt: str) -> TransformationResult:
        """
        Transform code according to the given prompt.

        Args:
            original_code: The original Python source code.
            prompt: The transformation instruction.

        Returns:
            TransformationResult with the modified code.
        """
        ...

    @abstractmethod
    def generate_evolved_prompt(self, analysis_context: str) -> str:
        """
        Generate an evolved/improved prompt based on analysis of previous results.

        Args:
            analysis_context: Formatted string with previous results and analysis.

        Returns:
            The new evolved prompt string.
        """
        ...
