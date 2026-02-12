"""
Anthropic (Claude) AI provider.

Uses the Anthropic Python SDK for code transformations and prompt evolution.
Requires ANTHROPIC_API_KEY environment variable.
"""

import os
from typing import Optional

from automation.config import AI_PROVIDER_DEFAULTS
from automation.providers.base import (
    AIProvider,
    TransformationResult,
    CODE_TRANSFORM_SYSTEM_PROMPT,
    extract_code_from_response,
)


class AnthropicProvider(AIProvider):
    """Anthropic Claude API provider."""

    def __init__(
        self,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        defaults = AI_PROVIDER_DEFAULTS["anthropic"]
        self.model = model or defaults["model"]
        self.max_tokens = max_tokens or defaults["max_tokens"]
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _get_client(self):
        """Lazily initialize the Anthropic client."""
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY not set. "
                    "Set it as an environment variable or pass api_key to the provider."
                )
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Install with: pip install anthropic"
                )
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    @property
    def name(self) -> str:
        return "anthropic"

    def transform_code(self, original_code: str, prompt: str) -> TransformationResult:
        """Transform code using Claude API."""
        try:
            client = self._get_client()

            user_message = (
                f"Apply the following transformation to this Python code:\n\n"
                f"**Transformation:** {prompt}\n\n"
                f"**Original code:**\n```python\n{original_code}\n```"
            )

            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=CODE_TRANSFORM_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            content = response.content[0].text
            modified_code = extract_code_from_response(content)

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            if not modified_code:
                return TransformationResult(
                    original_code=original_code,
                    modified_code="",
                    prompt_used=prompt,
                    provider_name=self.name,
                    model_name=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=False,
                    error="Empty code in response",
                )

            return TransformationResult(
                original_code=original_code,
                modified_code=modified_code,
                prompt_used=prompt,
                provider_name=self.name,
                model_name=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=True,
            )

        except Exception as e:
            return TransformationResult(
                original_code=original_code,
                modified_code="",
                prompt_used=prompt,
                provider_name=self.name,
                model_name=self.model,
                success=False,
                error=str(e),
            )

    def generate_evolved_prompt(self, analysis_context: str) -> str:
        """Generate an evolved prompt using Claude."""
        client = self._get_client()

        system_prompt = (
            "You are a prompt engineering expert. You will receive analysis of "
            "adversarial code transformation results and must generate an improved "
            "transformation prompt. Return ONLY the new prompt text, nothing else."
        )

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": analysis_context}],
        )

        return response.content[0].text.strip()
