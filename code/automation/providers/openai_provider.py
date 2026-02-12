"""
OpenAI (GPT) AI provider.

Uses the OpenAI Python SDK for code transformations and prompt evolution.
Requires OPENAI_API_KEY environment variable.
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


class OpenAIProvider(AIProvider):
    """OpenAI GPT API provider."""

    def __init__(
        self,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        defaults = AI_PROVIDER_DEFAULTS["openai"]
        self.model = model or defaults["model"]
        self.max_tokens = max_tokens or defaults["max_tokens"]
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _get_client(self):
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set. "
                    "Set it as an environment variable or pass api_key to the provider."
                )
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package not installed. Install with: pip install openai"
                )
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    @property
    def name(self) -> str:
        return "openai"

    def transform_code(self, original_code: str, prompt: str) -> TransformationResult:
        """Transform code using OpenAI GPT API."""
        try:
            client = self._get_client()

            user_message = (
                f"Apply the following transformation to this Python code:\n\n"
                f"**Transformation:** {prompt}\n\n"
                f"**Original code:**\n```python\n{original_code}\n```"
            )

            response = client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": CODE_TRANSFORM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )

            content = response.choices[0].message.content
            modified_code = extract_code_from_response(content)

            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

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
        """Generate an evolved prompt using GPT."""
        client = self._get_client()

        system_prompt = (
            "You are a prompt engineering expert. You will receive analysis of "
            "adversarial code transformation results and must generate an improved "
            "transformation prompt. Return ONLY the new prompt text, nothing else."
        )

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis_context},
            ],
        )

        return response.choices[0].message.content.strip()
