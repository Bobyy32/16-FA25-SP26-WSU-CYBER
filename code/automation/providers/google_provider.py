"""
Google Gemini AI provider.

Uses the Google GenAI Python SDK for code transformations and prompt evolution.
Requires GEMINI_API_KEY environment variable (free from https://aistudio.google.com).

Available free models (set via model parameter):
    "gemini-2.5-pro"         - Best quality, 100 requests/day
    "gemini-2.5-flash"       - Fast + capable, 250 requests/day
    "gemini-2.5-flash-lite"  - Fastest, 1000 requests/day
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


class GoogleProvider(AIProvider):
    """Google Gemini API provider."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        defaults = AI_PROVIDER_DEFAULTS["google"]
        self.model = model or defaults["model"]
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client = None

    def _get_client(self):
        """Lazily initialize the Google GenAI client."""
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY not set. "
                    "Get a free key at https://aistudio.google.com and set it "
                    "as an environment variable or pass api_key to the provider."
                )
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "google-genai package not installed. Install with: pip install google-genai"
                )
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    @property
    def name(self) -> str:
        return "google"

    def transform_code(self, original_code: str, prompt: str) -> TransformationResult:
        """Transform code using Gemini API."""
        try:
            client = self._get_client()
            from google.genai import types

            user_message = (
                f"Apply the following transformation to this Python code:\n\n"
                f"**Transformation:** {prompt}\n\n"
                f"**Original code:**\n```python\n{original_code}\n```"
            )

            response = client.models.generate_content(
                model=self.model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=CODE_TRANSFORM_SYSTEM_PROMPT,
                ),
            )

            content = response.text
            modified_code = extract_code_from_response(content)

            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

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
        """Generate an evolved prompt using Gemini."""
        client = self._get_client()
        from google.genai import types

        system_prompt = (
            "You are a prompt engineering expert. You will receive analysis of "
            "adversarial code transformation results and must generate an improved "
            "transformation prompt. Return ONLY the new prompt text, nothing else."
        )

        response = client.models.generate_content(
            model=self.model,
            contents=analysis_context,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            ),
        )

        return response.text.strip()
