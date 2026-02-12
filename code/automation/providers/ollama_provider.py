"""
Ollama AI provider for local inference.

Uses the Ollama REST API (http://localhost:11434) for free, local
code transformations and prompt evolution. No API key needed.
"""

import requests
from typing import Optional

from automation.config import AI_PROVIDER_DEFAULTS
from automation.providers.base import (
    AIProvider,
    TransformationResult,
    CODE_TRANSFORM_SYSTEM_PROMPT,
    extract_code_from_response,
)


class OllamaProvider(AIProvider):
    """Ollama local inference provider."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        defaults = AI_PROVIDER_DEFAULTS["ollama"]
        self.model = model or defaults["model"]
        self.base_url = (base_url or defaults["base_url"]).rstrip("/")

    @property
    def name(self) -> str:
        return "ollama"

    def _check_availability(self) -> None:
        """Check if Ollama is running and the model is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Start it with: ollama serve"
            )
        except requests.RequestException as e:
            raise ConnectionError(f"Ollama health check failed: {e}")

        models = [m["name"] for m in resp.json().get("models", [])]
        # Ollama model names may include :tag - match with or without tag
        model_base = self.model.split(":")[0]
        found = any(
            m == self.model or m.startswith(f"{model_base}:")
            for m in models
        )
        if not found:
            raise RuntimeError(
                f"Model '{self.model}' not found in Ollama. "
                f"Available models: {models}. "
                f"Pull it with: ollama pull {self.model}"
            )

    def _chat(self, system_prompt: str, user_message: str) -> dict:
        """Send a chat request to Ollama and return the response."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=300,  # 5 min timeout for large files
        )
        resp.raise_for_status()
        return resp.json()

    def transform_code(self, original_code: str, prompt: str) -> TransformationResult:
        """Transform code using Ollama local inference."""
        try:
            self._check_availability()

            user_message = (
                f"Apply the following transformation to this Python code:\n\n"
                f"**Transformation:** {prompt}\n\n"
                f"**Original code:**\n```python\n{original_code}\n```"
            )

            response = self._chat(CODE_TRANSFORM_SYSTEM_PROMPT, user_message)
            content = response.get("message", {}).get("content", "")
            modified_code = extract_code_from_response(content)

            # Ollama provides token counts in eval_count / prompt_eval_count
            input_tokens = response.get("prompt_eval_count", 0)
            output_tokens = response.get("eval_count", 0)

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

        except (ConnectionError, RuntimeError) as e:
            return TransformationResult(
                original_code=original_code,
                modified_code="",
                prompt_used=prompt,
                provider_name=self.name,
                model_name=self.model,
                success=False,
                error=str(e),
            )
        except Exception as e:
            return TransformationResult(
                original_code=original_code,
                modified_code="",
                prompt_used=prompt,
                provider_name=self.name,
                model_name=self.model,
                success=False,
                error=f"Unexpected error: {e}",
            )

    def generate_evolved_prompt(self, analysis_context: str) -> str:
        """Generate an evolved prompt using Ollama."""
        system_prompt = (
            "You are a prompt engineering expert. You will receive analysis of "
            "adversarial code transformation results and must generate an improved "
            "transformation prompt. Return ONLY the new prompt text, nothing else."
        )

        self._check_availability()
        response = self._chat(system_prompt, analysis_context)
        return response.get("message", {}).get("content", "").strip()
