"""
AI Provider registry for code transformation.

Supported providers:
- ollama (default): Local inference, no API key needed
- google: Gemini API, requires GEMINI_API_KEY (free at https://aistudio.google.com)
- anthropic: Claude API, requires ANTHROPIC_API_KEY
- openai: GPT API, requires OPENAI_API_KEY
"""

from typing import Optional

from automation.providers.base import AIProvider, TransformationResult
from automation.providers.ollama_provider import OllamaProvider
from automation.providers.anthropic_provider import AnthropicProvider
from automation.providers.openai_provider import OpenAIProvider
from automation.providers.google_provider import GoogleProvider

_PROVIDERS = {
    "ollama": OllamaProvider,
    "google": GoogleProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}


def get_provider(
    name: str = "ollama",
    model: Optional[str] = None,
    **kwargs,
) -> AIProvider:
    """
    Factory function to get an AI provider instance.

    Args:
        name: Provider name ('ollama', 'google', 'anthropic', 'openai').
        model: Override the default model for the provider.
        **kwargs: Additional keyword arguments passed to the provider constructor.

    Returns:
        An AIProvider instance.

    Raises:
        ValueError: If the provider name is not recognized.
    """
    if name not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider '{name}'. Available: {list(_PROVIDERS.keys())}"
        )
    provider_cls = _PROVIDERS[name]
    if model:
        kwargs["model"] = model
    return provider_cls(**kwargs)


def list_providers() -> list:
    """Return list of available provider names."""
    return list(_PROVIDERS.keys())


__all__ = [
    "AIProvider",
    "TransformationResult",
    "OllamaProvider",
    "GoogleProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "get_provider",
    "list_providers",
]
