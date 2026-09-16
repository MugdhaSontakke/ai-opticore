"""Provider registration and factory."""

from __future__ import annotations

from typing import Any

from opticore.core.config import ProviderConfig
from opticore.providers.base import (
    BaseProvider,
    ProviderError,
)
from opticore.providers.base import (
    ProviderResponse as ProviderResponse,
)

_REGISTRY: dict[str, type[BaseProvider]] = {}


def register_provider(name: str, klass: type[BaseProvider]) -> None:
    """Register a provider class under a canonical name."""
    _REGISTRY[name] = klass


def get_provider(
    name: str,
    config: ProviderConfig | None = None,
    **kwargs: Any,
) -> BaseProvider:
    """Instantiate a provider by name, optionally with overrides."""
    config = config or ProviderConfig(provider=name)
    for key in ("model", "base_url"):
        if key in kwargs:
            setattr(config, key, kwargs.pop(key))
    if name in _REGISTRY:
        return _REGISTRY[name](config=config, **kwargs)
    raise ProviderError(
        f"Unknown provider '{name}'. Available: {sorted(_REGISTRY)}"
    )


def available_providers() -> list[str]:
    """Return registered provider names."""
    return sorted(_REGISTRY)


def _init_registry() -> None:
    from opticore.providers.huggingface import HuggingFaceProvider
    from opticore.providers.ollama import OllamaProvider
    from opticore.providers.openai import OpenAIProvider

    register_provider("openai", OpenAIProvider)
    register_provider("ollama", OllamaProvider)
    register_provider("huggingface", HuggingFaceProvider)


_init_registry()
