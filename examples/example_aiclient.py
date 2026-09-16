"""Example: end-to-end generation with a fake provider (no API needed).

Swap the provider for "openai", "ollama", "huggingface" and the example works
against a real model.
"""

from __future__ import annotations

from opticore import AIClient
from opticore.providers.base import FakeProviderMixin


class EchoFake(FakeProviderMixin):
    name = "echo"


def main() -> None:
    client = AIClient(
        provider=EchoFake(),
        optimization=True,
    )

    prompt = "What is   machine learning?   "
    first = client.generate(prompt=prompt)
    second = client.generate(prompt=prompt)  # served from cache

    print("Response:", first.content)
    print("Tokens saved:", first.tokens_saved)
    print("Cache hit on repeat:", second.cache_hit)
    print("Cache stats:", client.cache.stats())
    print("Metrics:", client.metrics.summary()["counters"])


if __name__ == "__main__":
    main()