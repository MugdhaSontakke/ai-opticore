"""Example: run an empirical benchmark with the fake provider."""

from __future__ import annotations

from opticore.benchmarks.runner import BenchmarkRunner
from opticore.core.config import OptimizationConfig
from opticore.core.interfaces import OptimizerRequest
from opticore.providers.base import FakeProviderMixin


class Fake(FakeProviderMixin):
    name = "fake"


def main() -> None:
    samples = [
        OptimizerRequest(
            prompt=(
                "   Summarize the benefits   of   semantic caching for "
                "   large language   model applications.   "
            ),
            system="You are an expert technical writer.",
        ),
        OptimizerRequest(
            prompt="Tell me about context optimization. Context optimization "
                   "is the process of managing context. Tell me about context "
                   "optimization again.",
            system="You are a helpful assistant.",
        ),
        OptimizerRequest(
            prompt="Write a function to compute factorials.",
            system=None,
        ),
    ]

    runner = BenchmarkRunner(
        provider=Fake(),
        model="fake-model",
        samples=samples,
        config=OptimizationConfig(enable_semantic_cache=True),
        repeats=2,
    )
    result = runner.run()
    print(result.render())


if __name__ == "__main__":
    main()