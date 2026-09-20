"""Tests for benchmark honesty: no fabricated numbers, wired dataset loader."""

from __future__ import annotations

import json

import pytest

from opticore.benchmarks.runner import (
    BenchmarkRunner,
    load_benchmark_dataset,
)
from opticore.core.config import OptimizationConfig
from opticore.providers.base import FakeProviderMixin

FIXTURES = "benchmarks/data/fixtures.json"


class Fake(FakeProviderMixin):
    name = "fake"


class TestLoadBenchmarkDataset:
    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_benchmark_dataset("does-not-exist.json")

    def test_malformed_root_raises(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"not": "a list"}')
        with pytest.raises(ValueError):
            load_benchmark_dataset(str(path))

    def test_entry_without_prompt_raises(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('[{"system": "no prompt here"}]')
        with pytest.raises(ValueError, match="prompt"):
            load_benchmark_dataset(str(path))

    def test_fixtures_load_with_category_scenario(self) -> None:
        samples = load_benchmark_dataset(FIXTURES)
        assert len(samples) >= 6
        # category is mapped to scenario and expected_keywords preserved.
        first = samples[0]
        assert first.metadata["scenario"] == "simple_qa"
        assert first.metadata["expected_keywords"]
        assert first.metadata["id"] == "simple_qa"
        # The sample using tools keeps its tools intact.
        tool_caller = next(s for s in samples if s.metadata["id"] == "tool_calling")
        assert tool_caller.tools


class TestRunnerHonesty:
    def test_fake_provider_is_declared(self) -> None:
        runner = BenchmarkRunner(
            provider=Fake(),
            model="fake-model",
            samples=[load_benchmark_dataset(FIXTURES)[0]],
            config=OptimizationConfig(),
            repeats=1,
        )
        result = runner.run()
        assert "fake" in result.provider
        assert any("latency is not representative" in n for n in result.notes)

    def test_missing_pricing_is_declared(self) -> None:
        runner = BenchmarkRunner(
            provider=Fake(),
            model="fake-model",
            samples=[load_benchmark_dataset(FIXTURES)[0]],
            config=OptimizationConfig(pricing_input_per_1k=0, pricing_output_per_1k=0),
            repeats=1,
        )
        result = runner.run()
        assert any("No pricing configured" in n for n in result.notes)
        assert result.metrics["estimated_cost_baseline"] == "N/A"

    def test_environment_is_recorded(self) -> None:
        runner = BenchmarkRunner(
            provider=Fake(),
            model="fake-model",
            samples=[load_benchmark_dataset(FIXTURES)[0]],
            repeats=1,
        )
        result = runner.run()
        assert result.environment["opticore_version"]
        assert result.environment["timestamp"]
        assert result.environment["tokenizer"]

    def test_scenario_tags_flow_into_report(self) -> None:
        samples = load_benchmark_dataset(FIXTURES)
        runner = BenchmarkRunner(
            provider=Fake(),
            model="fake-model",
            samples=samples,
            repeats=1,
            config=OptimizationConfig(),
        )
        result = runner.run()
        assert result.dataset == "builtin"
        scenarios = result.metrics.get("scenarios") or {}
        assert "simple_qa" in scenarios


class TestCliDatasetFlag:
    def test_cli_benchmark_with_fixtures(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "opticore.cli.main",
                "benchmark",
                "--provider",
                "fake",
                "--dataset",
                str(root / FIXTURES),
                "--json",
            ],
            capture_output=True,
            text=True,
            cwd=root,
            env={"PYTHONPATH": str(root / "src")},
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["dataset"] == "fixtures.json"
        assert "metrics" in payload
