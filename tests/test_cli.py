"""Tests for the CLI using subprocess invocation of the entry point."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": str(ROOT / "src")}
    return subprocess.run(
        [PY, "-m", "opticore.cli.main", *args],
        capture_output=True,
        text=True,
        cwd=cwd or ROOT,
        env={**__import__("os").environ, **env},
    )


def test_cli_init_and_config(tmp_path: Path) -> None:
    result = run_cli("init", cwd=tmp_path)
    assert result.returncode == 0
    assert (tmp_path / "opticore.json").exists()
    out = run_cli("config", cwd=tmp_path)
    assert out.returncode == 0
    assert '"safety_mode"' in out.stdout


def test_cli_init_refuses_overwrite(tmp_path: Path) -> None:
    run_cli("init", cwd=tmp_path)
    result = run_cli("init", cwd=tmp_path)
    assert result.returncode != 0
    assert "already exists" in result.stderr


def test_cli_hardware() -> None:
    result = run_cli("hardware")
    assert result.returncode == 0
    assert "Detected backend" in result.stdout


def test_cli_models() -> None:
    result = run_cli("models")
    assert result.returncode == 0
    assert "openai" in result.stdout


def test_cli_optimize_prompt() -> None:
    result = run_cli("optimize", "--prompt", "hello   world")
    assert result.returncode == 0
    assert "hello world" in result.stdout


def test_cli_optimize_json() -> None:
    result = run_cli("optimize", "--prompt", "hello world", "--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["optimized_prompt"] == "hello world"


def test_cli_optimize_missing_prompt() -> None:
    result = run_cli("optimize")
    assert result.returncode != 0


def test_cli_benchmark_in_json_mode() -> None:
    result = run_cli(
        "benchmark", "--samples", "2", "--repeats", "1", "--provider", "fake", "--json"
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "baseline" in payload
    assert "optimized" in payload
    assert "metrics" in payload


def test_cli_benchmark_text_render() -> None:
    result = run_cli("benchmark", "--samples", "1", "--repeats", "1", "--provider", "fake")
    assert result.returncode == 0
    assert "AI-OptiCore Benchmark" in result.stdout


def test_cli_help_for_every_command() -> None:
    for command in ["init", "config", "hardware", "models", "optimize", "benchmark", "cache"]:
        result = run_cli(command, "--help")
        assert result.returncode == 0, f"--help failed for {command}"
        assert "usage" in result.stdout.lower()


def test_cli_unknown_command_fails_gracefully() -> None:
    result = run_cli("frobnicate")
    assert result.returncode != 0
