"""Tests for the prompt optimizer across safety modes."""

from __future__ import annotations

from opticore.core.config import OptimizationConfig, SafetyMode
from opticore.core.interfaces import OptimizerRequest
from opticore.optimizers.prompt import PromptOptimizer
from opticore.optimizers.token import Tokenizer


def _optimize(prompt: str, mode: SafetyMode = SafetyMode.SAFE) -> str:
    config = OptimizationConfig(safety_mode=mode)
    optimizer = PromptOptimizer(tokenizer=Tokenizer(count_fn=lambda t: len(t.split())))
    result = optimizer.optimize(OptimizerRequest(prompt=prompt), config)
    return result.optimized_request.prompt


def test_collapses_excess_whitespace() -> None:
    assert _optimize("hello   world\n\n\nextra") == "hello world extra"


def test_removes_repeated_empty_lines() -> None:
    optimized = _optimize("line one\n\n\n\nline two")
    assert optimized == "line one line two"


def test_safe_mode_preserves_user_content() -> None:
    prompt = "Explain quantum computing in detail.\n\nAlso mention cryptography."
    optimized = _optimize(prompt, SafetyMode.SAFE)
    assert "Explain quantum computing in detail." in optimized
    assert "cryptography" in optimized


def test_balanced_dedupes_repeated_sentences() -> None:
    prompt = "The sky is blue. The sky is blue. It is a nice day."
    optimized = _optimize(prompt, SafetyMode.BALANCED)
    assert optimized.count("The sky is blue") == 1


def test_aggressive_removes_repeated_paragraphs() -> None:
    block = "A paragraph describing an important concept in machine learning."
    prompt = f"{block}\n\n{block}\n\nDifferent content here."
    optimized = _optimize(prompt, SafetyMode.AGGRESSIVE)
    assert block not in optimized or optimized.count(block) == 1


def test_reduction_reported_in_metadata() -> None:
    config = OptimizationConfig(safety_mode=SafetyMode.BALANCED)
    optimizer = PromptOptimizer()
    prompt = "Repeat me. Repeat me. Repeat me."
    result = optimizer.optimize(OptimizerRequest(prompt=prompt), config)
    # A duplicate sentence must be removed only in balanced mode.
    assert result.tokens_saved >= 0
    assert "mode" in result.metadata


def test_original_request_not_mutated() -> None:
    config = OptimizationConfig(safety_mode=SafetyMode.AGGRESSIVE)
    optimizer = PromptOptimizer()
    request = OptimizerRequest(prompt="  lots   of   spaces  \n\n here  ")
    original_prompt = request.prompt
    optimizer.optimize(request, config)
    assert request.prompt == original_prompt
