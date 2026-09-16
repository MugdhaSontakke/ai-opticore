"""Tests for token counting and the tokenizer abstraction."""

from __future__ import annotations

import pytest

from opticore.core.interfaces import OptimizerRequest
from opticore.optimizers.token import TokenAnalyzer, Tokenizer, count_tokens


def custom_counter(text: str) -> int:
    return len(text.split())


def test_tokenizer_counts_with_real_tokenizer() -> None:
    tokenizer = Tokenizer()
    count = tokenizer.count("hello world")
    assert count > 0


def test_tokenizer_supports_custom_counter() -> None:
    tokenizer = Tokenizer(count_fn=custom_counter)
    assert tokenizer.count("one two three") == 3
    assert tokenizer.description() == "custom"


def test_tokenizer_empty_text_is_zero() -> None:
    assert Tokenizer().count("") == 0


def test_count_tokens_summarizes_all_fields() -> None:
    tokenizer = Tokenizer(count_fn=custom_counter)
    request = OptimizerRequest(
        prompt="hello world",
        system="system text",
        messages=[{"role": "user", "content": "a b c"}],
    )
    assert count_tokens(request, tokenizer) == 2 + 2 + 3


def test_token_analyzer_reports_composition() -> None:
    analyzer = TokenAnalyzer(Tokenizer(count_fn=custom_counter))
    result = analyzer.analyze("hello world")
    assert result["token_count"] == 2
    assert result["character_count"] == 11
    assert "leading_whitespace_tokens" in result


def test_tiktoken_availability() -> None:
    pytest.importorskip("tiktoken")
    assert Tokenizer().count("hello world") >= 1
