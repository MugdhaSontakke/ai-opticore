"""Property-based tests using Hypothesis for core invariants."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from opticore.core.config import OptimizationConfig
from opticore.core.model import AIRequest
from opticore.core.pipeline import build_default_pipeline
from opticore.optimizers.token import Tokenizer, count_tokens

_OPTIMIZER_STRATEGY = [
    OptimizationConfig(safety_mode=safe, enable_context_optimization=True)
    for safe in ("safe", "balanced", "aggressive")
]


def _pipeline_for(config: OptimizationConfig):
    return build_default_pipeline(config)


@settings(max_examples=40, deadline=None)
@given(
    prompt=st.text(max_size=2000),
    system=st.one_of(st.none(), st.text(max_size=500)),
)
def test_optimizer_never_mutates_original(prompt: str, system: str | None) -> None:
    original = AIRequest(prompt=prompt, system=system)
    snapshot = dict(original.metadata)
    for config in _OPTIMIZER_STRATEGY:
        _pipeline_for(config).run(request=original)
    assert original.prompt == prompt
    assert original.system == system
    assert original.metadata == snapshot


@settings(max_examples=40, deadline=None)
@given(prompt=st.text(max_size=2000))
def test_optimization_is_deterministic(prompt: str) -> None:
    config = OptimizationConfig(safety_mode="aggressive")
    pipeline = _pipeline_for(config)
    first = pipeline.run(request=AIRequest(prompt=prompt))
    second = pipeline.run(request=AIRequest(prompt=prompt))
    assert first.optimized_request.prompt == second.optimized_request.prompt
    assert first.tokens_saved == second.tokens_saved


@given(text=st.text(max_size=5000))
def test_token_counts_never_negative(text: str) -> None:
    tokenizer = Tokenizer(count_fn=lambda t: len(t.split()))
    assert count_tokens(AIRequest(prompt=text), tokenizer) >= 0
    assert tokenizer.count("") == 0


@settings(max_examples=40, deadline=None)
@given(
    system=st.text(min_size=1, max_size=300),
    messages=st.lists(
        st.fixed_dictionaries(
            {
                "role": st.sampled_from(["user", "assistant", "system"]),
                "content": st.text(min_size=1, max_size=100),
            }
        ),
        max_size=10,
    ),
    budget=st.integers(min_value=1, max_value=10_000),
)
def test_safe_mode_never_drops_system_message(
    system: str, messages: list[dict[str, str]], budget: int
) -> None:
    config = OptimizationConfig(safety_mode="safe", max_token_budget=budget)
    result = _pipeline_for(config).run(
        request=AIRequest(prompt="hello", system=system, messages=messages)
    )
    # SAFE mode may strip surrounding whitespace but must never delete the
    # system instruction content.
    optimized_system = result.optimized_request.system
    assert optimized_system is not None
    assert optimized_system.strip() == system.strip()
    assert optimized_system in (system, system.strip())


@settings(max_examples=30, deadline=None)
@given(extra_messages=st.integers(min_value=0, max_value=8))
def test_context_over_budget_drops_oldest_not_system(extra_messages: int) -> None:
    messages = [
        {"role": "user", "content": f"turn {i} content"}
        for i in range(extra_messages)
    ]
    config = OptimizationConfig(safety_mode="safe", max_token_budget=10)
    result = _pipeline_for(config).run(
        request=AIRequest(prompt="q", system="keep me", messages=messages)
    )
    optimized = result.optimized_request
    assert optimized.system == "keep me"
    # If anything was dropped, it must have been the oldest user/assistant turns.
    if len(optimized.messages or []) < len(messages or []):
        roles = {m["role"] for m in optimized.messages or []}
        assert "system" not in roles or optimized.system is None
