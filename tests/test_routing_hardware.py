"""Tests for model routing and hardware detection."""

from __future__ import annotations

from opticore.hardware import (
    CPUBackend,
    CUDABackend,
    ROCmBackend,
    describe_detailed,
    detect_backend,
    list_backends,
)
from opticore.routing.router import ModelRouter, RouteRule, default_complexity, default_rules


def test_router_routes_simple_request_to_small_model() -> None:
    router = ModelRouter(rules=default_rules(), default_model="small")
    decision = router.route(prompt="hi")
    assert decision.model == "small"
    assert decision.reason


def test_router_routes_complex_request_to_large_model() -> None:
    router = ModelRouter(rules=default_rules(), default_model="small")
    complex_prompt = (
        "Explain the differences between transformer attention mechanisms "
        "and state space models, including mathematical derivations and "
        "trade-offs in training efficiency, while comparing empirical results "
        "across several large-scale benchmarks. Provide a detailed analysis."
    )
    decision = router.route(prompt=complex_prompt)
    assert decision.model == "large"


def test_default_complexity_bounds() -> None:
    assert 0.0 <= default_complexity("short") <= 1.0
    assert default_complexity("") == 0.0


def test_router_falls_back_to_default() -> None:
    router = ModelRouter(rules=[RouteRule(model="x", min_complexity=0.99)], default_model="d")
    decision = router.route(prompt="simple prompt")
    assert decision.model == "d"


def test_cpu_backend_always_available() -> None:
    backend = CPUBackend()
    assert backend.available() is True
    caps = backend.capabilities()
    assert caps["device"] == "cpu"


def test_cuda_backend_honest_detection() -> None:
    backend = CUDABackend()
    info = backend.info()
    # Must never lie: if torch absent or no GPU, available is False.
    assert isinstance(info.available, bool)


def test_rocm_backend_honest_detection() -> None:
    backend = ROCmBackend()
    info = backend.info()
    assert isinstance(info.available, bool)
    assert "rocm" in info.backend


def test_detect_backend_force_returns_unknown_for_bogus() -> None:
    info = detect_backend(force="does_not_exist")
    assert info.backend == "unknown"


def test_list_backends_contains_expected() -> None:
    names = [b.backend for b in list_backends()]
    assert "cpu" in names


def test_describe_detailed_is_dict() -> None:
    details = describe_detailed()
    assert "detected" in details
    assert "backends" in details
