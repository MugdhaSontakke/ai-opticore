"""Mock-based tests for the Ollama health probe and CLI doctor (no real server).

The ``OPTICORE_TEST_LIVE=1`` gate in ``tests/test_providers_live.py`` covers the
real-endpoint regression path; here every network response is faked so CI
never needs Ollama.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from opticore.providers.ollama import OllamaProvider


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(
        self,
        payload: dict[str, Any],
        status: int = 200,
        error: bool = False,
    ) -> None:
        self._payload = payload
        self.status_code = status
        self._error = error

    def raise_for_status(self) -> None:
        if self._error:
            import requests

            raise requests.exceptions.HTTPError(
                f"HTTP {self.status_code}", response=self
            )

    def json(self) -> Any:
        return self._payload

    @property
    def text(self) -> str:
        return json.dumps(self._payload)


class FakeOllamaServer:
    """Scripted fake of the Ollama REST API."""

    def __init__(
        self,
        *,
        version: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        generate: dict[str, Any] | None = None,
        error_kind: str | None = None,
    ) -> None:
        self.version = version or {"version": "0.6.0"}
        self.tags = tags or {"models": [{"model": "llama3.2:latest"}]}
        self.generate = generate or {"response": "ok", "eval_count": 3}
        self.error_kind = error_kind

    def route(self, path: str, payload: dict[str, Any] | None) -> FakeResponse:
        if self.error_kind == "timeout":
            import requests

            raise requests.exceptions.Timeout("timed out")
        if self.error_kind == "connection":
            import requests

            raise requests.exceptions.ConnectionError("refused")
        if self.error_kind == "malformed":
            return FakeResponse({"not": "an object"}, status=200)
        if path == "/api/version":
            return FakeResponse(self.version)
        if path == "/api/tags":
            return FakeResponse(self.tags)
        if self.error_kind == "http_error":
            return FakeResponse({"error": "model not found"}, status=404, error=True)
        return FakeResponse(self.generate)


@pytest.fixture
def fake_server() -> FakeOllamaServer:
    return FakeOllamaServer()


def _patch_server(
    monkeypatch: pytest.MonkeyPatch, server: FakeOllamaServer
) -> OllamaProvider:
    import requests

    def get(url: str, **kwargs: Any) -> FakeResponse:
        return server.route("/api/version" if url.endswith("/api/version") else "/api/tags", None)

    def post(url: str, json: dict[str, Any], **kwargs: Any) -> FakeResponse:
        return server.route("/api/generate", json)

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", post)
    return OllamaProvider(base_url="http://127.0.0.1:9999")


class TestHealth:
    def test_healthy_when_reachable_with_probe(
        self, monkeypatch: pytest.MonkeyPatch, fake_server: FakeOllamaServer
    ) -> None:
        provider = _patch_server(monkeypatch, fake_server)
        report = provider.health(model="llama3.2", probe=True)
        assert report["healthy"] is True
        assert report["reachable"] is True
        assert report["ollama_version"] == "0.6.0"
        assert report["model_installed"] is True
        probe = report["probe"] or {}
        assert probe["ok"] is True
        assert probe["output_tokens"] == 3

    def test_shorthand_model_matches_tag(
        self, monkeypatch: pytest.MonkeyPatch, fake_server: FakeOllamaServer
    ) -> None:
        provider = _patch_server(monkeypatch, fake_server)
        report = provider.health(model="llama3.2")
        assert report["model_installed"] is True

    def test_not_running_is_typed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _patch_server(
            monkeypatch, FakeOllamaServer(error_kind="connection")
        )
        report = provider.health(model="llama3.2")
        assert report["healthy"] is False
        assert report["reachable"] is False
        assert report["error_kind"] == "not_running"

    def test_timeout_is_typed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _patch_server(
            monkeypatch, FakeOllamaServer(error_kind="timeout")
        )
        report = provider.health(model="llama3.2")
        assert report["error_kind"] == "timeout"
        assert report["reachable"] is False

    def test_model_not_installed_is_distinct(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _patch_server(
            monkeypatch,
            FakeOllamaServer(tags={"models": [{"model": "mistral"}]}),
        )
        report = provider.health(model="llama3.2")
        assert report["reachable"] is True
        assert report["model_installed"] is False
        assert report["error_kind"] == "model_not_found"
        assert "ollama pull llama3.2" in report["error"]

    def test_http_error_on_probe_is_typed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _patch_server(
            monkeypatch, FakeOllamaServer(error_kind="http_error")
        )
        report = provider.health(model="llama3.2", probe=True)
        assert report["probe"]["ok"] is False
        assert report["error_kind"] == "http:404"

    def test_health_never_raises_for_offline_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _patch_server(
            monkeypatch, FakeOllamaServer(error_kind="connection")
        )
        # health() must never raise; callers inspect the structured report.
        report = provider.health(probe=True)
        assert report["error_kind"] == "not_running"


class TestModels:
    def test_models_from_tags_when_reachable(
        self, monkeypatch: pytest.MonkeyPatch, fake_server: FakeOllamaServer
    ) -> None:
        provider = _patch_server(monkeypatch, fake_server)
        assert provider.models() == ["llama3.2:latest"]

    def test_models_empty_when_offline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _patch_server(
            monkeypatch, FakeOllamaServer(error_kind="connection")
        )
        assert provider.models() == []


class TestDoctorCli:
    @staticmethod
    def _exit_code(exc_info: pytest.ExceptionInfo[SystemExit]) -> int:
        code = exc_info.value.code
        assert isinstance(code, int)
        return code

    def test_doctor_reports_offline_ollama(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        from opticore.cli.main import main

        server = FakeOllamaServer(error_kind="connection")
        import requests

        def get(url: str, **kwargs: Any) -> FakeResponse:
            return server.route("/api/version", None)

        monkeypatch.setattr(requests, "get", get)

        with pytest.raises(SystemExit) as exc_info:
            main(["doctor", "--provider", "ollama", "--no-probe"])
        assert self._exit_code(exc_info) == 1
        captured = capsys.readouterr()
        assert "not_running" in captured.err or "FAIL" in captured.out

    def test_doctor_ok_when_health_healthy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_server: FakeOllamaServer,
    ) -> None:
        from opticore.cli.main import main

        provider = _patch_server(monkeypatch, fake_server)
        monkeypatch.setattr(
            "opticore.providers.get_provider",
            lambda *_a, **_k: provider,
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["doctor", "--provider", "ollama", "--no-probe"])
        assert self._exit_code(exc_info) == 0

    def test_python_dash_m_optcore_runs(self) -> None:
        from opticore.cli.main import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert self._exit_code(exc_info) == 0


class TestGenerateErrors:
    def test_generate_with_unknown_model_surfaces_typed_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        import requests

        from opticore.exceptions import ProviderError

        # Force the HTTP path (require requests, no ollama SDK).
        monkeypatch.setitem(sys.modules, "ollama", None)
        server = FakeOllamaServer(error_kind="http_error")

        def post(url: str, json: dict[str, Any], **kwargs: Any) -> FakeResponse:
            return server.route("/api/chat" if url.endswith("/api/chat") else "/api/generate", json)

        monkeypatch.setattr(requests, "post", post)
        provider = OllamaProvider(base_url="http://127.0.0.1:9999")
        with pytest.raises(ProviderError):
            provider.generate({"prompt": "hi"})
