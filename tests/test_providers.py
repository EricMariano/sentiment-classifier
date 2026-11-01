"""Escolha do modelo de resumo e o caminho pela OpenRouter, sem rede."""

from typing import Any

import pytest

from sentiment_classifier import providers
from sentiment_classifier.summarize import is_transient


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_init(spec: str, **kwargs: Any) -> str:
        calls.append((spec, kwargs))
        return "modelo"

    import langchain.chat_models

    monkeypatch.setattr(langchain.chat_models, "init_chat_model", fake_init)
    return calls


def test_openrouter_only_uses_providers_that_do_not_keep_data(
    captured: list[tuple[str, dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CONVERSA_OPENROUTER_PRIVADO", raising=False)
    providers.chat_model("openrouter:openai/gpt-5-nano")
    providers.chat_model("openai:gpt-5-nano")
    assert captured == [
        ("openrouter:openai/gpt-5-nano", {"openrouter_provider": {"data_collection": "deny"}}),
        ("openai:gpt-5-nano", {}),
    ]


def test_privacy_restriction_can_be_turned_off(
    captured: list[tuple[str, dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONVERSA_OPENROUTER_PRIVADO", "0")
    providers.chat_model("openrouter:qwen/qwen3-32b")
    assert captured == [("openrouter:qwen/qwen3-32b", {})]


def test_no_summarizer_without_the_providers_key(
    captured: list[tuple[str, dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONVERSA_CHAT_MODEL", "openrouter:openai/gpt-5-nano")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert providers.make_summarizer() is None  # sem chave: template, sem erro na tela
    monkeypatch.setenv("OPENROUTER_API_KEY", "teste")
    assert providers.make_summarizer() is not None
    assert providers.make_summarizer(offline=True) is None


def _named(name: str, message: str = "") -> Exception:
    return type(name, (Exception,), {})(message)


def test_openrouter_errors_by_name() -> None:
    assert is_transient(_named("TooManyRequestsResponseError", "slow down"))
    assert is_transient(_named("ProviderOverloadedResponseError"))
    assert not is_transient(_named("PaymentRequiredResponseError", "insufficient credits"))
    assert not is_transient(_named("UnauthorizedResponseError"))
    assert not is_transient(_named("TooManyRequestsResponseError", "quota exceeded"))
