"""Escolhe classificador, resumidor e nota de satisfação a partir do ambiente."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from sentiment_classifier.cache import ResponseCache
from sentiment_classifier.classify import (
    JevTurnClassifier,
    OfflineHeuristicClassifier,
    TurnClassifier,
)
from sentiment_classifier.csat import CsatEstimator, CsatModel, JevCsatEstimator
from sentiment_classifier.summarize import LLMSummarizer

DEFAULT_CSAT_MODEL = Path("csat_model.json")
DEFAULT_CACHE = Path.home() / ".cache" / "sentiment_classifier" / "jev.sqlite"
_cache: ResponseCache | None = None


def response_cache() -> ResponseCache | None:
    """Cache das respostas do Jev (só hash e probabilidades, sem texto).

    ``CONVERSA_CACHE=0`` desliga; ``CONVERSA_CACHE=/caminho.sqlite`` muda o arquivo.
    """
    global _cache
    setting = os.environ.get("CONVERSA_CACHE", "")
    if setting == "0":
        return None
    if _cache is None:
        _cache = ResponseCache(Path(setting) if setting else DEFAULT_CACHE)
    return _cache


def has_jev() -> bool:
    return bool(os.environ.get("TYPESAFE_API_KEY"))


def make_classifier(offline: bool = False) -> TurnClassifier:
    if offline or not has_jev():
        return OfflineHeuristicClassifier()
    return JevTurnClassifier(cache=response_cache())


OPENROUTER_PREFIX = "openrouter:"
_KEY_FOR = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def has_key_for(spec: str) -> bool:
    """O provedor do modelo tem chave no ambiente? (Provedores desconhecidos: assume que sim.)"""
    var = _KEY_FOR.get(spec.split(":", 1)[0])
    return var is None or bool(os.environ.get(var))


def chat_model(spec: str, **options: Any) -> BaseChatModel:
    """Modelo de chat a partir de ``provedor:modelo``, no formato do ``init_chat_model``.

    Com ``openrouter:`` (por exemplo ``openrouter:openai/gpt-5-nano``), a OpenRouter só
    encaminha para provedores que não guardam nem treinam com os dados
    (``data_collection: deny``). ``CONVERSA_OPENROUTER_PRIVADO=0`` desliga, se nenhum
    provedor do modelo escolhido aceitar a restrição.
    """
    from langchain.chat_models import init_chat_model

    if spec.startswith(OPENROUTER_PREFIX) and os.environ.get("CONVERSA_OPENROUTER_PRIVADO") != "0":
        options.setdefault("openrouter_provider", {"data_collection": "deny"})
    model: BaseChatModel = init_chat_model(spec, **options)
    return model


def make_summarizer(offline: bool = False) -> LLMSummarizer | None:
    """``None`` sem modelo configurado ou sem a chave do provedor: o resumo vira template."""
    spec = os.environ.get("CONVERSA_CHAT_MODEL")
    if offline or not spec or not has_key_for(spec):
        return None
    return LLMSummarizer(chat_model(spec))


def make_csat_estimator(offline: bool = False) -> CsatEstimator | None:
    return None if offline or not has_jev() else JevCsatEstimator(cache=response_cache())


def load_csat_model(path: Path | None = None) -> CsatModel | None:
    """Modelo calibrado, se existir (``CONVERSA_CSAT_MODEL`` ou ``csat_model.json``)."""
    path = path or Path(os.environ.get("CONVERSA_CSAT_MODEL", DEFAULT_CSAT_MODEL))
    return CsatModel.load(path) if path.exists() else None
