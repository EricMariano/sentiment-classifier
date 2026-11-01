"""Cache, desistência rápida do LLM e lote em paralelo."""

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from conftest import reading
from langchain_core.exceptions import ModelAuthenticationError, ModelRateLimitError
from test_jev_adapter import TURNS, FakeClient

from sentiment_classifier.batch import Conversation, run_batch
from sentiment_classifier.cache import ResponseCache
from sentiment_classifier.classify import JevTurnClassifier, OfflineHeuristicClassifier, Retry
from sentiment_classifier.dynamics import analyze_dynamics
from sentiment_classifier.models import Turn
from sentiment_classifier.summarize import LLMSummarizer, SummaryUnavailable, is_transient

# --- cache ---------------------------------------------------------------------------------


def _jev(client: FakeClient, cache: ResponseCache) -> JevTurnClassifier:
    return JevTurnClassifier(client=client, retry=Retry(sleep=lambda _: None), cache=cache)  # type: ignore[arg-type]


def test_cache_answers_repeated_requests_without_calling_jev(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "c.sqlite")
    first = _jev(client := FakeClient(), cache)
    first.read(TURNS)
    assert (len(client.requests), first.usage.requests, len(cache)) == (3, 3, 3)

    again = _jev(client2 := FakeClient(), ResponseCache(tmp_path / "c.sqlite"))  # outro processo
    result = again.read(TURNS)
    assert client2.requests == []
    assert (again.usage.requests, again.usage.cached) == (0, 3)
    assert [r.sentiment for r in result.readings] == [1.4, 1.4, 1.4]


def test_any_change_in_the_request_misses_the_cache(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "c.sqlite")
    _jev(FakeClient(), cache).read(TURNS)
    changed = [*TURNS[:2], Turn(2, "cliente", "c", voice={"leitura": "falou mais alto"})]
    client = FakeClient()
    _jev(client, cache).read(changed)
    assert len(client.requests) == 1  # só o turno que mudou


def test_cache_stores_no_conversation_text(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    secret = [Turn(0, "cliente", "texto que não pode ir para o disco")]
    _jev(FakeClient(), ResponseCache(path)).read(secret)
    dump = "".join(str(row) for row in sqlite3.connect(path).execute("SELECT * FROM jev"))
    assert "texto que não pode" not in dump and "turno_atual" not in dump


# --- LLM: desistir rápido e streaming ------------------------------------------------------


def test_out_of_credit_is_not_transient() -> None:
    assert is_transient(ModelRateLimitError("Rate limit reached, try again in 2s"))
    assert not is_transient(ModelRateLimitError("Error 429: insufficient_quota"))
    assert not is_transient(ModelAuthenticationError("invalid api key"))
    assert not is_transient(KeyError("bug"))


@dataclass
class Chunk:
    text: str


@dataclass
class FakeModel:
    """Falha com ``errors`` (em ordem) antes de responder ``reply``."""

    errors: list[Exception] = field(default_factory=list)
    reply: str = "resumo pronto"
    calls: int = 0

    def _maybe_fail(self) -> None:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)

    def invoke(self, messages: Any) -> Chunk:
        self._maybe_fail()
        return Chunk(self.reply)


def _summarizer(model: FakeModel) -> LLMSummarizer:
    return LLMSummarizer(model, sleep=lambda _: None)  # type: ignore[arg-type]


ARGS = (TURNS, [reading(0, "cliente", 3.0)], analyze_dynamics([reading(0, "cliente", 3.0)]))


def test_permanent_llm_error_gives_up_on_the_first_try() -> None:
    model = FakeModel([ModelRateLimitError("insufficient_quota")] * 3)
    with pytest.raises(SummaryUnavailable):
        _summarizer(model).summarize(*ARGS)
    assert model.calls == 1


def test_transient_llm_error_is_retried() -> None:
    model = FakeModel([ModelRateLimitError("slow down")])
    assert _summarizer(model).summarize(*ARGS).text == "resumo pronto"
    assert model.calls == 2


# --- lote em paralelo ----------------------------------------------------------------------


class SlowOffline(OfflineHeuristicClassifier):
    """Heurística com espera; conta quantas conversas diferentes estão sendo lidas juntas."""

    def __init__(self) -> None:
        super().__init__()
        self.active: dict[int, int] = {}
        self.peak = 0
        self._lock = threading.Lock()

    def read_one(self, turns: Any, index: int) -> Any:
        key = id(turns)  # a mesma conversa chega a todas as tarefas com a mesma lista
        with self._lock:
            self.active[key] = self.active.get(key, 0) + 1
            self.peak = max(self.peak, len(self.active))
        time.sleep(0.05)
        try:
            return super().read_one(turns, index)
        finally:
            with self._lock:
                self.active[key] -= 1
                if not self.active[key]:
                    del self.active[key]


def test_batch_runs_conversations_concurrently_with_the_same_results() -> None:
    items = [
        Conversation(
            f"c{i}", [Turn(0, "cliente", "absurdo, quero cancelar"), Turn(1, "atendente", "ok")]
        )
        for i in range(8)
    ]
    serial, parallel = SlowOffline(), SlowOffline()
    one = run_batch(items, serial, max_conversations=1)
    four = run_batch(items, parallel, max_conversations=4)
    assert (serial.peak, parallel.peak) == (1, 4)
    assert [(r.id, r.alertas, r.csat_estimado) for r in one] == [
        (r.id, r.alertas, r.csat_estimado) for r in four
    ]
