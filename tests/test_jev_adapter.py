import threading
from typing import Any

import httpx2
import pytest
from langchain_typesafe import ClassifierResponse
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeAuthenticationError,
    TypeSafeRateLimitError,
)

from sentiment_classifier.classify import JevTurnClassifier, Retry
from sentiment_classifier.models import Turn
from sentiment_classifier.questions import AGENT_QUESTIONS, CUSTOMER_QUESTIONS


def _response(speaker: str) -> ClassifierResponse:
    label, options = (
        ("emocao", ["raiva", "neutro"]) if speaker == "cliente" else ("postura", ["neutro"])
    )
    answers: dict[str, Any] = {
        "sentimento": {
            "type": "score",
            "score": 0.4,
            "legend": {i: str(i) for i in range(5)},
            "probabilities": {0: 0.6, 1: 0.4, 2: 0, 3: 0, 4: 0},
            "confidence": 0.7,
        },
        label: {
            "type": "choice",
            "choice": options[0],
            "probabilities": {o: 1 / len(options) for o in options},
            "confidence": 0.5,
        },
    }
    if speaker == "cliente":
        answers["sarcasmo"] = {"type": "noul", "noul": 0.8}
    return ClassifierResponse.model_validate(
        {
            "model": "jev-test",
            "answers": answers,
            "usage": {"input_tokens": 100, "output_tokens": 5},
        }
    )


class FakeClient:
    """Responde por turno; ``errors[i]`` é a fila de exceções a levantar antes de responder."""

    def __init__(self, errors: dict[int, list[Exception]] | None = None) -> None:
        self.errors = errors or {}
        self.requests: list[Any] = []
        self._lock = threading.Lock()

    def invoke(self, request: Any) -> ClassifierResponse:
        current = request["state"]["turno_atual"]
        index = len(request["state"]["contexto"])  # nos testes, índice = tamanho do contexto
        with self._lock:
            self.requests.append(request)
            queue = self.errors.get(index, [])
            if queue:
                raise queue.pop(0)
        return _response(current["falante"])


TURNS = [Turn(0, "cliente", "a"), Turn(1, "atendente", "b"), Turn(2, "cliente", "c")]


def _jev(client: FakeClient, attempts: int = 3) -> tuple[JevTurnClassifier, list[float]]:
    waits: list[float] = []
    retry = Retry(max_attempts=attempts, base_delay=1.0, sleep=waits.append)
    return JevTurnClassifier(client=client, retry=retry), waits  # type: ignore[arg-type]


def test_one_request_per_turn_with_that_speakers_questions() -> None:
    client = FakeClient()
    jev, _ = _jev(client)
    result = jev.read(TURNS)

    by_turn = {len(r["state"]["contexto"]): r for r in client.requests}
    assert [by_turn[i]["questions"] for i in range(3)] == [
        CUSTOMER_QUESTIONS,
        AGENT_QUESTIONS,
        CUSTOMER_QUESTIONS,
    ]
    assert by_turn[2]["state"] == {
        "contexto": [{"falante": "cliente", "texto": "a"}, {"falante": "atendente", "texto": "b"}],
        "turno_atual": {"falante": "cliente", "texto": "c"},
    }
    assert jev.usage.requests == 3 and jev.usage.input_tokens == 300
    assert result.failures == []
    first = result.readings[0]
    assert first.sentiment == 1.4  # rubrica do Jev começa em zero; a escala do app em 1
    assert (first.label, first.flags) == ("raiva", {"sarcasmo": 0.8})
    assert result.readings[1].label == "neutro" and result.readings[1].flags == {}


def test_transient_errors_are_retried_and_honor_retry_after() -> None:
    limited = TypeSafeRateLimitError(429, None, httpx2.Headers({"retry-after": "7"}))
    client = FakeClient({1: [limited, TypeSafeAPIConnectionError("rede")]})
    jev, waits = _jev(client)
    result = jev.read(TURNS)
    assert [r.turn.index for r in result.readings] == [0, 1, 2]
    assert jev.usage.retries == 2
    assert waits[0] >= 7  # Retry-After manda mais que o backoff


def test_turn_that_keeps_failing_becomes_a_gap() -> None:
    client = FakeClient({1: [TypeSafeAPIConnectionError("rede")] * 3})
    jev, _ = _jev(client, attempts=3)
    result = jev.read(TURNS)
    assert [r.turn.index for r in result.readings] == [0, 2]
    [failure] = result.failures
    assert (failure.turn.index, failure.attempts) == (1, 3)
    assert "TypeSafeAPIConnectionError" in failure.error


def test_permanent_errors_stop_the_analysis() -> None:
    client = FakeClient({0: [TypeSafeAuthenticationError(401, None, httpx2.Headers())]})
    jev, waits = _jev(client)
    with pytest.raises(TypeSafeAuthenticationError):
        jev.read(TURNS)
    assert waits == []


def test_all_turns_failing_is_an_error() -> None:
    client = FakeClient({i: [TypeSafeAPIConnectionError("rede")] * 2 for i in range(3)})
    jev, _ = _jev(client, attempts=2)
    with pytest.raises(RuntimeError, match="nenhum turno"):
        jev.read(TURNS)
