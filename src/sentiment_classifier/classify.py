"""Classificadores de turno: o Jev (TypeSafe) e uma heurística offline para demonstração."""

from __future__ import annotations

import random
import re
import threading
import time
import unicodedata
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

from langchain_typesafe import ClassifierRequest, ClassifierResponse, TypeSafeClassifier
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)

from sentiment_classifier.cache import ResponseCache, model_name, request_key
from sentiment_classifier.models import ReadResult, Turn, TurnFailure, TurnReading
from sentiment_classifier.questions import (
    AGENT_POSTURES,
    CUSTOMER_EMOTIONS,
    LABEL_QUESTION,
    QUESTIONS,
    turn_state,
)


class TurnClassifier(Protocol):
    name: str

    def read(self, turns: Sequence[Turn]) -> ReadResult:
        """Uma leitura por turno lido; os que falharam vão em ``failures``."""
        ...


@dataclass
class Usage:
    requests: int = 0
    retries: int = 0
    cached: int = 0
    """Respostas que vieram do cache, sem chamar o Jev."""
    input_tokens: int = 0
    output_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, response: ClassifierResponse, attempts: int) -> None:
        """``attempts == 0`` quer dizer que veio do cache."""
        with self._lock:  # o lote analisa várias conversas ao mesmo tempo
            if attempts == 0:
                self.cached += 1
                return
            self.requests += 1
            self.retries += attempts - 1
            self.input_tokens += response.usage.input_tokens or 0
            self.output_tokens += response.usage.output_tokens or 0


TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    TypeSafeRateLimitError,
    TypeSafeInternalServerError,
    TypeSafeAPIConnectionError,  # inclui timeout
)


class GaveUp(Exception):
    """Erro transitório que persistiu em todas as tentativas."""

    def __init__(self, error: Exception, attempts: int) -> None:
        super().__init__(f"{type(error).__name__}: {error}")
        self.error, self.attempts = error, attempts


@dataclass
class Retry:
    """Tentativas com espera exponencial; respeita o ``Retry-After`` do limite de taxa.

    Só erros transitórios são repetidos. Chave inválida ou requisição malformada (4xx)
    sobem na hora: repetir não resolve, e o mesmo erro valeria para todos os turnos.
    """

    max_attempts: int = 4
    base_delay: float = 1.0
    max_delay: float = 20.0
    sleep: Callable[[float], None] = time.sleep

    def call(
        self,
        client: TypeSafeClassifier,
        request: ClassifierRequest,
        cache: ResponseCache | None = None,
    ) -> tuple[ClassifierResponse, int]:
        """Resposta e número de tentativas (0 quando veio do ``cache``)."""
        key = request_key(request, model_name(client)) if cache is not None else ""
        if cache is not None and (hit := cache.get(key)) is not None:
            return hit, 0
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = client.invoke(request)
                if cache is not None:
                    cache.put(key, response)
                return response, attempt
            except TRANSIENT_ERRORS as error:
                if attempt == self.max_attempts:
                    raise GaveUp(error, attempt) from error
                wait = self.base_delay * 2 ** (attempt - 1) * (0.5 + random.random())
                retry_after = getattr(error, "retry_after_ms", None)
                if retry_after:
                    wait = max(wait, retry_after / 1000)
                self.sleep(min(wait, self.max_delay))
        raise AssertionError("inalcançável")


@dataclass
class JevTurnClassifier:
    """Uma requisição ao Jev por turno, com todas as perguntas daquele falante.

    As requisições saem em paralelo e cada uma tem suas próprias tentativas. Um turno que
    continua falhando vira uma ``TurnFailure`` e a análise segue com os demais: a curva
    fica com uma lacuna em vez de um valor inventado. Se nenhum turno for lido, ou se o
    erro não for transitório, a análise para com o erro.
    """

    client: TypeSafeClassifier = field(default_factory=TypeSafeClassifier)
    max_concurrency: int = 8
    retry: Retry = field(default_factory=Retry)
    cache: ResponseCache | None = None
    name: str = "Jev (TypeSafe)"
    usage: Usage = field(default_factory=Usage)

    def read(self, turns: Sequence[Turn]) -> ReadResult:
        def one(turn: Turn) -> tuple[ClassifierResponse, int]:
            request: ClassifierRequest = {
                "state": turn_state(turns, turn.index),  # type: ignore[typeddict-item]
                "questions": QUESTIONS[turn.speaker],
            }
            return self.retry.call(self.client, request, self.cache)

        readings: list[TurnReading] = []
        failures: list[TurnFailure] = []
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures = [(turn, pool.submit(one, turn)) for turn in turns]
            for turn, future in futures:
                try:
                    response, attempts = future.result()
                except GaveUp as gave_up:
                    failures.append(TurnFailure(turn, str(gave_up), gave_up.attempts))
                    continue
                self.usage.add(response, attempts)
                readings.append(_reading(turn, response))
        if turns and not readings:
            raise RuntimeError(f"nenhum turno foi lido; último erro: {failures[-1].error}")
        return ReadResult(readings, failures)


def _reading(turn: Turn, response: ClassifierResponse) -> TurnReading:
    score = response.scores["sentimento"]
    label = response.choices[LABEL_QUESTION[turn.speaker]]
    return TurnReading(
        turn=turn,
        sentiment=score.score + 1,  # o Jev numera a rubrica a partir de zero
        sentiment_confidence=score.confidence,
        label=label.choice,
        label_confidence=label.confidence,
        label_probabilities=dict(label.probabilities),
        flags={name: answer.noul for name, answer in response.nouls.items()},
    )


# --- Heurística offline -------------------------------------------------------------------
#
# Existe só para o app abrir sem chave de API. São listas de palavras, não um modelo: erra
# fácil, e o app avisa quando está nesse modo. Nenhum número de avaliação vem daqui.


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _hits(text: str, words: Sequence[str]) -> int:
    return sum(1 for w in words if re.search(rf"\b{w}", text))


_NEG = (
    "absurd",
    "ridicul",
    "pessim",
    "palhacada",
    "descaso",
    "vergonha",
    "inadmissivel",
    "nao aguento",
    "cansad",
    "de novo",
    "mais uma vez",
    "terceira vez",
    "quarta vez",
    "ninguem resolve",
    "horrivel",
    "lixo",
    "desrespeito",
    "nao funciona",
    "caiu",
    "sem internet",
    "cobrad",
    "errad",
    "demora",
    "esperando",
)
_POS = (
    "obrigad",
    "otimo",
    "perfeito",
    "resolveu",
    "resolvido",
    "maravilh",
    "excelente",
    "agradeco",
    "show",
    "funcionou",
    "valeu",
    "top",
    "gostei",
    "parabens",
)
_REPEAT = ("de novo", "mais uma vez", "terceira vez", "quarta vez", "toda vez")
_CANCEL = (
    "cancel",
    "encerrar",
    "outra operadora",
    "outra empresa",
    "trocar de",
    "portabilidade",
    "rescindir",
    "nao quero mais",
)
_ESCALATE = (
    "supervisor",
    "gerente",
    "procon",
    "reclame aqui",
    "justica",
    "advogad",
    "ouvidoria",
    "anatel",
    "processar",
)
_EMPATHY = (
    "entendo",
    "lamento",
    "sinto muito",
    "compreendo",
    "peco desculpas",
    "imagino",
    "tem toda razao",
)
_ACTION = (
    "vou ",
    "ja ",
    "agora mesmo",
    "estornei",
    "estorno",
    "abri",
    "registrei",
    "ate amanha",
    "em ate",
    "horas",
    "credito",
    "liberei",
    "enviei",
    "agendei",
)
_DEFENSIVE = (
    "o senhor nao",
    "a senhora nao",
    "nao e com a gente",
    "outro setor",
    "culpa",
    "o sistema",
    "nao posso fazer nada",
    "o senhor deveria",
    "a senhora deveria",
)
_BUREAU = (
    "procedimento",
    "conforme",
    "politica",
    "infelizmente nao",
    "nao e possivel",
    "protocolo",
    "aguarde",
    "prazo de",
    "dias uteis",
)


_MARKERS = {"NOME", "EMAIL", "CNPJ", "CARTAO", "TELEFONE", "ENDERECO"}


def _clip(x: float) -> float:
    return max(0.02, min(0.98, x))


@dataclass
class OfflineHeuristicClassifier:
    name: str = "heurística offline (não é o Jev)"

    def read(self, turns: Sequence[Turn]) -> ReadResult:
        return ReadResult([self._read(t) for t in turns])

    def _read(self, turn: Turn) -> TurnReading:
        text = _norm(turn.text)
        words = re.findall(r"[^\W\d_]+", turn.text)  # ignora marcadores como [CPF]
        shouting = sum(1 for w in words if len(w) > 3 and w.isupper() and w not in _MARKERS)
        neg = _hits(text, _NEG) + shouting + turn.text.count("!!")
        pos = _hits(text, _POS)
        if turn.speaker == "atendente":
            return self._agent(turn, text, neg, pos)

        # Elogio na mesma fala que uma reclamação ou repetição ("ótimo, mais uma vez").
        sarcasm = pos > 0 and (neg > 0 or _hits(text, _REPEAT) > 0 or "..." in turn.text)
        if sarcasm:
            neg, pos = neg + pos, 0
        cancel = _hits(text, _CANCEL)
        escalate = _hits(text, _ESCALATE)
        # Cancelar sem nenhuma palavra cordial conta como negativo; com cordialidade, não
        # (o cliente educado que vai embora mesmo assim).
        friction = neg + escalate + (cancel if not pos else 0)
        sentiment = 3 + min(pos, 2) - min(friction, 2)
        if friction:
            label = "raiva" if friction >= 3 or shouting else "frustracao"
        elif pos:
            label = "alivio" if re.search(r"\b(resolv|funcion|finalmente)", text) else "satisfacao"
        elif "?" in turn.text and re.search(r"\b(como|nao entendi|o que)", text):
            label = "confusao"
        else:
            label = "neutro"
        return TurnReading(
            turn=turn,
            sentiment=float(max(1, min(5, sentiment))),
            sentiment_confidence=0.5,
            label=label,
            label_confidence=0.5,
            label_probabilities={k: (0.6 if k == label else 0.4 / 6) for k in CUSTOMER_EMOTIONS},
            flags={
                "sarcasmo": _clip(0.85 if sarcasm else 0.05),
                "cancelamento": _clip(0.3 * cancel + (0.5 if cancel else 0)),
                "pede_escalada": _clip(0.4 * escalate + (0.4 if escalate else 0)),
            },
        )

    def _agent(self, turn: Turn, text: str, neg: int, pos: int) -> TurnReading:
        empathy, action = _hits(text, _EMPATHY), _hits(text, _ACTION)
        defensive, bureau = _hits(text, _DEFENSIVE), _hits(text, _BUREAU)
        scores = {
            "empatico": empathy,
            "resolutivo": action,
            "defensivo": defensive * 1.5,
            "burocratico": bureau,
        }
        label = max(scores, key=lambda k: scores[k]) if max(scores.values()) else "neutro"
        tone = (
            3 + min(empathy, 1) + min(action, 1) - min(defensive, 2) - min(bureau, 1) - min(neg, 1)
        )
        return TurnReading(
            turn=turn,
            sentiment=float(max(1, min(5, tone))),
            sentiment_confidence=0.5,
            label=label,
            label_confidence=0.5,
            label_probabilities={k: (0.6 if k == label else 0.1) for k in AGENT_POSTURES},
            flags={"solucao_concreta": _clip(0.3 + 0.3 * action - 0.2 * bureau)},
        )
