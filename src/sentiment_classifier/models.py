"""Tipos de dados: turnos, a leitura de cada turno e o relatório da conversa."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Speaker = Literal["cliente", "atendente"]
SPEAKERS: tuple[Speaker, ...] = ("cliente", "atendente")
AlertKind = Literal["cancelamento", "escalar", "sarcasmo"]


@dataclass(frozen=True)
class Turn:
    """Uma fala, já atribuída a quem falou."""

    index: int
    speaker: Speaker
    text: str
    voice: dict[str, float | str] | None = None
    """Sinais do áudio (só em ligações), relativos ao começo da ligação do falante. Vão
    para o Jev junto com o texto. Ver ``sentiment_classifier.audio.acoustics``."""


@dataclass(frozen=True)
class TurnReading:
    """O que o classificador respondeu sobre um turno.

    As dimensões ficam separadas de propósito: sentimento (o quão bem a pessoa está),
    emoção (qual sentimento) e intenção (o que ela quer fazer) não são a mesma coisa.
    Um cliente educado pode estar decidido a cancelar.

    ``sentiment`` vai de 1 (muito negativo) a 5 (muito positivo). Para o atendente, a
    escala mede o tom (1 hostil, 5 empático e resolutivo). ``flags`` guarda as
    probabilidades de sim das perguntas binárias (``Noul``) feitas para esse falante.
    """

    turn: Turn
    sentiment: float
    sentiment_confidence: float
    label: str
    label_confidence: float
    label_probabilities: dict[str, float]
    flags: dict[str, float]


@dataclass(frozen=True)
class TurnFailure:
    """Turno que ficou sem leitura depois das tentativas. A análise segue sem ele."""

    turn: Turn
    error: str
    attempts: int


@dataclass(frozen=True)
class ReadResult:
    readings: list[TurnReading]
    failures: list[TurnFailure] = field(default_factory=list)


@dataclass(frozen=True)
class TurningPoint:
    """Mudança brusca no sentimento do cliente entre dois turnos seguidos dele."""

    customer_before: int
    customer_after: int
    delta: float
    agent_turns: tuple[int, ...]
    """Falas do atendente entre os dois turnos do cliente. Vieram antes da mudança, o que
    não prova que a causaram: é por onde começar a olhar."""
    missing_turns: tuple[int, ...] = ()
    """Turnos no intervalo que ficaram sem leitura (falha do provedor)."""

    @property
    def direction(self) -> Literal["piorou", "melhorou"]:
        return "melhorou" if self.delta > 0 else "piorou"


@dataclass(frozen=True)
class Alert:
    kind: AlertKind
    turn_index: int
    probability: float | None
    """Probabilidade do Jev, ou ``None`` quando o alerta vem da curva e não de uma pergunta."""
    reason: str


@dataclass
class Dynamics:
    """Agregados da conversa, calculados sem LLM."""

    curves: dict[Speaker, list[tuple[int, float]]]
    turning_points: list[TurningPoint]
    alerts: list[Alert]
    customer_start: float | None
    customer_end: float | None
    key_turns: list[int] = field(default_factory=list)
    """Turnos que vão para o LLM: pontos de virada e alertas, com um turno de contexto."""
    unread_turns: list[int] = field(default_factory=list)
    """Turnos sem leitura (falha do provedor depois das tentativas)."""
