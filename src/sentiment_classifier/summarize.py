"""Resumo da conversa: o LLM só vê os momentos-chave, nunca a conversa inteira."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

from langchain_core.exceptions import (
    ModelAPIError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from sentiment_classifier.models import Dynamics, Turn, TurnReading


@dataclass(frozen=True)
class Summary:
    text: str
    source: Literal["llm", "template"]
    turns_sent: int
    """Quantos turnos foram enviados ao LLM (0 quando não houve chamada)."""
    note: str | None = None
    """Aviso para quem lê, por exemplo quando o LLM falhou e o template assumiu."""


T = TypeVar("T")


class SummaryUnavailable(Exception):
    """O provedor do LLM falhou depois das tentativas. Só este erro ativa o template."""


class Summarizer(Protocol):
    def summarize(
        self, turns: Sequence[Turn], readings: Sequence[TurnReading], dynamics: Dynamics
    ) -> Summary: ...


SYSTEM_PROMPT = """Você analisa atendimentos ao cliente em português do Brasil para um \
supervisor de qualidade. Você recebe os sinais já calculados por turno e só os trechos \
que importam, não a conversa inteira. Os dados pessoais já foram trocados por marcadores \
como [NOME] e [CPF]; não tente adivinhá-los.

Responda em Markdown, com estas seções e nada além delas:
### Resumo
Três frases no máximo: o que o cliente queria, como terminou e o risco que fica.
### Momentos-chave
Para cada ponto de virada, uma linha: a fala do atendente que veio antes da mudança, o \
que ele disse e como ela pode ter contribuído. A fala veio antes; isso não prova que \
causou a mudança, então não afirme causa.
### O que o atendente poderia ter feito diferente
Até três sugestões concretas, cada uma ligada a um turno marcado ATENDENTE. Nunca \
atribua ao atendente uma fala marcada CLIENTE (por exemplo, o sarcasmo é do cliente). \
Se o atendimento foi bom, diga isso e explique o que funcionou.

Baseie-se nos trechos e nos sinais. Não invente fatos que não estejam neles."""


def _fmt(value: float | None) -> str:
    return "?" if value is None else f"{value:.1f}"


def build_prompt(turns: Sequence[Turn], readings: Sequence[TurnReading], dynamics: Dynamics) -> str:
    by_index = {r.turn.index: r for r in readings}
    lines = [
        f"Conversa com {len(turns)} turnos. Sentimento do cliente (1 a 5): começou em "
        f"{_fmt(dynamics.customer_start)} e terminou em {_fmt(dynamics.customer_end)}.",
        "",
        "Pontos de virada (sentimento do cliente entre dois turnos dele):",
    ]
    lines += [
        f"- turno {tp.customer_before} → {tp.customer_after}: {tp.direction} {tp.delta:+.1f}; "
        f"falas do atendente no meio: {list(tp.agent_turns) or 'nenhuma'}"
        for tp in dynamics.turning_points
    ] or ["- nenhum"]
    lines += ["", "Alertas:"]
    lines += [
        f"- {a.kind} no turno {a.turn_index}"
        + (f" (p={a.probability:.2f})" if a.probability is not None else "")
        + f": {a.reason}"
        for a in dynamics.alerts
    ] or ["- nenhum"]
    if dynamics.unread_turns:
        lines += ["", f"Turnos sem leitura (falha técnica, ignore-os): {dynamics.unread_turns}"]
    lines += ["", "Trechos (só os momentos-chave; '…' marca turnos omitidos):"]
    previous = None
    for index in dynamics.key_turns:
        if previous is not None and index != previous + 1:
            lines.append("…")
        r = by_index.get(index)
        if r is None:
            lines.append(f"[{index}] (sem leitura)")
            previous = index
            continue
        flags = ", ".join(f"{k}={v:.2f}" for k, v in r.flags.items() if v >= 0.5)
        lines.append(
            f"[{index}] {r.turn.speaker.upper()} (sent={r.sentiment:.1f}, {r.label}"
            + (f", {flags}" if flags else "")
            + f"): {r.turn.text}"
        )
        previous = index
    return "\n".join(lines)


_TRANSIENT = (ModelRateLimitError, ModelAPIError, ModelConnectionError, ModelTimeoutError)
_OUT_OF_CREDIT = ("insufficient_quota", "credit_balance", "billing", "quota")


# O SDK da OpenRouter não usa os tipos do LangChain; pelo nome, sem importar o pacote.
_TRANSIENT_NAMES = {
    "TooManyRequestsResponseError",
    "ProviderOverloadedResponseError",
    "ServiceUnavailableResponseError",
    "BadGatewayResponseError",
    "InternalServerResponseError",
    "EdgeNetworkTimeoutResponseError",
    "RequestTimeoutResponseError",
    "NoResponseError",
}


def is_transient(error: Exception) -> bool:
    """Vale tentar de novo? Sem créditos chega como 429 (OpenAI) ou 402 (OpenRouter): não passa."""
    if not isinstance(error, _TRANSIENT) and type(error).__name__ not in _TRANSIENT_NAMES:
        return False
    return not any(word in str(error).lower() for word in _OUT_OF_CREDIT)


@dataclass
class LLMSummarizer:
    """Tenta até três vezes erros temporários; erro permanente (sem crédito, chave inválida)
    desiste na hora. Se não der, o grafo usa o template e avisa."""

    model: BaseChatModel
    attempts: int = 3
    base_delay: float = 1.0
    sleep: Callable[[float], None] = time.sleep

    def _messages(
        self, turns: Sequence[Turn], readings: Sequence[TurnReading], dynamics: Dynamics
    ) -> list[BaseMessage]:
        return [SystemMessage(SYSTEM_PROMPT), HumanMessage(build_prompt(turns, readings, dynamics))]

    def _retrying(self, call: Callable[[], T]) -> T:
        for attempt in range(1, self.attempts + 1):
            try:
                return call()
            except Exception as error:
                if attempt == self.attempts or not is_transient(error):
                    raise SummaryUnavailable(f"{type(error).__name__}: {error}") from error
                self.sleep(self.base_delay * 2 ** (attempt - 1))
        raise AssertionError("inalcançável")

    def summarize(
        self, turns: Sequence[Turn], readings: Sequence[TurnReading], dynamics: Dynamics
    ) -> Summary:
        messages = self._messages(turns, readings, dynamics)
        message = self._retrying(lambda: self.model.invoke(messages))
        return Summary(message.text, "llm", len(dynamics.key_turns))


class TemplateSummarizer:
    """Resumo sem LLM, montado só com os agregados. Usado offline e em conversas estáveis."""

    def summarize(
        self, turns: Sequence[Turn], readings: Sequence[TurnReading], dynamics: Dynamics
    ) -> Summary:
        start, end = dynamics.customer_start, dynamics.customer_end
        trend = "ficou estável"
        if start is not None and end is not None and abs(end - start) >= 0.5:
            trend = "melhorou" if end > start else "piorou"
        parts = [
            f"### Resumo\nConversa de {len(turns)} turnos. O sentimento do cliente {trend} "
            f"(de {_fmt(start)} para {_fmt(end)}, escala 1–5)."
        ]
        if dynamics.turning_points:
            parts.append("### Momentos-chave")
            by_index = {r.turn.index: r for r in readings}
            for tp in dynamics.turning_points:
                agents = [by_index[i] for i in tp.agent_turns]
                cause = (
                    "; ".join(f"[{a.turn.index}] {a.label}" for a in agents)
                    or "sem fala do atendente"
                )
                parts.append(
                    f"- Turno {tp.customer_after}: {tp.direction} {tp.delta:+.1f}. Antes disso: {cause}."
                )
        if not dynamics.turning_points and not dynamics.alerts:
            parts.append("Nenhum ponto de virada nem alerta: o LLM não foi chamado.")
        return Summary("\n".join(parts), "template", 0)
