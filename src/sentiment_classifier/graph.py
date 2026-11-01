"""O fluxo em LangGraph.

```
START → anonymize ─┬─ Send × N ─→ read_turn ──┐
                   └────────────→ estimate_csat ┴─→ aggregate ─┬→ summarize_llm ──────┐
                                                               └→ summarize_template ─┴→ review → END
```

- **anonymize**: dados pessoais viram marcadores antes de qualquer chamada externa.
- **read_turn** (fan-out com ``Send``): uma tarefa por turno, todas em paralelo; cada uma lê
  o turno com o Jev e devolve uma leitura ou uma falha. Os resultados se juntam no estado
  por um reducer (``operator.add``).
- **estimate_csat**: a nota de satisfação da conversa inteira, em paralelo com os turnos.
- **aggregate**: só roda quando todas as leituras e a nota chegaram (junção das duas
  arestas). Calcula curvas, pontos de virada, alertas e a nota calibrada, sem LLM.
- **summarize_llm / summarize_template**: o LLM só entra quando há momento-chave.
- **review** (opcional): se houver alerta de escalada ou cancelamento, o grafo pausa com
  ``interrupt`` e espera a decisão de um supervisor (``Command(resume=...)``).

Com um checkpointer, cada passo fica salvo: uma análise interrompida retoma de onde parou.
"""

from __future__ import annotations

import operator
from collections.abc import Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Annotated, Any, Literal, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer, Command, Send, interrupt

from sentiment_classifier.anonymize import redact_turns
from sentiment_classifier.classify import GaveUp, TurnClassifier
from sentiment_classifier.csat import (
    CsatEstimate,
    CsatEstimator,
    CsatModel,
    conversation_features,
    offline_estimate,
)
from sentiment_classifier.dynamics import analyze_dynamics
from sentiment_classifier.models import Dynamics, Turn, TurnFailure, TurnReading
from sentiment_classifier.summarize import (
    Summarizer,
    Summary,
    SummaryUnavailable,
    TemplateSummarizer,
)

REVIEWABLE = ("escalar", "cancelamento")
"""Alertas que pedem decisão humana antes de virar ação."""
MAX_CONCURRENCY = 8
"""Tarefas do grafo (turnos lidos) rodando ao mesmo tempo."""


@dataclass(frozen=True)
class Review:
    """Decisão do supervisor sobre os alertas de uma conversa."""

    escalate: bool
    note: str = ""


class AnalysisState(TypedDict, total=False):
    turns: list[Turn]
    known_names: list[str]
    redacted: list[Turn]
    redaction_counts: dict[str, int]
    readings: Annotated[list[TurnReading], operator.add]
    """Uma entrada por tarefa ``read_turn``; o reducer junta o que chega em paralelo."""
    failures: Annotated[list[TurnFailure], operator.add]
    csat_jev: CsatEstimate | None
    dynamics: Dynamics
    csat: CsatEstimate
    csat_features: dict[str, float]
    csat_predicted: float | None
    """Nota do modelo calibrado, quando há um treinado com notas reais."""
    summary: Summary
    review: Review | None


class TurnTask(TypedDict):
    """O que cada ``Send`` leva para ``read_turn``: a conversa e qual turno ler."""

    redacted: list[Turn]
    index: int


Graph = CompiledStateGraph[AnalysisState, None, AnalysisState, AnalysisState]


def build_graph(
    classifier: TurnClassifier,
    summarizer: Summarizer | None = None,
    csat_estimator: CsatEstimator | None = None,
    csat_model: CsatModel | None = None,
    *,
    review: bool = False,
    checkpointer: Checkpointer = None,
) -> Graph:
    """``summarizer=None`` usa só o resumo por template; ``csat_estimator=None``, a nota
    offline (sentimento final do cliente). ``review=True`` pausa o grafo, então precisa de
    um checkpointer: o deste grafo ou, se ele for subgrafo, o do grafo de fora."""
    template = TemplateSummarizer()
    llm = summarizer or template

    def anonymize(state: AnalysisState) -> AnalysisState:
        result = redact_turns(state["turns"], state.get("known_names", []))
        return {"redacted": result.turns, "redaction_counts": result.counts}

    def fan_out(state: AnalysisState) -> list[Send]:
        return [
            Send("read_turn", TurnTask(redacted=state["redacted"], index=i))
            for i in range(len(state["redacted"]))
        ]

    def read_turn(task: TurnTask) -> AnalysisState:
        result = classifier.read_one(task["redacted"], task["index"])
        if isinstance(result, TurnFailure):
            return {"failures": [result]}
        return {"readings": [result]}

    def estimate_csat(state: AnalysisState) -> AnalysisState:
        if csat_estimator is not None:
            # A nota é um extra: se o Jev não responder, fica a offline e a análise segue.
            with suppress(GaveUp):
                return {"csat_jev": csat_estimator.estimate(state["redacted"])}
        return {"csat_jev": None}

    def aggregate(state: AnalysisState) -> AnalysisState:
        readings = in_order(state.get("readings", []))
        failures = sorted(state.get("failures", []), key=lambda f: f.turn.index)
        if not readings:
            last = failures[-1].error if failures else "conversa vazia"
            raise RuntimeError(f"nenhum turno foi lido; último erro: {last}")
        dynamics = analyze_dynamics(readings, failures)
        estimate = state.get("csat_jev") or offline_estimate(dynamics)
        features = conversation_features(
            readings, dynamics, estimate.value if estimate.source == "jev" else None
        )
        return {
            "dynamics": dynamics,
            "csat": estimate,
            "csat_features": features,
            "csat_predicted": csat_model.predict(features) if csat_model else None,
        }

    def route(state: AnalysisState) -> Literal["summarize_llm", "summarize_template"]:
        return "summarize_llm" if state["dynamics"].key_turns else "summarize_template"

    def summarize_with(s: Summarizer):  # type: ignore[no-untyped-def]
        def node(state: AnalysisState) -> AnalysisState:
            args = (state["redacted"], in_order(state["readings"]), state["dynamics"])
            try:
                return {"summary": s.summarize(*args)}
            except SummaryUnavailable as error:  # o resumo é o último passo: não perde a análise
                cause = type(error.__cause__ or error).__name__
                note = f"LLM indisponível ({cause}); resumo montado dos sinais."
                return {"summary": replace(template.summarize(*args), note=note)}

        return node

    def review_alerts(state: AnalysisState) -> AnalysisState:
        alerts = [a for a in state["dynamics"].alerts if a.kind in REVIEWABLE]
        if not alerts:
            return {"review": None}
        decision: dict[str, Any] = interrupt(
            {
                "pergunta": "Escalar este atendimento para um supervisor?",
                "alertas": [
                    {"tipo": a.kind, "turno": a.turn_index, "p": a.probability, "motivo": a.reason}
                    for a in alerts
                ],
                "satisfacao_estimada": round(state["csat"].value, 1),
            }
        )
        return {"review": Review(bool(decision.get("escalar")), str(decision.get("nota", "")))}

    graph = StateGraph(AnalysisState)
    graph.add_node("anonymize", anonymize)
    graph.add_node("read_turn", read_turn)  # type: ignore[arg-type]  # recebe TurnTask, não o estado
    graph.add_node("estimate_csat", estimate_csat)
    graph.add_node("aggregate", aggregate)
    graph.add_node("summarize_llm", summarize_with(llm))
    graph.add_node("summarize_template", summarize_with(template))
    graph.add_edge(START, "anonymize")
    graph.add_conditional_edges("anonymize", fan_out, ["read_turn"])
    graph.add_edge("anonymize", "estimate_csat")
    graph.add_edge(["read_turn", "estimate_csat"], "aggregate")  # espera os dois ramos
    graph.add_conditional_edges("aggregate", route)
    last = END
    if review:
        graph.add_node("review", review_alerts)
        graph.add_edge("review", END)
        last = "review"
    graph.add_edge("summarize_llm", last)
    graph.add_edge("summarize_template", last)
    return graph.compile(checkpointer=checkpointer)


CHECKPOINT_TYPES: list[tuple[str, str]] = [
    ("sentiment_classifier.models", name)
    for name in ("Turn", "TurnReading", "TurnFailure", "TurningPoint", "Alert", "Dynamics")
] + [
    ("sentiment_classifier.csat", "CsatEstimate"),
    ("sentiment_classifier.summarize", "Summary"),
    ("sentiment_classifier.graph", "Review"),
    ("sentiment_classifier.audio.pipeline", "CallTranscript"),
    ("sentiment_classifier.audio.acoustics", "CallTurn"),
    ("sentiment_classifier.audio.acoustics", "RawSignals"),
]
"""Tipos do estado que o checkpoint pode ler de volta (e nenhum outro)."""


def serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)


def memory_checkpointer() -> InMemorySaver:
    """Checkpoint em memória: para pausar e retomar dentro do mesmo processo (revisão)."""
    return InMemorySaver(serde=serializer())


def sqlite_checkpointer(path: str) -> Any:
    """Checkpoint em disco: um lote interrompido retoma de onde parou.

    Guarda o estado da análise, inclusive o texto das conversas. Por isso o lote só usa
    checkpoint com as conversas já anonimizadas (ver ``batch.run_batch``).
    """
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    return SqliteSaver(sqlite3.connect(path, check_same_thread=False), serde=serializer())


def in_order(readings: Sequence[TurnReading]) -> list[TurnReading]:
    """As tarefas paralelas chegam fora de ordem; a análise usa a ordem dos turnos."""
    return sorted(readings, key=lambda r: r.turn.index)


def run_config(thread_id: str | None = None) -> RunnableConfig:
    config: RunnableConfig = {"max_concurrency": MAX_CONCURRENCY}
    if thread_id is not None:
        config["configurable"] = {"thread_id": thread_id}
    return config


def finalize(state: AnalysisState | dict[str, Any]) -> AnalysisState:
    """Estado final com leituras e falhas na ordem dos turnos."""
    out: AnalysisState = dict(state)  # type: ignore[assignment]
    out["readings"] = in_order(out.get("readings", []))
    out["failures"] = sorted(out.get("failures", []), key=lambda f: f.turn.index)
    return out


def analyze(
    turns: Sequence[Turn],
    classifier: TurnClassifier,
    summarizer: Summarizer | None = None,
    known_names: Sequence[str] = (),
    csat_estimator: CsatEstimator | None = None,
    csat_model: CsatModel | None = None,
) -> AnalysisState:
    """Análise completa numa chamada, sem pausa para revisão humana."""
    graph = build_graph(classifier, summarizer, csat_estimator, csat_model)
    initial: AnalysisState = {"turns": list(turns), "known_names": list(known_names)}
    return finalize(graph.invoke(initial, run_config()))


@dataclass(frozen=True)
class Event:
    """O que ``stream_analysis`` emite, na ordem em que acontece.

    - ``transcript``: a ligação foi transcrita (``payload`` é a ``CallTranscript``);
    - ``analysis``: leituras, curvas, alertas e nota prontas (o LLM ainda não terminou);
    - ``token``: um pedaço do resumo do LLM;
    - ``interrupt``: o grafo pausou para o supervisor (``payload`` diz o que decidir);
    - ``done``: estado final.
    """

    kind: Literal["transcript", "analysis", "token", "interrupt", "done"]
    state: AnalysisState | None = None
    text: str = ""
    payload: Any = None


def stream_analysis(
    graph: Graph, request: AnalysisState | Command[Any], config: RunnableConfig
) -> Iterator[Event]:
    """Roda o grafo emitindo o progresso. ``request`` é a entrada ou um ``Command(resume=...)``.

    Funciona também quando a análise é um subgrafo (ligações): os eventos de dentro dele
    chegam com ``subgraphs=True``, e o estado final é o do grafo de fora.
    """
    announced = False
    last: dict[str, Any] = {}
    parts = cast(
        Iterator[tuple[tuple[str, ...], str, Any]],
        graph.stream(
            request, config, stream_mode=["values", "updates", "messages"], subgraphs=True
        ),
    )
    for namespace, mode, chunk in parts:
        if mode == "values":
            if not namespace:
                if "call" in chunk and "call" not in last:
                    yield Event("transcript", payload=chunk["call"])
                last = chunk
            if not announced and "dynamics" in chunk:
                announced = True
                yield Event("analysis", finalize(chunk))
        elif mode == "messages":
            message, meta = chunk
            text = getattr(message, "text", "")
            if meta.get("langgraph_node") == "summarize_llm" and text:
                yield Event("token", text=str(text))
        elif mode == "updates" and not namespace and "__interrupt__" in chunk:
            yield Event("interrupt", payload=chunk["__interrupt__"][0].value)
    yield Event("done", finalize(last))
