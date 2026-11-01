"""O fluxo em LangGraph: anonimizar → ler cada turno → agregar → resumir.

O LLM só entra quando a agregação encontra algo (ponto de virada ou alerta). Conversa
estável recebe um resumo montado dos próprios sinais, sem custo de LLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from dataclasses import replace
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

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


class AnalysisState(TypedDict, total=False):
    turns: list[Turn]
    known_names: list[str]
    redacted: list[Turn]
    redaction_counts: dict[str, int]
    readings: list[TurnReading]
    failures: list[TurnFailure]
    dynamics: Dynamics
    csat: CsatEstimate
    csat_features: dict[str, float]
    csat_predicted: float | None
    """Nota do modelo calibrado, quando há um treinado com notas reais."""
    summary: Summary


def build_graph(
    classifier: TurnClassifier,
    summarizer: Summarizer | None = None,
    csat_estimator: CsatEstimator | None = None,
    csat_model: CsatModel | None = None,
) -> CompiledStateGraph[AnalysisState, None, AnalysisState, AnalysisState]:
    """``summarizer=None`` usa só o resumo por template; ``csat_estimator=None``, a nota
    offline (sentimento final do cliente)."""
    template = TemplateSummarizer()
    llm = summarizer or template

    def anonymize(state: AnalysisState) -> AnalysisState:
        result = redact_turns(state["turns"], state.get("known_names", []))
        return {"redacted": result.turns, "redaction_counts": result.counts}

    def classify(state: AnalysisState) -> AnalysisState:
        result = classifier.read(state["redacted"])
        return {"readings": result.readings, "failures": result.failures}

    def aggregate(state: AnalysisState) -> AnalysisState:
        return {"dynamics": analyze_dynamics(state["readings"], state["failures"])}

    def satisfaction(state: AnalysisState) -> AnalysisState:
        estimate = offline_estimate(state["dynamics"])
        if csat_estimator is not None:
            # A nota é um extra: se o Jev não responder, fica a offline e a análise segue.
            with suppress(GaveUp):
                estimate = csat_estimator.estimate(state["redacted"])
        features = conversation_features(
            state["readings"],
            state["dynamics"],
            estimate.value if estimate.source == "jev" else None,
        )
        predicted = csat_model.predict(features) if csat_model else None
        return {"csat": estimate, "csat_features": features, "csat_predicted": predicted}

    def route(state: AnalysisState) -> Literal["summarize_llm", "summarize_template"]:
        return "summarize_llm" if state["dynamics"].key_turns else "summarize_template"

    def summarize_with(s: Summarizer):  # type: ignore[no-untyped-def]
        def node(state: AnalysisState) -> AnalysisState:
            args = (state["redacted"], state["readings"], state["dynamics"])
            try:
                return {"summary": s.summarize(*args)}
            except SummaryUnavailable as error:  # o resumo é o último passo: não perde a análise
                cause = type(error.__cause__ or error).__name__
                note = f"LLM indisponível ({cause}); resumo montado dos sinais."
                return {"summary": replace(template.summarize(*args), note=note)}

        return node

    graph = StateGraph(AnalysisState)
    graph.add_node("anonymize", anonymize)
    graph.add_node("classify", classify)
    graph.add_node("aggregate", aggregate)
    graph.add_node("satisfaction", satisfaction)
    graph.add_node("summarize_llm", summarize_with(llm))
    graph.add_node("summarize_template", summarize_with(template))
    graph.add_edge(START, "anonymize")
    graph.add_edge("anonymize", "classify")
    graph.add_edge("classify", "aggregate")
    graph.add_edge("aggregate", "satisfaction")
    graph.add_conditional_edges("satisfaction", route)
    graph.add_edge("summarize_llm", END)
    graph.add_edge("summarize_template", END)
    return graph.compile()


def analyze(
    turns: Sequence[Turn],
    classifier: TurnClassifier,
    summarizer: Summarizer | None = None,
    known_names: Sequence[str] = (),
    csat_estimator: CsatEstimator | None = None,
    csat_model: CsatModel | None = None,
) -> AnalysisState:
    graph = build_graph(classifier, summarizer, csat_estimator, csat_model)
    return graph.invoke({"turns": list(turns), "known_names": list(known_names)})  # type: ignore[return-value]
