"""Áudio → turnos com sinais de voz → o mesmo grafo do texto, como subgrafo."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from langchain_typesafe import TypeSafeClassifier
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from sentiment_classifier.audio.acoustics import CallTurn, call_turns, to_turns
from sentiment_classifier.audio.roles import agent_label, assign_roles
from sentiment_classifier.audio.transcribe import Transcriber, Transcript
from sentiment_classifier.graph import AnalysisState, Graph
from sentiment_classifier.models import Turn


@dataclass
class CallTranscript:
    turns: list[Turn]
    """Turnos prontos para ``analyze`` (texto ainda não anonimizado + sinais de voz)."""
    call: list[CallTurn]
    """Os mesmos turnos com tempos e medidas brutas, para a linha do tempo."""
    agent_confidence: float | None = None
    """Confiança do Jev ao escolher o atendente no áudio mono (``None`` no estéreo)."""
    duration: float = 0.0


def transcribe_call(
    path: Path, transcriber: Transcriber, role_client: TypeSafeClassifier | None = None
) -> CallTranscript:
    transcript: Transcript = transcriber.transcribe(path)
    if not transcript.utterances:
        raise ValueError("nenhuma fala reconhecida no áudio")
    confidence = None
    if not transcript.roles_known:
        agent, confidence = agent_label(transcript.utterances, role_client)
        transcript.utterances = assign_roles(transcript.utterances, agent)
    call = call_turns(transcript)
    duration = max(t.end for t in call)
    return CallTranscript(to_turns(call), call, confidence, duration)


class CallState(AnalysisState, total=False):
    """Estado do grafo da ligação: o da análise de texto mais o áudio."""

    path: str
    without_voice: bool
    """Manda só o texto ao Jev (para comparar com e sem os sinais de voz)."""
    call: CallTranscript


def build_call_graph(
    transcriber: Transcriber,
    analysis: Graph,
    role_client: TypeSafeClassifier | None = None,
    checkpointer: Checkpointer = None,
) -> CompiledStateGraph[CallState, None, CallState, CallState]:
    """``START → transcribe → analysis → END``, com ``analysis`` sendo o grafo do texto.

    O grafo do texto entra como subgrafo: compartilha as chaves do estado (``turns``,
    ``readings``...), herda o checkpointer e repassa a pausa para o supervisor. Com
    checkpoint, a transcrição (a etapa cara) não se repete se a análise cair depois dela.
    O estado guarda a transcrição ainda não anonimizada: use checkpoint em memória.
    """

    def transcribe(state: CallState) -> CallState:
        call = transcribe_call(Path(state["path"]), transcriber, role_client)
        turns = call.turns
        if state.get("without_voice"):
            turns = [replace(t, voice=None) for t in turns]
        return {"call": call, "turns": turns}

    graph = StateGraph(CallState)
    graph.add_node("transcribe", transcribe)
    graph.add_node("analysis", analysis)
    graph.add_edge(START, "transcribe")
    graph.add_edge("transcribe", "analysis")
    graph.add_edge("analysis", END)
    return graph.compile(checkpointer=checkpointer)
