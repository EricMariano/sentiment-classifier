"""Áudio → turnos com sinais de voz → o mesmo grafo do texto."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_typesafe import TypeSafeClassifier

from sentiment_classifier.audio.acoustics import CallTurn, call_turns, to_turns
from sentiment_classifier.audio.roles import agent_label, assign_roles
from sentiment_classifier.audio.transcribe import Transcriber, Transcript
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
