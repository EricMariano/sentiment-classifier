"""No áudio mono, o diarizador diz "falante A" e "falante B". Quem é o atendente?

Uma pergunta ``Choice`` ao Jev sobre as primeiras falas de cada um. Sem Jev, assume que
quem fala primeiro é o atendente (comum em ligação receptiva: "Central X, bom dia").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from langchain_typesafe import Choice, TypeSafeClassifier

from sentiment_classifier.anonymize import redact_text
from sentiment_classifier.audio.transcribe import Utterance
from sentiment_classifier.classify import Retry

FIRST_LINES = 4


def _opening(utterances: Sequence[Utterance], label: str) -> str:
    lines = [u.text for u in utterances if u.speaker == label][:FIRST_LINES]
    return redact_text(" / ".join(lines))  # o texto vai para o Jev: anonimiza antes


def agent_label(
    utterances: Sequence[Utterance], client: TypeSafeClassifier | None = None
) -> tuple[str, float | None]:
    """Rótulo do atendente e a confiança do Jev (``None`` quando é a regra de quem abre)."""
    labels = list(dict.fromkeys(u.speaker for u in utterances))
    if len(labels) == 1:
        return labels[0], None
    if client is None:
        return labels[0], None
    question = Choice(
        instructions=(
            "Ligação de atendimento ao cliente em português. Cada opção mostra as primeiras "
            "falas de uma pessoa. Qual delas é o ATENDENTE da empresa (quem atende, segue "
            "procedimento, pede dados, oferece solução)?"
        ),
        criteria={label: _opening(utterances, label) for label in labels},
    )
    response, _ = Retry().call(
        client,
        {"state": "Escolha o atendente.", "questions": {"atendente": question}},
    )
    answer = response.choices["atendente"]
    return answer.choice, answer.confidence


def assign_roles(utterances: Sequence[Utterance], agent: str) -> list[Utterance]:
    """Atendente vira ``atendente``; qualquer outro rótulo, ``cliente``."""
    return [
        replace(u, speaker="atendente" if u.speaker == agent else "cliente") for u in utterances
    ]
