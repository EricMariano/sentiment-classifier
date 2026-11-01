"""Garante que a demo offline continua contando a história das amostras.

Isto testa a heurística da demonstração, não a qualidade do Jev.
"""

from pathlib import Path

import pytest

from sentiment_classifier.classify import OfflineHeuristicClassifier
from sentiment_classifier.graph import analyze
from sentiment_classifier.parsing import parse_chat


def _run(samples: Path, name: str):  # type: ignore[no-untyped-def]
    return analyze(parse_chat((samples / name).read_text()), OfflineHeuristicClassifier())


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("01_calmo_para_irritado.txt", {"cancelamento", "escalar"}),
        ("03_educado_decidido_a_cancelar.txt", {"cancelamento"}),
        ("04_sarcasmo.txt", {"sarcasmo", "escalar"}),
    ],
)
def test_alert_kinds(samples: Path, name: str, expected: set[str]) -> None:
    state = _run(samples, name)
    assert {a.kind for a in state["dynamics"].alerts} == expected


def test_saved_conversation_improves(samples: Path) -> None:
    dyn = _run(samples, "02_salvo_pelo_atendente.txt")["dynamics"]
    assert dyn.alerts == []
    assert dyn.customer_end is not None and dyn.customer_start is not None
    assert dyn.customer_end - dyn.customer_start >= 2


def test_first_sample_turns_worse(samples: Path) -> None:
    state = _run(samples, "01_calmo_para_irritado.txt")
    assert state["redaction_counts"]["CPF"] == 1
    assert any(tp.direction == "piorou" for tp in state["dynamics"].turning_points)
