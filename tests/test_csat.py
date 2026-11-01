import math
from pathlib import Path

import pytest
from conftest import reading

from sentiment_classifier.csat import (
    FEATURES,
    CsatModel,
    _spearman,
    conversation_features,
    cross_validated,
    evaluate,
    metrics,
)
from sentiment_classifier.dynamics import analyze_dynamics


def _row(**values: float) -> dict[str, float]:
    return {f: values.get(f, 3.0) for f in FEATURES}


def test_features_summarize_the_conversation() -> None:
    readings = [
        reading(0, "cliente", 4.0),
        reading(1, "atendente", 2.0, solucao_concreta=0.2),
        reading(2, "cliente", 1.5, cancelamento=0.9),
        reading(3, "atendente", 4.0, solucao_concreta=0.8),
        reading(4, "cliente", 2.5),
    ]
    f = conversation_features(readings, analyze_dynamics(readings), jev_csat=2.2)
    assert f["jev_csat"] == 2.2
    assert (f["cliente_fim"], f["cliente_min"], f["cliente_variacao"]) == (2.5, 1.5, -1.5)
    assert f["viradas_piora"] == 1.0
    assert (f["max_cancelamento"], f["atendente_solucao"]) == (0.9, 0.8)
    assert f["atendente_media"] == 3.0


def test_without_jev_estimate_falls_back_to_final_sentiment() -> None:
    readings = [reading(0, "cliente", 2.0)]
    assert conversation_features(readings, analyze_dynamics(readings), None)["jev_csat"] == 2.0


def test_model_learns_a_linear_relation_and_round_trips(tmp_path: Path) -> None:
    rows = [_row(jev_csat=1 + i % 5, cliente_fim=3 + (i % 3) / 10) for i in range(50)]
    targets = [r["jev_csat"] for r in rows]
    model = CsatModel.fit(rows, targets, l2=0.01)
    assert model.predict(_row(jev_csat=4.0, cliente_fim=3.1)) == pytest.approx(4.0, abs=0.05)
    assert model.predict(_row(jev_csat=99.0)) == 5.0  # sai sempre entre 1 e 5

    path = tmp_path / "m.json"
    model.save(path)
    assert CsatModel.load(path).predict(_row(jev_csat=2.0)) == model.predict(_row(jev_csat=2.0))


def test_cross_validation_never_trains_on_the_row_it_predicts() -> None:
    # Alvo é ruído puro: um modelo honesto não pode acertar em CV.
    rows = [_row(jev_csat=float(i)) for i in range(40)]
    targets = [float(1 + (i * 7919) % 5) for i in range(40)]
    pred = cross_validated(rows, targets)
    assert metrics(pred, targets).mae > 0.8


def test_metrics_and_spearman_with_ties() -> None:
    m = metrics([1.0, 2.0, 4.0, 5.0], [1.0, 3.0, 2.0, 5.0])
    assert m.mae == pytest.approx(0.75)
    # notas reais ≤2: 1.0 (previsto 1.0, marcado) e 2.0 (previsto 4.0, perdido)
    # previsões ≤2,5: 1.0 (nota 1, certo) e 2.0 (nota 3, alarme falso)
    assert (m.recall_insatisfeito, m.precisao_insatisfeito) == (0.5, 0.5)
    assert _spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(math.sqrt(3) / 2)
    assert math.isnan(_spearman([3, 3, 3], [1, 2, 3]))


def test_evaluate_reports_the_baselines() -> None:
    rows = [_row(jev_csat=1 + i % 5, cliente_fim=1 + i % 5) for i in range(20)]
    targets = [r["jev_csat"] for r in rows]
    report = evaluate(rows, targets)
    assert list(report) == [
        "sempre a média",
        "sentimento final do cliente",
        "estimativa do Jev",
        "modelo calibrado (CV 5 dobras)",
    ]
    assert report["estimativa do Jev"].mae == 0
    assert math.isnan(report["sempre a média"].spearman)


def test_unsatisfied_uses_rounded_rating() -> None:
    # 2,33 (média de anotadores) conta como insatisfeito; 2,67 não
    m = metrics([2.0, 2.0], [2.33, 2.67])
    assert (m.recall_insatisfeito, m.precisao_insatisfeito) == (1.0, 0.5)
