"""Nota de satisfação (CSAT, 1 a 5): estimativa do Jev e, com notas reais, um modelo calibrado.

Duas camadas, para não chamar de previsão o que ainda não foi validado:

1. **Estimativa (sem treino).** Uma pergunta ``Score`` ao Jev sobre a conversa inteira:
   "que nota este cliente daria?". Custa uma requisição por conversa. É um palpite
   informado, sem garantia de bater com a pesquisa real.
2. **Modelo calibrado (com notas reais).** Com conversas que tenham a nota da pesquisa, uma
   regressão linear (ridge) aprende a combinar a estimativa do Jev com os sinais da
   conversa (como terminou, a pior fase, alertas, tom do atendente). ``evaluate`` mede o
   erro com validação cruzada contra baselines simples; só depois disso vale chamar de
   previsão.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Literal, Protocol

import numpy as np
from langchain_typesafe import ClassifierRequest, Score, TypeSafeClassifier

from sentiment_classifier.cache import ResponseCache
from sentiment_classifier.classify import Retry
from sentiment_classifier.models import Dynamics, Turn, TurnReading

CSAT_QUESTION = Score(
    instructions=(
        "Esta é uma conversa de atendimento ao cliente. Se o cliente respondesse agora a "
        "pesquisa de satisfação sobre ESTE atendimento, que nota daria? Considere se o "
        "problema foi resolvido ou bem encaminhado, como o cliente foi tratado e como ele "
        "estava no fim da conversa."
    ),
    criteria=[
        "1: muito insatisfeito; problema não resolvido e/ou cliente maltratado.",
        "2: insatisfeito; resolução ruim ou atendimento frio, cliente sai incomodado.",
        "3: neutro; atendimento cumpriu o mínimo, sem encantar nem irritar.",
        "4: satisfeito; problema resolvido ou bem encaminhado, atendimento cordial.",
        "5: muito satisfeito; resolução rápida e atendimento acima do esperado.",
    ],
)


@dataclass(frozen=True)
class CsatEstimate:
    value: float
    """Nota esperada, de 1 a 5 (pode ser fracionária)."""
    source: Literal["jev", "calibrado", "offline"]
    probabilities: dict[int, float] = field(default_factory=dict)
    """Distribuição sobre as notas 1–5 (só na estimativa do Jev)."""
    confidence: float | None = None


class CsatEstimator(Protocol):
    def estimate(self, turns: Sequence[Turn]) -> CsatEstimate: ...


def conversation_state(turns: Sequence[Turn]) -> list[dict[str, str]]:
    return [{"falante": t.speaker, "texto": t.text} for t in turns]


@dataclass
class JevCsatEstimator:
    client: TypeSafeClassifier = field(default_factory=TypeSafeClassifier)
    retry: Retry = field(default_factory=Retry)
    cache: ResponseCache | None = None

    def estimate(self, turns: Sequence[Turn]) -> CsatEstimate:
        request: ClassifierRequest = {
            "state": conversation_state(turns),  # type: ignore[typeddict-item]
            "questions": {"csat": CSAT_QUESTION},
        }
        response, _ = self.retry.call(self.client, request, self.cache)
        answer = response.scores["csat"]
        return CsatEstimate(
            value=answer.score + 1,
            source="jev",
            probabilities={level + 1: p for level, p in answer.probabilities.items()},
            confidence=answer.confidence,
        )


def offline_estimate(dynamics: Dynamics) -> CsatEstimate:
    """Sem Jev: repete o sentimento final do cliente. Não é uma estimativa de verdade."""
    end = dynamics.customer_end if dynamics.customer_end is not None else 3.0
    return CsatEstimate(value=end, source="offline")


# --- Sinais da conversa e modelo calibrado ------------------------------------------------

FEATURES = (
    "jev_csat",
    "cliente_fim",
    "cliente_min",
    "cliente_media",
    "cliente_variacao",
    "viradas_piora",
    "max_cancelamento",
    "max_escalada",
    "atendente_media",
    "atendente_solucao",
)


def conversation_features(
    readings: Sequence[TurnReading], dynamics: Dynamics, jev_csat: float | None
) -> dict[str, float]:
    customer = [r for r in readings if r.turn.speaker == "cliente"]
    agent = [r for r in readings if r.turn.speaker == "atendente"]
    c = [r.sentiment for r in customer] or [3.0]
    a = [r.sentiment for r in agent] or [3.0]

    def max_flag(rs: Sequence[TurnReading], name: str) -> float:
        return max((r.flags.get(name, 0.0) for r in rs), default=0.0)

    return {
        "jev_csat": jev_csat if jev_csat is not None else c[-1],
        "cliente_fim": c[-1],
        "cliente_min": min(c),
        "cliente_media": mean(c),
        "cliente_variacao": c[-1] - c[0],
        "viradas_piora": float(sum(1 for t in dynamics.turning_points if t.delta < 0)),
        "max_cancelamento": max_flag(customer, "cancelamento"),
        "max_escalada": max_flag(customer, "pede_escalada"),
        "atendente_media": mean(a),
        "atendente_solucao": max_flag(agent, "solucao_concreta"),
    }


@dataclass
class CsatModel:
    """Regressão ridge sobre ``FEATURES`` padronizadas. Treine com ``fit``."""

    weights: list[float]
    bias: float
    means: list[float]
    scales: list[float]
    n_train: int

    @classmethod
    def fit(
        cls, rows: Sequence[dict[str, float]], targets: Sequence[float], l2: float = 1.0
    ) -> CsatModel:
        x = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
        y = np.array(targets, dtype=float)
        means, scales = x.mean(axis=0), x.std(axis=0)
        scales[scales == 0] = 1.0
        z = (x - means) / scales
        bias = float(y.mean())
        w = np.linalg.solve(z.T @ z + l2 * np.eye(z.shape[1]), z.T @ (y - bias))
        return cls(w.tolist(), bias, means.tolist(), scales.tolist(), len(rows))

    def predict(self, row: dict[str, float]) -> float:
        z = [(row[f] - m) / s for f, m, s in zip(FEATURES, self.means, self.scales, strict=True)]
        value = self.bias + sum(w * v for w, v in zip(self.weights, z, strict=True))
        return float(min(5.0, max(1.0, value)))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"features": FEATURES, **self.__dict__}, indent=2))

    @classmethod
    def load(cls, path: Path) -> CsatModel:
        data = json.loads(path.read_text())
        if tuple(data.pop("features")) != FEATURES:
            raise ValueError("modelo treinado com outro conjunto de sinais; treine de novo")
        return cls(**data)


# --- Avaliação ---------------------------------------------------------------------------


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    def ranks(v: Sequence[float]) -> np.ndarray:
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        values = np.asarray(v)[order]
        i = 0
        while i < len(v):  # empates recebem o posto médio
            j = i
            while j + 1 < len(v) and values[j + 1] == values[i]:
                j += 1
            r[order[i : j + 1]] = (i + j) / 2
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


@dataclass(frozen=True)
class Metrics:
    mae: float
    spearman: float
    recall_insatisfeito: float
    """Das conversas insatisfeitas (nota real < 2,5, isto é, 1 ou 2 arredondando), quantas o
    método marcou com 2,5 ou menos. A nota real pode ser média de anotadores (2,33)."""
    precisao_insatisfeito: float


def metrics(pred: Sequence[float], true: Sequence[float]) -> Metrics:
    pairs = list(zip(pred, true, strict=True))
    hits = sum(1 for p, t in pairs if t < 2.5 and p <= 2.5)
    bad = sum(1 for _, t in pairs if t < 2.5)
    flagged = sum(1 for p, _ in pairs if p <= 2.5)
    return Metrics(
        mae=mean(abs(p - t) for p, t in pairs),
        spearman=_spearman(pred, true),
        recall_insatisfeito=hits / bad if bad else float("nan"),
        precisao_insatisfeito=hits / flagged if flagged else float("nan"),
    )


def cross_validated(
    rows: Sequence[dict[str, float]], targets: Sequence[float], folds: int = 5, seed: int = 0
) -> list[float]:
    """Previsão de cada conversa por um modelo que não a viu no treino."""
    order = np.random.default_rng(seed).permutation(len(rows))
    pred = [0.0] * len(rows)
    for k in range(folds):
        test = set(order[k::folds].tolist())
        train = [i for i in range(len(rows)) if i not in test]
        model = CsatModel.fit([rows[i] for i in train], [targets[i] for i in train])
        for i in test:
            pred[i] = model.predict(rows[i])
    return pred


def evaluate(rows: Sequence[dict[str, float]], targets: Sequence[float]) -> dict[str, Metrics]:
    """Compara o modelo calibrado com baselines que qualquer um conseguiria."""
    avg = float(np.mean(targets))
    return {
        "sempre a média": metrics([avg] * len(rows), targets),
        "sentimento final do cliente": metrics([r["cliente_fim"] for r in rows], targets),
        "estimativa do Jev": metrics([r["jev_csat"] for r in rows], targets),
        "modelo calibrado (CV 5 dobras)": metrics(cross_validated(rows, targets), targets),
    }
