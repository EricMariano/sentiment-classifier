"""A voz ajuda o Jev a detectar sarcasmo? Mesma pergunta, com e sem os sinais de voz.

Cada fala do MUStARD vai ao Jev duas vezes, em ordem aleatória por fala:

- ``texto``: falas anteriores + a fala-alvo;
- ``texto+voz``: o mesmo, mais ``voz`` (números relativos ao personagem + leitura em
  palavras), no formato que o app usa nas ligações.

As respostas ficam em cache (``benchmarks/results/mustard_respostas.jsonl``); rodar de novo
só pede o que falta. O relatório compara AUC e macro-F1 (limiar 0,5) com intervalos por
bootstrap pareado, e mede se os sinais de voz sozinhos (regressão logística, validação
cruzada) já separam sarcasmo, para saber se há algo a ser aproveitado.

    python benchmarks/mustard/run.py
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_typesafe import Noul, NoulCriteria, TypeSafeClassifier

from sentiment_classifier.classify import Retry

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks" / "data" / "mustard" / "mustard.jsonl"
CACHE = ROOT / "benchmarks" / "results" / "mustard_respostas.jsonl"
CONDITIONS = ("texto", "texto+voz")
VOICE_KEYS = (
    "volume_vs_normal_db",
    "tom_vs_normal_semitons",
    "variacao_tom_vs_normal_semitons",
    "velocidade_vs_normal_pct",
)

QUESTION = Noul(
    instructions=(
        "Diálogo de uma série de TV em inglês. A fala em `turno_atual` é sarcástica ou "
        "irônica, isto é, o sentido pretendido é diferente (geralmente o oposto) do sentido "
        "literal? Use `contexto` para entender a situação. Se `turno_atual.voz` existir, são "
        "medidas do áudio desta fala comparadas ao jeito normal deste personagem (volume, tom "
        "de voz, variação da entonação, velocidade), com uma leitura em palavras; use-as junto "
        "com o texto."
    ),
    criteria=NoulCriteria(
        true="Sarcasmo ou ironia: elogio, concordância ou entusiasmo que não são sinceros.",
        false="Fala sincera, mesmo que seja engraçada, rude ou exagerada.",
    ),
)


def state(row: dict, condition: str) -> dict:
    current: dict = {"falante": row["speaker"], "texto": row["utterance"]}
    if condition == "texto+voz":
        current["voz"] = row["voz"]
    return {
        "contexto": [
            {"falante": s, "texto": t}
            for s, t in zip(row["context_speakers"], row["context"], strict=False)
        ],
        "turno_atual": current,
    }


def collect(rows: list[dict]) -> dict[tuple[str, str], float]:
    done: dict[tuple[str, str], float] = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            r = json.loads(line)
            done[(r["id"], r["condicao"])] = r["p"]
    jobs = [(row, c) for row in rows for c in CONDITIONS if (row["id"], c) not in done]
    random.Random(0).shuffle(jobs)  # condições intercaladas, sem bloco de uma depois da outra
    print(f"{len(done)} respostas em cache, {len(jobs)} a pedir")
    client, retry = TypeSafeClassifier(), Retry()

    def ask(job: tuple[dict, str]) -> tuple[str, str, float]:
        row, condition = job
        response, _ = retry.call(
            client, {"state": state(row, condition), "questions": {"sarcasmo": QUESTION}}
        )
        return row["id"], condition, response.nouls["sarcasmo"].noul

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as pool, CACHE.open("a") as f:
        for n, (uid, condition, p) in enumerate(pool.map(ask, jobs), 1):
            done[(uid, condition)] = p
            f.write(json.dumps({"id": uid, "condicao": condition, "p": p}) + "\n")
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)}")
    return done


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def macro_f1(pred: np.ndarray, labels: np.ndarray) -> float:
    f1s = []
    for cls in (0, 1):
        tp = ((pred == cls) & (labels == cls)).sum()
        fp = ((pred == cls) & (labels != cls)).sum()
        fn = ((pred != cls) & (labels == cls)).sum()
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s))


def bootstrap_diff(
    a: np.ndarray, b: np.ndarray, y: np.ndarray, metric, n: int = 2000
) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(metric(b[i], y[i]) - metric(a[i], y[i]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def voice_only_auc(rows: list[dict], y: np.ndarray, folds: int = 5) -> float:
    """Regressão logística só com os números de voz, validação cruzada por fala."""
    x = np.array([[float(r["voz"].get(k, 0.0)) for k in VOICE_KEYS] for r in rows])
    scores = np.zeros(len(rows))
    order = np.random.default_rng(0).permutation(len(rows))
    for k in range(folds):
        test = order[k::folds]
        train = np.setdiff1d(order, test)
        mu, sd = x[train].mean(0), x[train].std(0) + 1e-9
        xt = np.c_[np.ones(len(train)), (x[train] - mu) / sd]
        w = np.zeros(xt.shape[1])
        for _ in range(2000):  # gradiente simples, com um pouco de L2
            p = 1 / (1 + np.exp(-xt @ w))
            w -= 0.1 * (xt.T @ (p - y[train]) / len(train) + 0.01 * w)
        scores[test] = np.c_[np.ones(len(test)), (x[test] - mu) / sd] @ w
    return auc(scores, y)


def main() -> None:
    load_dotenv(ROOT / ".env")
    rows = [json.loads(line) for line in DATA.read_text().splitlines()]
    answers = collect(rows)
    y = np.array([int(r["sarcasm"]) for r in rows])
    p = {c: np.array([answers[(r["id"], c)] for r in rows]) for c in CONDITIONS}

    print(f"\n{len(rows)} falas, {y.sum()} sarcásticas\n")
    print(f"{'condição':<11} {'AUC':>6} {'macro-F1':>9} {'acerto':>7} {'p média (sarc / não)':>22}")
    for c in CONDITIONS:
        pred = (p[c] >= 0.5).astype(int)
        print(
            f"{c:<11} {auc(p[c], y):>6.3f} {macro_f1(pred, y):>9.3f} {(pred == y).mean():>7.3f} "
            f"{p[c][y == 1].mean():>10.2f} / {p[c][y == 0].mean():.2f}"
        )
    a, b = p["texto"], p["texto+voz"]
    lo, hi = bootstrap_diff(a, b, y, auc)
    print(f"\nΔAUC (voz − texto): {auc(b, y) - auc(a, y):+.3f}  IC95% [{lo:+.3f}, {hi:+.3f}]")
    f1 = lambda s, t: macro_f1((s >= 0.5).astype(int), t)  # noqa: E731
    lo, hi = bootstrap_diff(a, b, y, f1)
    print(f"Δmacro-F1:          {f1(b, y) - f1(a, y):+.3f}  IC95% [{lo:+.3f}, {hi:+.3f}]")
    flips = ((a >= 0.5) != (b >= 0.5)).sum()
    print(f"falas em que a voz mudou a resposta (limiar 0,5): {flips}")
    print(f"\nsó os números de voz (logística, CV 5): AUC {voice_only_auc(rows, y):.3f}")
    for show in sorted({r["show"] for r in rows}):
        m = np.array([r["show"] == show for r in rows])
        if y[m].min() != y[m].max():
            print(
                f"  {show:<15} n={m.sum():<4} AUC texto {auc(a[m], y[m]):.3f}  texto+voz {auc(b[m], y[m]):.3f}"
            )


if __name__ == "__main__":
    main()
