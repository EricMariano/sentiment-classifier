"""Jev × LLM: o mesmo trabalho, com acerto, tempo e custo.

A. Satisfação da conversa (JDDC traduzido, nota real): o Jev responde o ``Score`` de
   satisfação; o LLM recebe a mesma pergunta e a mesma rubrica e dá a nota.
B. Leitura por turno (piloto de 34 conversas): as mesmas perguntas de cada turno (do
   ``questions.py``) ao Jev e ao LLM. Mede custo, tempo, concordância e se a "nota pelo
   sentimento final do cliente" acerta igual.

Os dois rodam sem cache. O LLM responde em JSON, com as opções e rubricas copiadas das
perguntas do Jev, para que a única diferença seja quem responde.

Preço do Jev: US$ 0,042 por milhão de tokens de entrada, saída grátis, segundo sites de
terceiros (não confirmado na página da TypeSafe). Preço do LLM: catálogo da OpenRouter.

    python benchmarks/jev_vs_llm/run.py --modelo mistralai/mistral-small-3.2-24b-instruct
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Any

import certifi
import numpy as np
from dotenv import load_dotenv
from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier

from sentiment_classifier.anonymize import redact_turns
from sentiment_classifier.classify import Retry
from sentiment_classifier.csat import CSAT_QUESTION, _spearman, conversation_state
from sentiment_classifier.parsing import turns_from_messages
from sentiment_classifier.providers import chat_model
from sentiment_classifier.questions import QUESTIONS, turn_state

ROOT = Path(__file__).resolve().parents[2]
JEV_USD_PER_INPUT_TOKEN = 0.042 / 1e6


# --- as mesmas perguntas, em texto, para o LLM ---------------------------------------------


def describe(name: str, q: Any) -> str:
    if isinstance(q, Score):
        levels = "; ".join(f"{i + 1} = {c}" for i, c in enumerate(q.criteria))
        return f'- "{name}": número de 1 a {len(q.criteria)} (pode usar decimais). {q.instructions} Níveis: {levels}'
    if isinstance(q, Choice):
        opts = "; ".join(f'"{k}" ({v})' for k, v in q.criteria.items())
        return f'- "{name}": uma destas opções: {opts}. {q.instructions}'
    if isinstance(q, Noul):
        return f'- "{name}": probabilidade de 0 a 1 de a resposta ser sim. {q.instructions}'
    raise TypeError(q)


def prompt(state: Any, questions: dict[str, Any]) -> str:
    asks = "\n".join(describe(k, q) for k, q in questions.items())
    return (
        "Responda às perguntas sobre o estado abaixo. Devolva SÓ um objeto JSON com as chaves "
        f"pedidas, sem texto antes ou depois.\n\nPerguntas:\n{asks}\n\nEstado:\n"
        + json.dumps(state, ensure_ascii=False)
    )


def parse_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"sem JSON: {text[:80]!r}")
    return json.loads(match.group(0))


# --- quem responde --------------------------------------------------------------------------


class Jev:
    def __init__(self) -> None:
        self.client, self.retry = TypeSafeClassifier(), Retry()

    def ask(
        self, state: Any, questions: dict[str, Any]
    ) -> tuple[dict[str, float | str], float, int]:
        t0 = time.perf_counter()
        r, _ = self.retry.call(self.client, {"state": state, "questions": questions})
        dt = time.perf_counter() - t0
        out: dict[str, float | str] = {k: a.score + 1 for k, a in r.scores.items()}
        out |= {k: a.choice for k, a in r.choices.items()} | {k: a.noul for k, a in r.nouls.items()}
        return out, dt, (r.usage.input_tokens or 0)


class LLM:
    def __init__(self, model_id: str, price: tuple[float, float]) -> None:
        self.model, self.price = chat_model(f"openrouter:{model_id}"), price

    def ask(self, state: Any, questions: dict[str, Any]) -> tuple[dict[str, Any], float, float]:
        t0 = time.perf_counter()
        last: Exception | None = None
        cost = 0.0
        for _ in range(3):
            msg = self.model.invoke(prompt(state, questions))
            u = msg.usage_metadata or {}
            cost += (
                u.get("input_tokens", 0) * self.price[0] + u.get("output_tokens", 0) * self.price[1]
            )
            try:
                return parse_json(msg.text), time.perf_counter() - t0, cost
            except (ValueError, json.JSONDecodeError) as error:
                last = error
        raise RuntimeError(f"LLM sem JSON válido: {last}")


def or_price(model_id: str) -> tuple[float, float]:
    ctx = ssl.create_default_context(cafile=certifi.where())
    data = json.load(urllib.request.urlopen("https://openrouter.ai/api/v1/models", context=ctx))
    p = next(m["pricing"] for m in data["data"] if m["id"] == model_id)
    return float(p["prompt"]), float(p["completion"])


# --- A. satisfação --------------------------------------------------------------------------


def part_a(rows: list[dict], jev: Jev, llm: LLM) -> None:
    states = [
        conversation_state(redact_turns(turns_from_messages(r["messages"])).turns) for r in rows
    ]
    y = np.array([r["nota"] for r in rows])
    q = {"csat": CSAT_QUESTION}
    with ThreadPoolExecutor(8) as pool:
        t0 = time.perf_counter()
        jr = list(pool.map(lambda s: jev.ask(s, q), states))
        jev_wall = time.perf_counter() - t0
        t0 = time.perf_counter()
        lr = list(pool.map(lambda s: _safe(llm, s, q), states))
        llm_wall = time.perf_counter() - t0
    ok = [i for i, r in enumerate(lr) if r is not None]
    je = np.array([jr[i][0]["csat"] for i in ok], float)
    le = np.array([float(lr[i][0]["csat"]) for i in ok])
    yy = y[ok]
    print(
        f"\nA. Satisfação, {len(ok)} de {len(rows)} conversas (LLM sem JSON válido: {len(rows) - len(ok)})"
    )
    for name, e, lat, wall, cost in (
        (
            "Jev",
            je,
            [jr[i][1] for i in ok],
            jev_wall,
            sum(jr[i][2] for i in ok) * JEV_USD_PER_INPUT_TOKEN,
        ),
        ("LLM", le, [lr[i][1] for i in ok], llm_wall, sum(lr[i][2] for i in ok)),
    ):
        bad, flag = yy < 2.5, e <= 2.5
        print(
            f"  {name}: Spearman {_spearman(e, yy):.2f}  MAE {np.mean(abs(e - yy)):.2f}  "
            f"insatisfeitas {int((bad & flag).sum())}/{int(bad.sum())}  alarmes falsos {int((flag & ~bad).sum())}  "
            f"| mediana {median(lat):.2f}s por conversa, total {wall:.0f}s  | US$ {cost:.4f} "
            f"(US$ {cost / len(ok) * 1000:.2f} por 1.000)"
        )
    rng = np.random.default_rng(0)
    d = [
        _spearman(je[i], yy[i]) - _spearman(le[i], yy[i])
        for i in (rng.integers(0, len(ok), len(ok)) for _ in range(2000))
    ]
    print(
        f"  ΔSpearman (Jev − LLM): {np.mean(d):+.2f}  IC95% [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}]"
    )


def _safe(llm: LLM, state: Any, q: dict[str, Any]) -> Any:
    try:
        return llm.ask(state, q)
    except Exception:
        return None


# --- B. leitura por turno -------------------------------------------------------------------


def part_b(rows: list[dict], jev: Jev, llm: LLM) -> None:
    jobs = []
    for r in rows:
        turns = redact_turns(turns_from_messages(r["messages"])).turns
        for t in turns:
            jobs.append((r["id"], t, turn_state(turns, t.index), QUESTIONS[t.speaker]))
    with ThreadPoolExecutor(8) as pool:
        t0 = time.perf_counter()
        jr = list(pool.map(lambda j: jev.ask(j[2], j[3]), jobs))
        jev_wall = time.perf_counter() - t0
        t0 = time.perf_counter()
        lr = list(pool.map(lambda j: _safe(llm, j[2], j[3]), jobs))
        llm_wall = time.perf_counter() - t0
    ok = [i for i, r in enumerate(lr) if r is not None]
    jev_cost = sum(jr[i][2] for i in ok) * JEV_USD_PER_INPUT_TOKEN
    llm_cost = sum(lr[i][2] for i in ok)
    print(
        f"\nB. Leitura por turno, {len(ok)} de {len(jobs)} turnos (LLM sem JSON válido: {len(jobs) - len(ok)})"
    )
    print(
        f"  Jev: mediana {median(jr[i][1] for i in ok):.2f}s por turno, total {jev_wall:.0f}s, US$ {jev_cost:.4f}"
    )
    print(
        f"  LLM: mediana {median(lr[i][1] for i in ok):.2f}s por turno, total {llm_wall:.0f}s, US$ {llm_cost:.4f}"
    )

    cust = [i for i in ok if jobs[i][1].speaker == "cliente"]
    js = np.array([float(jr[i][0]["sentimento"]) for i in cust])
    ls = np.array([float(lr[i][0].get("sentimento", np.nan)) for i in cust])
    good = ~np.isnan(ls)
    same_emotion = np.mean([jr[i][0]["emocao"] == lr[i][0].get("emocao") for i in cust])
    print(
        f"  concordância no cliente: sentimento Spearman {_spearman(js[good], ls[good]):.2f}, mesma emoção {same_emotion:.0%}"
    )
    for flag in ("sarcasmo", "cancelamento", "pede_escalada"):
        a = np.array([float(jr[i][0][flag]) >= 0.7 for i in cust])
        b = np.array([float(lr[i][0].get(flag, 0)) >= 0.7 for i in cust])
        print(
            f"  alerta {flag:<13}: Jev {a.sum():>3}  LLM {b.sum():>3}  os dois {int((a & b).sum()):>3}"
        )

    # nota pelo sentimento final do cliente, contra a nota real
    rating = {r["id"]: r["nota"] for r in rows}
    last: dict[str, dict[str, float]] = {}
    for i in cust:
        cid = jobs[i][0]
        last.setdefault(cid, {})
        last[cid]["jev"] = float(jr[i][0]["sentimento"])
        if not np.isnan(float(lr[i][0].get("sentimento", np.nan))):
            last[cid]["llm"] = float(lr[i][0]["sentimento"])
    ids = [c for c in last if {"jev", "llm"} <= set(last[c])]
    y = np.array([rating[c] for c in ids])
    for who in ("jev", "llm"):
        e = np.array([last[c][who] for c in ids])
        print(
            f"  nota pelo sentimento final ({who}): Spearman {_spearman(e, y):.2f} com a nota real ({len(ids)} conversas)"
        )


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default="mistralai/mistral-small-3.2-24b-instruct")
    parser.add_argument("--so", choices=["a", "b"])
    args = parser.parse_args()
    jev, llm = Jev(), LLM(args.modelo, or_price(args.modelo))
    print(f"LLM: {args.modelo} | Jev: US$ 0,042/M tokens de entrada (preço de terceiros)")
    if args.so in (None, "a"):
        a_rows = [
            json.loads(line)
            for line in (ROOT / "benchmarks/data/jddc_300_pt.jsonl").read_text().splitlines()
        ]
        part_a(a_rows, jev, llm)
    if args.so in (None, "b"):
        b_rows = [
            json.loads(line)
            for line in (ROOT / "benchmarks/data/jddc_piloto_pt.jsonl").read_text().splitlines()
        ]
        part_b(b_rows, jev, llm)


if __name__ == "__main__":
    main()
