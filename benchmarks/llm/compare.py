"""Qual LLM escreve o resumo dos momentos-chave? Teste pequeno e barato.

As mesmas conversas, o mesmo prompt (o do app), modelos diferentes pela OpenRouter. Para
cada resumo: tempo, tokens, custo (preço público da OpenRouter) e uma checagem automática:

- formato: as três seções pedidas, nessa ordem;
- atribuição: as sugestões citam só turnos do ATENDENTE;
- causa: não usa palavras de causa ("causou", "por causa", "provocou", "levou o cliente a").

A checagem automática é grosseira; leia os textos em ``benchmarks/results/llm_resumos.jsonl``.

    python benchmarks/llm/compare.py qwen/qwen3.7-flash mistralai/mistral-small-3.2-24b-instruct
"""

from __future__ import annotations

import json
import re
import ssl
import sys
import time
import urllib.request
from pathlib import Path

import certifi
from dotenv import load_dotenv

from sentiment_classifier.batch import load_conversations
from sentiment_classifier.graph import analyze
from sentiment_classifier.providers import chat_model, make_classifier
from sentiment_classifier.summarize import LLMSummarizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "results" / "llm_resumos.jsonl"
SECTIONS = ["### Resumo", "### Momentos-chave", "### O que o atendente poderia ter feito diferente"]
HYPHENS = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-"})
CAUSAL = re.compile(r"\b(causou|causaram|por causa d|provocou|provocaram|levou o cliente a)", re.I)


def prices() -> dict[str, tuple[float, float]]:
    ctx = ssl.create_default_context(cafile=certifi.where())
    data = json.load(urllib.request.urlopen("https://openrouter.ai/api/v1/models", context=ctx))
    return {
        m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"]))
        for m in data["data"]
        if m.get("pricing")
    }


def check(text: str, agent_turns: set[int], customer_turns: set[int]) -> dict[str, object]:
    text = text.translate(HYPHENS)  # "Momentos‑chave" com hífen especial é o mesmo título
    positions = [text.find(s) for s in SECTIONS]
    fmt = all(p >= 0 for p in positions) and positions == sorted(positions)
    tips = text[text.find(SECTIONS[2]) :] if positions[2] >= 0 else ""
    cited = {int(n) for n in re.findall(r"(?:turno|\[)\s*(\d+)", tips, re.I)}
    return {
        "formato_ok": fmt,
        "cita_turno_do_cliente": sorted(cited & customer_turns),
        "palavras_de_causa": sorted({m.group(0).lower() for m in CAUSAL.finditer(text)}),
    }


def conversations() -> list:
    picked = [c for c in load_conversations([ROOT / "samples"]) if not isinstance(c, tuple)]
    jddc = ROOT / "benchmarks" / "data" / "jddc_piloto_pt.jsonl"
    if jddc.exists():
        rows = [c for c in load_conversations([jddc]) if not isinstance(c, tuple)]
        low = sorted(rows, key=lambda c: c.rating or 3)
        picked += [low[0], low[len(low) // 2]]  # uma insatisfeita e uma do meio
    return picked


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = sys.argv[1:]
    no_reasoning = "--sem-raciocinio" in args  # para modelos que "pensam" antes de responder
    models = [a for a in args if not a.startswith("--")]
    options = {"reasoning": {"enabled": False}} if no_reasoning else {}
    price = prices()
    classifier = make_classifier()
    cases = []
    for conv in conversations():
        state = analyze(conv.turns, classifier)  # Jev (com cache), sem resumo
        if state["dynamics"].key_turns:
            cases.append((conv.id, state))
    print(f"{len(cases)} conversas com momentos-chave, {len(models)} modelos\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    total_cost = 0.0
    summary_rows = []
    with OUT.open("a") as f:
        for model_id in models:
            llm = LLMSummarizer(chat_model(f"openrouter:{model_id}", **options))
            label = model_id + (" (sem raciocínio)" if no_reasoning else "")
            pin, pout = price.get(model_id, (0.0, 0.0))
            stats = []
            for cid, state in cases:
                messages = llm._messages(state["redacted"], state["readings"], state["dynamics"])
                t0 = time.perf_counter()
                try:
                    msg = llm.model.invoke(messages)
                except Exception as error:
                    print(f"  {model_id} {cid}: ERRO {type(error).__name__}: {str(error)[:160]}")
                    stats.append(None)
                    continue
                dt = time.perf_counter() - t0
                usage = msg.usage_metadata or {}
                tin, tout = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                cost = tin * pin + tout * pout
                total_cost += cost
                agent = {r.turn.index for r in state["readings"] if r.turn.speaker == "atendente"}
                customer = {r.turn.index for r in state["readings"] if r.turn.speaker == "cliente"}
                qc = check(msg.text, agent, customer)
                stats.append((dt, tin, tout, cost, qc))
                f.write(
                    json.dumps(
                        {
                            "modelo": label,
                            "conversa": cid,
                            "segundos": round(dt, 2),
                            "tokens_entrada": tin,
                            "tokens_saida": tout,
                            "custo_usd": cost,
                            **qc,
                            "texto": msg.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                print(
                    f"  {model_id:<42} {cid:<36} {dt:5.1f}s  {tout:>5} tok saída  ${cost:.5f}  {qc}"
                )
            ok = [s for s in stats if s]
            if ok:
                summary_rows.append(
                    (
                        label,
                        len(ok),
                        len(stats),
                        sorted(s[0] for s in ok)[len(ok) // 2],
                        sum(s[2] for s in ok) / len(ok),
                        sum(s[3] for s in ok) / len(ok) * 1000,
                        sum(s[4]["formato_ok"] for s in ok),
                        sum(bool(s[4]["cita_turno_do_cliente"]) for s in ok),
                        sum(bool(s[4]["palavras_de_causa"]) for s in ok),
                    )
                )
    print(
        f"\n{'modelo':<42} {'ok':>5} {'mediana':>8} {'tok saída':>10} {'US$/1000':>9} {'formato':>8} {'cita cliente':>13} {'causa':>6}"
    )
    for r in summary_rows:
        print(
            f"{r[0]:<42} {r[1]:>2}/{r[2]:<2} {r[3]:>7.1f}s {r[4]:>10.0f} {r[5]:>9.3f} {r[6]:>5}/{r[1]:<2} {r[7]:>10}/{r[1]:<2} {r[8]:>3}/{r[1]}"
        )
    print(f"\ncusto total do teste: US$ {total_cost:.4f}")


if __name__ == "__main__":
    main()
