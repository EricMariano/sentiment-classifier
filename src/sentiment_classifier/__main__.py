"""Linha de comando.

python -m sentiment_classifier analisar conversa.txt
python -m sentiment_classifier lote pasta/ dados.jsonl --saida relatorio
python -m sentiment_classifier csat avaliar relatorio.jsonl
python -m sentiment_classifier csat treinar relatorio.jsonl --saida csat_model.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_typesafe import TypeSafeClassifier
from langgraph.types import Command

from sentiment_classifier.batch import load_conversations, read_report, run_batch, write_reports
from sentiment_classifier.classify import TurnClassifier
from sentiment_classifier.csat import FEATURES, CsatModel, evaluate
from sentiment_classifier.graph import (
    AnalysisState,
    Graph,
    build_graph,
    memory_checkpointer,
    run_config,
    sqlite_checkpointer,
    stream_analysis,
)
from sentiment_classifier.models import Turn
from sentiment_classifier.parsing import parse_chat
from sentiment_classifier.providers import (
    has_jev,
    load_csat_model,
    make_classifier,
    make_csat_estimator,
    make_summarizer,
)


def cmd_analisar(args: argparse.Namespace) -> None:
    report(parse_chat(args.arquivo.read_text()), args)


def cmd_ligacao(args: argparse.Namespace) -> None:
    from sentiment_classifier.audio.pipeline import build_call_graph
    from sentiment_classifier.audio.transcribe import (
        LocalStereoTranscriber,
        OpenAIDiarizedTranscriber,
    )

    transcriber = (
        OpenAIDiarizedTranscriber()
        if args.mono_openai
        else LocalStereoTranscriber(agent_channel=args.canal_atendente)
    )
    role_client = None if args.offline or not has_jev() else TypeSafeClassifier()
    classifier, analysis, review = analysis_graph(args)
    graph = build_call_graph(
        transcriber, analysis, role_client, memory_checkpointer() if review else None
    )
    request: dict[str, Any] = {"path": str(args.arquivo), "known_names": args.nome}
    if args.sem_voz:
        request["without_voice"] = True
        print("(sem sinais de voz: só o texto vai para o Jev)")
    run(graph, request, classifier.name)  # type: ignore[arg-type]


def analysis_graph(args: argparse.Namespace) -> tuple[TurnClassifier, Graph, bool]:
    classifier = make_classifier(args.offline)
    review = getattr(args, "revisar", False)
    graph = build_graph(
        classifier,
        make_summarizer(args.offline),
        make_csat_estimator(args.offline),
        load_csat_model(),
        review=review,
        checkpointer=memory_checkpointer() if review else None,
    )
    return classifier, graph, review


def print_analysis(state: AnalysisState, classifier_name: str) -> None:
    print(f"classificador: {classifier_name}")
    print(f"dados pessoais removidos: {state['redaction_counts'] or 'nenhum'}")
    if state["failures"]:
        print(f"turnos sem leitura: {[f.turn.index for f in state['failures']]}")
    print()
    for r in state["readings"]:
        flags = " ".join(f"{k}={v:.2f}" for k, v in r.flags.items() if v >= 0.5)
        print(f"[{r.turn.index:>2}] {r.turn.speaker:<9} {r.sentiment:.1f} {r.label:<12} {flags}")
    print()
    for a in state["dynamics"].alerts:
        p = f" (p={a.probability:.2f})" if a.probability is not None else ""
        print(f"ALERTA {a.kind} no turno {a.turn_index}{p}: {a.reason}")
    csat = state["csat"]
    print(f"\nsatisfação estimada ({csat.source}): {csat.value:.1f}")
    if state["csat_predicted"] is not None:
        print(f"satisfação pelo modelo calibrado: {state['csat_predicted']:.1f}")
    print()


def report(turns: list[Turn], args: argparse.Namespace) -> None:
    classifier, graph, _ = analysis_graph(args)
    run(graph, {"turns": turns, "known_names": args.nome}, classifier.name)


def run(graph: Graph, request: AnalysisState | Command[Any], classifier_name: str) -> None:
    """Roda o grafo em streaming: a análise do Jev aparece na hora, o resumo aos pedaços e,
    com ``--revisar``, o grafo pausa para você decidir se escala."""
    config = run_config("terminal")
    shown = streamed = False
    final: AnalysisState = {}
    while True:
        pending = None
        for event in stream_analysis(graph, request, config):
            if event.kind == "transcript":
                call = event.payload
                print(f"ligação de {call.duration:.0f}s, {len(call.turns)} turnos")
                if call.agent_confidence is not None:
                    print(
                        f"atendente identificado pelo Jev (confiança {call.agent_confidence:.2f})"
                    )
                for t, c in zip(call.turns, call.call, strict=True):
                    print(f"  {c.start:6.1f}s {t.speaker:<9} {t.text}")
                    print(f"          voz: {(t.voice or {}).get('leitura', '-')}")
                print()
            elif event.kind == "analysis" and not shown and event.state is not None:
                print_analysis(event.state, classifier_name)
                shown = True
            elif event.kind == "token":
                print(event.text, end="", flush=True)
                streamed = True
            elif event.kind == "interrupt":
                pending = event.payload
            elif event.kind == "done" and event.state is not None:
                final = event.state
        if pending is None:
            break
        if streamed:
            print()
        print(f"\n>>> {pending['pergunta']}")
        for a in pending["alertas"]:
            print(f"    {a['tipo']} no turno {a['turno']}: {a['motivo']}")
        answer = input("    escalar? [s/N] ").strip().lower()
        note = input("    observação (opcional): ").strip()
        request = Command(resume={"escalar": answer.startswith("s"), "nota": note})
    summary = final["summary"]
    if summary.note or not streamed:
        if summary.note:
            print(f"[{summary.note}]")
        print(summary.text)
    else:
        print()
    decision = final.get("review")
    if decision is not None:
        verdict = "ESCALAR" if decision.escalate else "não escalar"
        print(f"\nrevisão: {verdict}" + (f" ({decision.note})" if decision.note else ""))


def cmd_lote(args: argparse.Namespace) -> None:
    items = list(load_conversations(args.caminhos))
    if not items:
        sys.exit("nenhuma conversa encontrada")
    classifier = make_classifier(args.offline)
    print(f"{len(items)} conversas · {classifier.name}", file=sys.stderr)

    def progress(n: int, total: int, row: object) -> None:
        print(f"\r{n}/{total}", end="", file=sys.stderr, flush=True)

    stats: dict[str, int] = {}
    rows = run_batch(
        items,
        classifier,
        make_summarizer(args.offline) if args.com_llm else None,
        make_csat_estimator(args.offline),
        load_csat_model(),
        progress,
        args.paralelo,
        sqlite_checkpointer(str(args.retomar)) if args.retomar else None,
        stats,
    )
    csv_path, jsonl_path = write_reports(rows, args.saida)
    if args.retomar:
        print(
            f"\ncheckpoint: {stats['novas']} novas, {stats['retomadas']} retomadas, "
            f"{stats['prontas']} já prontas",
            file=sys.stderr,
        )
    print(file=sys.stderr)
    flagged = [r for r in rows if r.alertas or r.erro]
    print(f"\n{len(flagged)} de {len(rows)} conversas pedem atenção:\n")
    for r in flagged:
        what = f"ERRO {r.erro}" if r.erro else r.alertas
        print(f"  {r.id:<40} csat≈{r.csat_estimado or '?':<5} {what}")
    print(f"\ntabela: {csv_path}\ndetalhes: {jsonl_path}")


def _labeled(path: Path) -> tuple[list[dict[str, float]], list[float]]:
    rows = [r for r in read_report(path) if not r.erro and r.nota_real is not None and r.sinais]
    if len(rows) < 10:
        sys.exit(f"só {len(rows)} conversas com nota real e sem erro; são precisas pelo menos 10")
    return [r.sinais for r in rows], [float(r.nota_real) for r in rows if r.nota_real is not None]


def cmd_csat_avaliar(args: argparse.Namespace) -> None:
    signals, targets = _labeled(args.relatorio)
    print(f"{len(targets)} conversas com nota real\n")
    print(f"{'método':<32} {'MAE':>5} {'Spearman':>9} {'recall ≤2':>10} {'precisão ≤2':>12}")
    for name, m in evaluate(signals, targets).items():
        print(
            f"{name:<32} {m.mae:>5.2f} {m.spearman:>9.2f} "
            f"{m.recall_insatisfeito:>10.2f} {m.precisao_insatisfeito:>12.2f}"
        )


def cmd_csat_treinar(args: argparse.Namespace) -> None:
    signals, targets = _labeled(args.relatorio)
    model = CsatModel.fit(signals, targets)
    model.save(args.saida)
    print(f"modelo treinado com {len(targets)} conversas → {args.saida}")
    for f, w in sorted(zip(FEATURES, model.weights, strict=True), key=lambda x: -abs(x[1])):
        print(f"  {f:<20} {w:+.3f}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="sentiment_classifier")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("analisar", help="uma conversa, relatório detalhado")
    p.add_argument("arquivo", type=Path)
    p.add_argument("--offline", action="store_true", help="não chama Jev nem LLM")
    p.add_argument("--nome", action="append", default=[], help="nome conhecido a anonimizar")
    p.add_argument("--revisar", action="store_true", help="pausa para você decidir se escala")
    p.set_defaults(fn=cmd_analisar)

    p = sub.add_parser("ligacao", help="uma ligação gravada (requer o extra audio)")
    p.add_argument("arquivo", type=Path, help="wav, flac, ogg ou mp3")
    p.add_argument("--canal-atendente", choices=["esquerdo", "direito"], default="esquerdo")
    p.add_argument(
        "--mono-openai",
        action="store_true",
        help="áudio mono: transcreve e separa falantes na OpenAI (o áudio sai da máquina)",
    )
    p.add_argument("--sem-voz", action="store_true", help="ignora os sinais de voz (comparação)")
    p.add_argument("--revisar", action="store_true", help="pausa para você decidir se escala")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--nome", action="append", default=[])
    p.set_defaults(fn=cmd_ligacao)

    p = sub.add_parser("lote", help="várias conversas, tabela de alertas")
    p.add_argument("caminhos", type=Path, nargs="+", help="pastas com .txt e/ou arquivos .jsonl")
    p.add_argument("--saida", type=Path, default=Path("relatorio"))
    p.add_argument("--offline", action="store_true")
    p.add_argument(
        "--com-llm", action="store_true", help="resumo com LLM nas conversas com virada ou alerta"
    )
    p.add_argument("--paralelo", type=int, default=4, help="conversas analisadas ao mesmo tempo")
    p.add_argument(
        "--retomar",
        type=Path,
        metavar="CHECKPOINT.sqlite",
        help="salva cada passo em disco; rodando de novo, retoma de onde parou",
    )
    p.set_defaults(fn=cmd_lote)

    csat = sub.add_parser("csat", help="avaliar ou treinar a nota de satisfação").add_subparsers(
        required=True
    )
    p = csat.add_parser("avaliar", help="compara métodos contra as notas reais (sem chamar o Jev)")
    p.add_argument("relatorio", type=Path, help="o .jsonl gerado pelo modo lote")
    p.set_defaults(fn=cmd_csat_avaliar)
    p = csat.add_parser("treinar", help="ajusta o modelo calibrado")
    p.add_argument("relatorio", type=Path)
    p.add_argument("--saida", type=Path, default=Path("csat_model.json"))
    p.set_defaults(fn=cmd_csat_treinar)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
