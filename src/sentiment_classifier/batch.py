"""Modo em lote: várias conversas, uma tabela de alertas no fim.

Entradas aceitas:

- uma pasta com arquivos ``.txt`` (uma conversa por arquivo, formato ``Cliente:``);
- arquivos ``.jsonl``, uma conversa por linha: ``{"id": ..., "messages": [{"role",
  "content"}], "nota": 4}``. ``nota`` (opcional) é a nota real da pesquisa, usada para
  treinar e avaliar o modelo de satisfação.

Uma conversa que falha vira uma linha com ``erro`` e o lote continua.
"""

from __future__ import annotations

import csv
import json
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langgraph.types import Checkpointer

from sentiment_classifier.anonymize import redact_turns
from sentiment_classifier.classify import TurnClassifier
from sentiment_classifier.csat import CsatEstimator, CsatModel
from sentiment_classifier.graph import AnalysisState, build_graph, finalize, run_config
from sentiment_classifier.models import Turn
from sentiment_classifier.parsing import parse_chat, turns_from_messages
from sentiment_classifier.summarize import Summarizer


@dataclass(frozen=True)
class Conversation:
    id: str
    turns: list[Turn]
    rating: float | None = None


def load_conversations(paths: Sequence[Path]) -> Iterator[Conversation | tuple[str, str]]:
    """Gera conversas; um item ilegível vem como ``(id, erro)`` para não parar o lote."""
    for path in paths:
        if path.is_dir():
            yield from load_conversations(
                sorted(p for p in path.iterdir() if p.suffix in {".txt", ".jsonl"})
            )
        elif path.suffix == ".jsonl":
            for n, line in enumerate(path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                cid = f"{path.name}:{n}"
                try:
                    row = json.loads(line)
                    cid = str(row.get("id", cid))
                    rating = row.get("nota")
                    yield Conversation(
                        cid,
                        turns_from_messages(row["messages"]),
                        float(rating) if rating is not None else None,
                    )
                except (ValueError, KeyError, TypeError) as error:
                    yield cid, f"{type(error).__name__}: {error}"
        else:
            try:
                yield Conversation(path.name, parse_chat(path.read_text()))
            except ValueError as error:
                yield path.name, f"ValueError: {error}"


@dataclass
class BatchRow:
    id: str
    turnos: int = 0
    cliente_inicio: float | None = None
    cliente_fim: float | None = None
    csat_estimado: float | None = None
    csat_fonte: str = ""
    csat_calibrado: float | None = None
    nota_real: float | None = None
    viradas_piora: int = 0
    alertas: str = ""
    cancelamento: bool = False
    escalar: bool = False
    sarcasmo: bool = False
    turnos_sem_leitura: int = 0
    resumo: str = ""
    erro: str = ""
    sinais: dict[str, float] = field(default_factory=dict)
    """Sinais usados pelo modelo de satisfação (vão para o JSONL, não para o CSV)."""

    @property
    def priority(self) -> tuple[int, float]:
        """Ordem da tabela: erros e alertas primeiro, depois a pior nota."""
        alerts = self.cancelamento + self.escalar + self.sarcasmo
        return (0 if self.erro else 1 if alerts else 2, self.csat_estimado or 3.0)


def _row(conv: Conversation, state: AnalysisState) -> BatchRow:
    dyn = state["dynamics"]
    kinds = sorted({a.kind for a in dyn.alerts})
    return BatchRow(
        id=conv.id,
        turnos=len(conv.turns),
        cliente_inicio=dyn.customer_start,
        cliente_fim=dyn.customer_end,
        csat_estimado=round(state["csat"].value, 2),
        csat_fonte=state["csat"].source,
        csat_calibrado=(
            round(state["csat_predicted"], 2) if state["csat_predicted"] is not None else None
        ),
        nota_real=conv.rating,
        viradas_piora=sum(1 for t in dyn.turning_points if t.delta < 0),
        alertas=", ".join(kinds),
        cancelamento="cancelamento" in kinds,
        escalar="escalar" in kinds,
        sarcasmo="sarcasmo" in kinds,
        turnos_sem_leitura=len(dyn.unread_turns),
        resumo=state["summary"].text if state["summary"].source == "llm" else "",
        sinais=state["csat_features"],
    )


def run_batch(
    items: Sequence[Conversation | tuple[str, str]],
    classifier: TurnClassifier,
    summarizer: Summarizer | None = None,
    csat_estimator: CsatEstimator | None = None,
    csat_model: CsatModel | None = None,
    on_progress: Callable[[int, int, BatchRow], None] | None = None,
    max_conversations: int = 4,
    checkpointer: Checkpointer = None,
    stats: dict[str, int] | None = None,
) -> list[BatchRow]:
    """Analisa ``max_conversations`` conversas ao mesmo tempo, num grafo só.

    Cada conversa já lê seus turnos em paralelo (fan-out no grafo), mas é curta: sem isso,
    o lote passaria a maior parte do tempo esperando a última requisição de uma conversa.

    Com ``checkpointer``, cada conversa é uma *thread* do LangGraph (o id dela): numa nova
    execução, as conversas terminadas vêm prontas do checkpoint e as que pararam no meio
    retomam do último passo salvo. As conversas são anonimizadas **antes** de entrar no
    grafo, para que o checkpoint em disco não guarde dado pessoal. ``stats`` recebe quantas
    foram ``novas``, ``retomadas`` e ``prontas``.
    """
    graph = build_graph(
        classifier, summarizer, csat_estimator, csat_model, checkpointer=checkpointer
    )
    counts = stats if stats is not None else {}
    for key in ("novas", "retomadas", "prontas"):
        counts.setdefault(key, 0)
    lock = threading.Lock()

    def count(key: str) -> None:
        with lock:
            counts[key] += 1

    def one(item: Conversation | tuple[str, str]) -> BatchRow:
        if isinstance(item, tuple):
            return BatchRow(id=item[0], erro=item[1])
        try:
            if checkpointer is None:
                state = graph.invoke({"turns": item.turns}, run_config())
                count("novas")
                return _row(item, finalize(state))
            config = run_config(item.id)
            snapshot = graph.get_state(config)
            if snapshot.values and not snapshot.next:
                count("prontas")
                return _row(item, finalize(snapshot.values))
            if snapshot.values:
                count("retomadas")
                return _row(item, finalize(graph.invoke(None, config)))
            count("novas")
            turns = redact_turns(item.turns).turns  # o checkpoint não guarda dado pessoal
            return _row(item, finalize(graph.invoke({"turns": turns}, config)))
        except Exception as error:  # uma conversa ruim não derruba o lote
            return BatchRow(
                id=item.id,
                turnos=len(item.turns),
                nota_real=item.rating,
                erro=f"{type(error).__name__}: {error}",
            )

    rows: list[BatchRow] = []
    with ThreadPoolExecutor(max(1, max_conversations)) as pool:
        futures = [pool.submit(one, item) for item in items]
        for n, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            if on_progress:
                on_progress(n, len(items), row)
    return sorted(rows, key=lambda r: (r.priority, r.id))


CSV_COLUMNS = [f for f in BatchRow.__dataclass_fields__ if f not in {"sinais", "resumo"}]


def write_reports(rows: Sequence[BatchRow], out: Path) -> tuple[Path, Path]:
    """Grava ``<out>.csv`` (a tabela) e ``<out>.jsonl`` (tudo, inclusive sinais e resumos)."""
    csv_path, jsonl_path = out.with_suffix(".csv"), out.with_suffix(".jsonl")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
    with jsonl_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    return csv_path, jsonl_path


def read_report(path: Path) -> list[BatchRow]:
    return [BatchRow(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
