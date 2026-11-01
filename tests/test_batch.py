import json
from pathlib import Path
from typing import Any

from sentiment_classifier.batch import (
    Conversation,
    load_conversations,
    read_report,
    run_batch,
    write_reports,
)
from sentiment_classifier.classify import OfflineHeuristicClassifier
from sentiment_classifier.models import Turn
from sentiment_classifier.parsing import parse_chat


def _write(tmp: Path) -> Path:
    folder = tmp / "conversas"
    folder.mkdir()
    (folder / "ok.txt").write_text("Cliente: Quero cancelar, que absurdo!\nAtendente: Ok.")
    (folder / "ruim.txt").write_text("sem rótulo nenhum")
    (folder / "dados.jsonl").write_text(
        json.dumps(
            {
                "id": "j1",
                "nota": 4,
                "messages": [
                    {"role": "customer", "content": "Obrigado, resolveu!"},
                    {"role": "agent", "content": "Disponha."},
                ],
            }
        )
        + "\n"
        + "{quebrado\n"
    )
    return folder


def test_bad_items_become_error_rows_and_the_batch_continues(tmp_path: Path) -> None:
    items = list(load_conversations([_write(tmp_path)]))
    rows = run_batch(items, OfflineHeuristicClassifier())
    by_id = {r.id: r for r in rows}
    assert set(by_id) == {"ok.txt", "ruim.txt", "j1", "dados.jsonl:2"}
    assert by_id["ruim.txt"].erro.startswith("ValueError")
    assert by_id["dados.jsonl:2"].erro.startswith("JSONDecodeError")
    assert by_id["ok.txt"].cancelamento and by_id["ok.txt"].alertas == "cancelamento"
    assert by_id["j1"].nota_real == 4.0 and by_id["j1"].sinais["cliente_fim"] == 5.0
    # erros primeiro, depois quem tem alerta, depois o resto
    assert [r.id for r in rows][2:] == ["ok.txt", "j1"]


class ExplodingClassifier:
    name = "explode"

    def read_one(self, turns: list[Turn], index: int) -> Any:
        raise RuntimeError("provedor fora")


def test_provider_failure_in_one_conversation_is_reported(tmp_path: Path) -> None:
    items = list(load_conversations([_write(tmp_path)]))
    rows = run_batch(items, ExplodingClassifier())  # type: ignore[arg-type]
    assert {r.erro for r in rows if r.id == "ok.txt"} == {"RuntimeError: provedor fora"}


def test_reports_round_trip(tmp_path: Path) -> None:
    rows = run_batch(list(load_conversations([_write(tmp_path)])), OfflineHeuristicClassifier())
    csv_path, jsonl_path = write_reports(rows, tmp_path / "rel")
    header = csv_path.read_text().splitlines()[0]
    assert "sinais" not in header and header.startswith("id,turnos")
    assert [r.id for r in read_report(jsonl_path)] == [r.id for r in rows]


class CrashOnce(OfflineHeuristicClassifier):
    """Cai uma vez ao ler a conversa que menciona 'modem', como um processo interrompido."""

    def __init__(self) -> None:
        super().__init__()
        self.crashed = False
        self.reads: list[str] = []

    def read_one(self, turns: Any, index: int) -> Any:
        self.reads.append(turns[0].text[:12])
        if "modem" in turns[index].text and not self.crashed:
            self.crashed = True
            raise RuntimeError("queda no meio do lote")
        return super().read_one(turns, index)


def test_batch_resumes_from_checkpoint_without_storing_personal_data(tmp_path: Path) -> None:
    from sentiment_classifier.graph import sqlite_checkpointer

    convs = [
        Conversation(
            "a", parse_chat("Cliente: Meu CPF é 123.456.789-09, obrigado!\nAtendente: Ok.")
        ),
        Conversation("b", parse_chat("Cliente: O modem caiu.\nAtendente: Vou verificar.")),
    ]
    db = tmp_path / "lote.sqlite"
    classifier = CrashOnce()

    first_stats: dict[str, int] = {}
    first = run_batch(
        convs, classifier, checkpointer=sqlite_checkpointer(str(db)), stats=first_stats
    )
    assert {r.id: bool(r.erro) for r in first} == {"a": False, "b": True}

    classifier.reads.clear()
    again: dict[str, int] = {}
    second = run_batch(convs, classifier, checkpointer=sqlite_checkpointer(str(db)), stats=again)
    assert all(not r.erro for r in second)
    assert again == {"novas": 0, "retomadas": 1, "prontas": 1}
    assert set(classifier.reads) == {"O modem caiu"}  # só a conversa que tinha parado

    assert b"123.456.789-09" not in db.read_bytes()  # anonimizado antes do checkpoint
