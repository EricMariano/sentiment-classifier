import json
from pathlib import Path

from sentiment_classifier.batch import load_conversations, read_report, run_batch, write_reports
from sentiment_classifier.classify import OfflineHeuristicClassifier
from sentiment_classifier.models import ReadResult, Turn


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

    def read(self, turns: list[Turn]) -> ReadResult:
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
