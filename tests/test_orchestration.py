"""O que o LangGraph faz aqui: fan-out, junção, pausa para o supervisor, retomada, streaming."""

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest
from conftest import reading
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.types import Command

from sentiment_classifier.graph import (
    Review,
    build_graph,
    finalize,
    memory_checkpointer,
    run_config,
)
from sentiment_classifier.models import Turn, TurnFailure, TurnReading
from sentiment_classifier.parsing import parse_chat
from sentiment_classifier.summarize import LLMSummarizer

CHAT = """Cliente: Minha internet caiu de novo.
Atendente: Conforme o procedimento, reinicie o modem.
Cliente: Já fiz isso três vezes.
Atendente: Não posso fazer nada.
Cliente: Quero cancelar e falar com um supervisor."""

SENTIMENT = [3.0, 3.0, 2.0, 2.0, 1.0]


@dataclass
class Counting:
    """Lê cada turno; ``crash`` faz um turno levantar erro (uma vez) para simular queda."""

    crash: set[int] = field(default_factory=set)
    calls: list[int] = field(default_factory=list)
    threads: set[int] = field(default_factory=set)
    name: str = "contador"

    def read_one(self, turns: Sequence[Turn], index: int) -> TurnReading | TurnFailure:
        self.calls.append(index)
        self.threads.add(threading.get_ident())
        if index in self.crash:
            self.crash.discard(index)
            raise RuntimeError("processo caiu")
        flags = {"cancelamento": 0.95, "pede_escalada": 0.9} if index == 4 else {}
        return reading(index, turns[index].speaker, SENTIMENT[index], **flags)


def test_one_task_per_turn_and_results_in_turn_order() -> None:
    classifier = Counting()
    graph = build_graph(classifier)
    state = finalize(graph.invoke({"turns": parse_chat(CHAT)}, run_config()))
    assert sorted(classifier.calls) == [0, 1, 2, 3, 4]  # um Send por turno
    assert [r.turn.index for r in state["readings"]] == [0, 1, 2, 3, 4]
    assert state["csat"].source == "offline" and state["dynamics"].alerts


def test_supervisor_decides_before_escalating() -> None:
    graph = build_graph(Counting(), review=True, checkpointer=memory_checkpointer())
    config = run_config("conversa-1")
    paused = graph.invoke({"turns": parse_chat(CHAT)}, config)
    [pending] = paused["__interrupt__"]
    assert {a["tipo"] for a in pending.value["alertas"]} == {"cancelamento", "escalar"}
    assert graph.get_state(config).next == ("review",)

    done = graph.invoke(Command(resume={"escalar": True, "nota": "cliente ligou 3 vezes"}), config)
    assert done["review"] == Review(True, "cliente ligou 3 vezes")
    assert graph.get_state(config).next == ()


def test_no_alert_no_pause() -> None:
    calm = "Cliente: Oi, quero a segunda via.\nAtendente: Enviei agora.\nCliente: Obrigado!"
    graph = build_graph(Counting(), review=True, checkpointer=memory_checkpointer())
    done = graph.invoke({"turns": parse_chat(calm)}, run_config("calma"))
    assert "__interrupt__" not in done and done["review"] is None


def test_resume_after_a_crash_only_redoes_what_failed() -> None:
    classifier = Counting(crash={2})
    graph = build_graph(classifier, checkpointer=memory_checkpointer())
    config = run_config("lote-42")
    with pytest.raises(RuntimeError, match="processo caiu"):
        graph.invoke({"turns": parse_chat(CHAT)}, config)
    assert sorted(classifier.calls) == [0, 1, 2, 3, 4]

    classifier.calls.clear()
    state = finalize(graph.invoke(None, config))  # retoma do checkpoint
    assert classifier.calls == [2]  # os outros turnos já estavam salvos
    assert [r.turn.index for r in state["readings"]] == [0, 1, 2, 3, 4]


def test_llm_summary_streams_through_the_graph() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage("### Resumo\nCliente saiu irritado.")]))
    graph = build_graph(Counting(), LLMSummarizer(model))
    pieces = [
        chunk.content
        for chunk, meta in graph.stream(
            {"turns": parse_chat(CHAT)}, run_config(), stream_mode="messages"
        )
        if meta["langgraph_node"] == "summarize_llm"
    ]
    assert len(pieces) > 1  # chegou aos pedaços, não de uma vez
    assert "".join(pieces) == "### Resumo\nCliente saiu irritado."
