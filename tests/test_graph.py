from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest
from conftest import reading

from sentiment_classifier.classify import GaveUp
from sentiment_classifier.csat import CsatEstimate, CsatModel
from sentiment_classifier.graph import analyze
from sentiment_classifier.models import Dynamics, Turn, TurnFailure, TurnReading
from sentiment_classifier.parsing import parse_chat
from sentiment_classifier.summarize import Summary, SummaryUnavailable, build_prompt


@dataclass
class ScriptedClassifier:
    sentiments: list[float]
    flags: dict[int, dict[str, float]] = field(default_factory=dict)
    name: str = "scripted"
    seen: list[Turn] = field(default_factory=list)
    fail: set[int] = field(default_factory=set)

    def read_one(self, turns: Sequence[Turn], index: int) -> TurnReading | TurnFailure:
        t = turns[index]
        self.seen.append(t)
        if t.index in self.fail:
            return TurnFailure(t, "rede", 4)
        return reading(
            t.index, t.speaker, self.sentiments[t.index], text=t.text, **self.flags.get(t.index, {})
        )


@dataclass
class RecordingSummarizer:
    calls: list[str] = field(default_factory=list)

    def summarize(
        self, turns: Sequence[Turn], readings: Sequence[TurnReading], dynamics: Dynamics
    ) -> Summary:
        self.calls.append(build_prompt(turns, readings, dynamics))
        return Summary("resumo do LLM", "llm", len(dynamics.key_turns))


CHAT = """Cliente: Oi, meu CPF é 123.456.789-09 e meu nome é Ana Lima.
Atendente: Olá Ana, como posso ajudar?
Cliente: A fatura veio errada.
Atendente: Não posso fazer nada, é o sistema.
Cliente: Absurdo. Vou cancelar.
Atendente: Ok.
Cliente: Tchau."""


def test_classifier_only_sees_redacted_text() -> None:
    classifier = ScriptedClassifier([3, 3, 3, 3, 3, 3, 3])
    state = analyze(parse_chat(CHAT), classifier, RecordingSummarizer(), known_names=["Ana"])
    sent = " ".join(t.text for t in classifier.seen)
    assert "123.456" not in sent and "Ana" not in sent
    assert state["redaction_counts"] == {"CPF": 1, "NOME": 2}


def test_stable_conversation_skips_the_llm() -> None:
    llm = RecordingSummarizer()
    state = analyze(parse_chat(CHAT), ScriptedClassifier([3, 3, 3, 3, 3.5, 3, 3]), llm)
    assert llm.calls == []
    assert state["summary"].source == "template"
    assert "LLM não foi chamado" in state["summary"].text


def test_llm_gets_only_key_turns() -> None:
    llm = RecordingSummarizer()
    classifier = ScriptedClassifier([3, 3, 3, 2, 1, 3, 1], flags={4: {"cancelamento": 0.95}})
    state = analyze(parse_chat(CHAT), classifier, llm)
    assert state["summary"].source == "llm"
    assert len(llm.calls) == 1
    prompt = llm.calls[0]
    # virada entre os turnos 2 e 4 (3 → 1), cancelamento no 4, queda 3 → 1 pede escalada no 6
    assert state["dynamics"].key_turns == [2, 3, 4, 5, 6]
    assert "[3] ATENDENTE" in prompt and "[4] CLIENTE" in prompt
    assert "[0]" not in prompt and "[1]" not in prompt
    assert "cancelamento no turno 4 (p=0.95)" in prompt
    assert "escalar no turno 6: sentimento do cliente caiu de 3.0 para 1.0" in prompt


class BrokenSummarizer:
    def summarize(self, *args: object) -> Summary:
        try:
            raise TimeoutError("provedor fora")
        except TimeoutError as error:
            raise SummaryUnavailable(str(error)) from error


class BuggySummarizer:
    def summarize(self, *args: object) -> Summary:
        raise KeyError("bug nosso")


def test_bugs_in_the_summarizer_are_not_hidden() -> None:
    with pytest.raises(KeyError):
        analyze(parse_chat(CHAT), ScriptedClassifier([3, 3, 3, 2, 1, 3, 1]), BuggySummarizer())


def test_llm_failure_falls_back_to_template_with_a_note() -> None:
    classifier = ScriptedClassifier([3, 3, 3, 2, 1, 3, 1])
    state = analyze(parse_chat(CHAT), classifier, BrokenSummarizer())
    assert state["summary"].source == "template"
    assert state["summary"].note == "LLM indisponível (TimeoutError); resumo montado dos sinais."


def test_unread_turns_leave_a_gap_and_are_reported() -> None:
    llm = RecordingSummarizer()
    classifier = ScriptedClassifier([3, 3, 3, 2, 1, 3, 1], fail={2})
    state = analyze(parse_chat(CHAT), classifier, llm)
    dyn = state["dynamics"]
    assert [i for i, _ in dyn.curves["cliente"]] == [0, 4, 6]
    assert dyn.unread_turns == [2]
    [tp] = dyn.turning_points  # 0 → 4 atravessa a lacuna
    assert (tp.customer_before, tp.customer_after, tp.missing_turns) == (0, 4, (2,))
    assert "Turnos sem leitura (falha técnica, ignore-os): [2]" in llm.calls[0]


@dataclass
class FixedCsat:
    value: float = 4.2
    fail: bool = False

    def estimate(self, turns: Sequence[Turn]) -> CsatEstimate:
        if self.fail:
            raise GaveUp(ConnectionError("rede"), 4)
        return CsatEstimate(self.value, "jev", {4: 0.8, 5: 0.2}, 0.6)


def test_csat_estimate_and_calibrated_prediction() -> None:
    classifier = ScriptedClassifier([3, 3, 3, 3, 3.5, 3, 3])
    model = CsatModel([1.0] + [0.0] * 9, 3.0, [3.0] * 10, [1.0] * 10, 50)
    state = analyze(parse_chat(CHAT), classifier, csat_estimator=FixedCsat(), csat_model=model)
    assert state["csat"].source == "jev" and state["csat"].value == 4.2
    assert state["csat_features"]["jev_csat"] == 4.2
    assert state["csat_predicted"] == pytest.approx(4.2)  # bias 3 + 1·(4.2 − 3)


def test_csat_failure_keeps_the_analysis_with_the_offline_value() -> None:
    classifier = ScriptedClassifier([3, 3, 3, 3, 3.5, 3, 2.5])
    state = analyze(parse_chat(CHAT), classifier, csat_estimator=FixedCsat(fail=True))
    assert (state["csat"].source, state["csat"].value) == ("offline", 2.5)
