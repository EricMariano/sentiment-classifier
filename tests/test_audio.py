"""Áudio sem rede e sem modelo: sinais sintéticos e falas montadas à mão."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

pytest.importorskip("librosa")

from sentiment_classifier.audio.acoustics import call_turns, describe, measure, to_turns  # noqa: E402
from sentiment_classifier.audio.roles import agent_label, assign_roles  # noqa: E402
from sentiment_classifier.audio.transcribe import (  # noqa: E402
    SAMPLE_RATE,
    Transcript,
    Utterance,
    parse_diarized,
    split_on_pauses,
)


def tone(seconds: float, hz: float, amplitude: float) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype("float32")


def test_measure_reads_volume_and_pitch() -> None:
    quiet = measure(tone(1.0, 150, 0.05), 0, 1.0, "um dois três")
    loud = measure(tone(1.0, 150, 0.2), 0, 1.0, "um dois três")
    assert loud.rms_db is not None and quiet.rms_db is not None
    assert loud.rms_db - quiet.rms_db == pytest.approx(12.0, abs=0.5)  # 4× amplitude ≈ +12 dB
    assert loud.f0_hz == pytest.approx(150, rel=0.03)
    assert loud.rate == 3.0


def test_silence_has_no_volume_or_pitch() -> None:
    s = measure(np.zeros(SAMPLE_RATE, "float32"), 0, 1.0, "nada")
    assert (s.rms_db, s.f0_hz) == (None, None)


def _call() -> Transcript:
    """Atendente calmo; cliente começa a 0,05 e termina a 0,2, interrompendo."""
    agent = np.zeros(10 * SAMPLE_RATE, "float32")
    customer = np.zeros(10 * SAMPLE_RATE, "float32")
    agent[0 : 2 * SAMPLE_RATE] = tone(2, 200, 0.1)
    customer[int(2.5 * SAMPLE_RATE) : int(4.5 * SAMPLE_RATE)] = tone(2, 120, 0.05)
    agent[5 * SAMPLE_RATE : 7 * SAMPLE_RATE] = tone(2, 200, 0.1)
    customer[int(6.5 * SAMPLE_RATE) : int(8.5 * SAMPLE_RATE)] = tone(2, 160, 0.2)
    words = "uma frase com várias palavras ditas aqui"
    return Transcript(
        [
            Utterance("atendente", 0, 2, "Bom dia, em que posso ajudar?"),
            Utterance("cliente", 2.5, 4.5, words),
            Utterance("atendente", 5, 7, "Vou verificar o seu pedido agora."),
            Utterance("cliente", 6.5, 8.5, words + " de novo"),
        ],
        {"atendente": agent, "cliente": customer},
    )


def test_voice_is_relative_to_the_start_of_the_call() -> None:
    turns = to_turns(call_turns(_call()))
    assert [t.speaker for t in turns] == ["atendente", "cliente", "atendente", "cliente"]
    first, last = turns[1].voice, turns[3].voice
    assert first is not None and last is not None
    assert first["volume_vs_inicio_db"] == 0.0  # o começo é a referência
    assert last["volume_vs_inicio_db"] == pytest.approx(12.0, abs=0.5)
    assert last["tom_vs_inicio_semitons"] == pytest.approx(12 * np.log2(160 / 120), abs=0.3)
    assert last["interrompeu_o_outro_s"] == 0.5
    assert first["silencio_antes_s"] == 0.5
    assert "bem mais alto" in str(last["leitura"]) and "interrompeu" in str(last["leitura"])


def test_consecutive_utterances_of_the_same_speaker_merge() -> None:
    t = _call()
    t.utterances.insert(2, Utterance("cliente", 4.6, 4.9, "só isso"))
    turns = call_turns(t)
    assert len(turns) == 4 and turns[1].text.endswith("só isso") and turns[1].end == 4.9


def test_short_utterances_get_no_speed() -> None:
    t = _call()
    t.utterances[3] = Utterance("cliente", 6.5, 8.5, "Ótimo. Obrigado.")
    voice = to_turns(call_turns(t))[3].voice
    assert voice is not None and "velocidade_vs_inicio_pct" not in voice


def test_describe_thresholds() -> None:
    assert describe({}) == "voz parecida com a do começo da ligação"
    text = describe(
        {"volume_vs_inicio_db": 2.5, "velocidade_vs_inicio_pct": -30, "silencio_antes_s": 4}
    )
    assert text == (
        "falou um pouco mais alto que no começo da ligação; falou bem mais devagar (-30%); "
        "demorou a responder"
    )
    other = describe(
        {"tom_vs_normal_semitons": -2.5, "variacao_tom_vs_normal_semitons": -1.2},
        ref="normal",
        ref_words="o normal dele",
        same="nada fora do normal",
    )
    assert other == "voz mais grave que o normal dele; entonação mais plana, monótona"
    assert describe({}, ref="normal", same="nada fora do normal") == "nada fora do normal"


def test_pitch_range_separates_flat_from_lively_speech() -> None:
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    hz = 150 * 2 ** (3 * np.sin(2 * np.pi * 2 * t) / 12)  # sobe e desce ±3 semitons
    lively = (0.1 * np.sin(2 * np.pi * np.cumsum(hz) / SAMPLE_RATE)).astype("float32")
    flat = measure(tone(1.0, 150, 0.1), 0, 1.0, "a b c")
    wavy = measure(lively, 0, 1.0, "a b c")
    assert flat.f0_range_st is not None and wavy.f0_range_st is not None
    assert flat.f0_range_st < 0.3 < 1.5 < wavy.f0_range_st


@dataclass
class W:
    word: str
    start: float
    end: float


def test_split_on_pauses() -> None:
    words = [
        (W(" Bom", 0.0, 0.3), -0.1),
        (W(" dia.", 0.35, 0.6), -0.3),
        (W(" Ótimo.", 5.0, 5.4), -0.2),
    ]
    out = split_on_pauses("cliente", words, max_pause=0.6)
    assert [(u.start, u.end, u.text) for u in out] == [(0.0, 0.6, "Bom dia."), (5.0, 5.4, "Ótimo.")]
    assert out[0].asr_logprob == pytest.approx(-0.2)


def test_parse_diarized_accepts_objects_and_dicts() -> None:
    @dataclass
    class Seg:
        speaker: str
        start: float
        end: float
        text: str

    out = parse_diarized(
        [
            Seg("B", 2, 3, " oi "),
            {"speaker": "A", "start": 0, "end": 1, "text": "olá"},
            {"speaker": "A", "start": 4, "end": 5, "text": "  "},
        ]
    )
    assert [(u.speaker, u.text) for u in out] == [("A", "olá"), ("B", "oi")]


class FakeRoleClient:
    def __init__(self) -> None:
        self.request: Any = None

    def invoke(self, request: Any) -> Any:
        from langchain_typesafe import ClassifierResponse

        self.request = request
        return ClassifierResponse.model_validate(
            {
                "model": "t",
                "answers": {
                    "atendente": {
                        "type": "choice",
                        "choice": "B",
                        "probabilities": {"A": 0.1, "B": 0.9},
                        "confidence": 0.8,
                    }
                },
            }
        )


def test_roles_from_jev_with_redacted_openings() -> None:
    utterances = [
        Utterance("A", 0, 1, "Oi, meu CPF é 123.456.789-09"),
        Utterance("B", 1, 2, "Central X, bom dia"),
    ]
    client = FakeRoleClient()
    label, confidence = agent_label(utterances, client)  # type: ignore[arg-type]
    assert (label, confidence) == ("B", 0.8)
    criteria = client.request["questions"]["atendente"].criteria
    assert criteria["A"] == "Oi, meu CPF é [CPF]"  # anonimizado antes de sair
    assert [u.speaker for u in assign_roles(utterances, label)] == ["cliente", "atendente"]


def test_roles_without_jev_assume_who_opens_is_the_agent() -> None:
    utterances = [Utterance("A", 0, 1, "Central X"), Utterance("B", 1, 2, "Oi")]
    assert agent_label(utterances, None) == ("A", None)
