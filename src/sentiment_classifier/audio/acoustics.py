"""Sinais acústicos simples e determinísticos por turno.

Nada de modelo de emoção na voz: só medidas que dá para conferir no áudio.

- velocidade: palavras por segundo;
- volume: energia RMS em dB, medida só nos trechos com voz;
- tom: frequência fundamental mediana (algoritmo YIN), em Hz;
- silêncio antes de falar e interrupção (começou a falar antes do outro terminar).

Cada medida vira **relativa ao começo da ligação daquele falante** (mediana do primeiro terço
das falas dele, no mínimo uma): gente que fala alto sempre não parece irritada só por isso,
e quem passa a maior parte da ligação irritado não "vira normal" (o que aconteceria com a
mediana da ligação inteira). É assim que vai para o Jev.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

import numpy as np

from sentiment_classifier.audio.transcribe import SAMPLE_RATE, Transcript, Utterance
from sentiment_classifier.models import Speaker, Turn

_FRAME, _HOP = 1024, 256
_SILENCE_DB = -45.0


@dataclass(frozen=True)
class RawSignals:
    duration: float
    words: int
    rms_db: float | None
    f0_hz: float | None
    f0_range_st: float | None = None
    """Variação do tom: desvio-padrão da frequência fundamental, em semitons."""

    @property
    def rate(self) -> float:
        return self.words / max(self.duration, 0.3)


def measure(signal: np.ndarray, start: float, end: float, text: str) -> RawSignals:
    import librosa

    chunk = signal[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
    words = len(text.split())
    if len(chunk) < _FRAME:
        return RawSignals(end - start, words, None, None)
    rms = librosa.feature.rms(y=chunk, frame_length=_FRAME, hop_length=_HOP)[0]
    db = 20 * np.log10(rms + 1e-10)
    voiced = db > _SILENCE_DB
    if not voiced.any():
        return RawSignals(end - start, words, None, None)
    f0 = librosa.yin(chunk, fmin=65, fmax=400, sr=SAMPLE_RATE, frame_length=_FRAME, hop_length=_HOP)
    n = min(len(f0), len(voiced))
    f0_voiced = f0[:n][voiced[:n]]
    f0_median = float(np.median(f0_voiced)) if len(f0_voiced) else None
    f0_range = (
        float(np.std(12 * np.log2(f0_voiced / f0_median)))
        if f0_median and len(f0_voiced) > 3
        else None
    )
    return RawSignals(
        duration=end - start,
        words=words,
        rms_db=float(np.median(db[voiced])),
        f0_hz=f0_median,
        f0_range_st=f0_range,
    )


@dataclass(frozen=True)
class CallTurn:
    """Um turno da ligação: o texto, quando aconteceu e as medidas brutas."""

    speaker: Speaker
    text: str
    start: float
    end: float
    signals: RawSignals
    silence_before: float
    interrupted_by_s: float
    """Quantos segundos este turno começou antes de o outro terminar (0 se esperou)."""
    asr_logprob: float | None


def _merge(group: list[tuple[Utterance, RawSignals]]) -> RawSignals:
    duration = sum(s.duration for _, s in group)

    def weighted(values: list[tuple[float | None, float]]) -> float | None:
        known = [(v, w) for v, w in values if v is not None]
        total = sum(w for _, w in known)
        return sum(v * w for v, w in known) / total if known and total else None

    return RawSignals(
        duration=duration,
        words=sum(s.words for _, s in group),
        rms_db=weighted([(s.rms_db, s.duration) for _, s in group]),
        f0_hz=weighted([(s.f0_hz, s.duration) for _, s in group]),
        f0_range_st=weighted([(s.f0_range_st, s.duration) for _, s in group]),
    )


def call_turns(transcript: Transcript) -> list[CallTurn]:
    """Junta falas seguidas da mesma pessoa em turnos e mede cada um."""
    measured = [
        (u, measure(transcript.signal_for(u.speaker), u.start, u.end, u.text))
        for u in sorted(transcript.utterances, key=lambda u: u.start)
    ]
    groups: list[list[tuple[Utterance, RawSignals]]] = []
    for item in measured:
        if groups and groups[-1][-1][0].speaker == item[0].speaker:
            groups[-1].append(item)
        else:
            groups.append([item])

    turns: list[CallTurn] = []
    for group in groups:
        first = group[0][0]
        previous_end = turns[-1].end if turns else first.start
        gap = first.start - previous_end
        logprobs = [u.asr_logprob for u, _ in group if u.asr_logprob is not None]
        turns.append(
            CallTurn(
                speaker=first.speaker,  # type: ignore[arg-type]
                text=" ".join(u.text for u, _ in group),
                start=first.start,
                end=max(u.end for u, _ in group),
                signals=_merge(group),
                silence_before=max(gap, 0.0),
                interrupted_by_s=max(-gap, 0.0),
                asr_logprob=mean_or_none(logprobs),
            )
        )
    return turns


def mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


MIN_WORDS_FOR_RATE = 6
"""Abaixo disso, uma pausa entre duas palavras domina a conta e a velocidade não diz nada."""


def describe(
    features: dict[str, float | str],
    ref: str = "inicio",
    ref_words: str = "no começo da ligação",
    same: str = "voz parecida com a do começo da ligação",
) -> str:
    """Leitura em palavras dos números, com limiares fixos. Vai para o Jev junto com eles.

    ``ref`` é o sufixo das chaves (``volume_vs_<ref>_db``), ``ref_words`` diz em palavras
    contra o que se compara e ``same`` é o texto quando nada se destaca.
    """
    parts = []
    volume = float(features.get(f"volume_vs_{ref}_db", 0))
    if volume >= 4:
        parts.append(f"falou bem mais alto que {ref_words} (+{volume:.0f} dB)")
    elif volume >= 2:
        parts.append(f"falou um pouco mais alto que {ref_words}")
    elif volume <= -4:
        parts.append(f"falou bem mais baixo que {ref_words}")
    speed = float(features.get(f"velocidade_vs_{ref}_pct", 0))
    if speed >= 25:
        parts.append(f"falou bem mais rápido ({speed:+.0f}%)")
    elif speed <= -25:
        parts.append(f"falou bem mais devagar ({speed:+.0f}%)")
    pitch = float(features.get(f"tom_vs_{ref}_semitons", 0))
    if pitch >= 2:
        parts.append(f"voz mais aguda que {ref_words}")
    elif pitch <= -2:
        parts.append(f"voz mais grave que {ref_words}")
    pitch_range = float(features.get(f"variacao_tom_vs_{ref}_semitons", 0))
    if pitch_range <= -1:
        parts.append("entonação mais plana, monótona")
    elif pitch_range >= 1:
        parts.append("entonação mais exagerada, com mais sobe e desce")
    if float(features.get("interrompeu_o_outro_s", 0)) >= 0.3:
        parts.append("interrompeu o outro, começou a falar antes de ele terminar")
    if float(features.get("silencio_antes_s", 0)) >= 3:
        parts.append("demorou a responder")
    return "; ".join(parts) or same


def voice_features(turns: Sequence[CallTurn]) -> list[dict[str, float | str]]:
    """Medidas relativas ao começo da ligação de cada falante, no formato que vai para o Jev."""
    baselines: dict[str, dict[str, float | None]] = {}
    for speaker in {t.speaker for t in turns}:
        all_mine = [t.signals for t in turns if t.speaker == speaker]
        mine = all_mine[: max(1, len(all_mine) // 3)]  # o começo da ligação
        baselines[speaker] = {
            "rate": median(s.rate for s in mine),
            "rms": _median([s.rms_db for s in mine]),
            "f0": _median([s.f0_hz for s in mine]),
            "f0_range": _median([s.f0_range_st for s in mine]),
        }
    out = []
    for t in turns:
        base, s = baselines[t.speaker], t.signals
        features: dict[str, float | str] = {}
        if base["rate"] and s.words >= MIN_WORDS_FOR_RATE:
            features["velocidade_vs_inicio_pct"] = round((s.rate / base["rate"] - 1) * 100)
        if s.rms_db is not None and base["rms"] is not None:
            features["volume_vs_inicio_db"] = round(s.rms_db - base["rms"], 1)
        if s.f0_hz and base["f0"]:
            features["tom_vs_inicio_semitons"] = round(12 * math.log2(s.f0_hz / base["f0"]), 1)
        if s.f0_range_st is not None and base["f0_range"] is not None:
            features["variacao_tom_vs_inicio_semitons"] = round(s.f0_range_st - base["f0_range"], 1)
        features["silencio_antes_s"] = round(t.silence_before, 1) + 0.0  # sem "-0.0"
        features["interrompeu_o_outro_s"] = round(t.interrupted_by_s, 1) + 0.0
        features["leitura"] = describe(features)
        out.append(features)
    return out


def _median(values: Sequence[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return median(known) if known else None


def to_turns(call: Sequence[CallTurn]) -> list[Turn]:
    return [
        Turn(i, t.speaker, t.text, voice)
        for i, (t, voice) in enumerate(zip(call, voice_features(call), strict=True))
    ]
