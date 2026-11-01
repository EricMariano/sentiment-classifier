"""Transcrição com quem falou o quê.

- ``LocalStereoTranscriber``: ligação gravada em estéreo, um canal por pessoa (o padrão de
  muitas centrais). Transcreve cada canal com faster-whisper, localmente. O canal diz quem
  falou, então não há diarização a errar, e o áudio não sai da máquina.
- ``OpenAIDiarizedTranscriber``: áudio mono. Usa ``gpt-4o-transcribe-diarize`` da OpenAI,
  que separa os falantes (rótulos A, B...). **O áudio original, com dados pessoais falados,
  vai para a OpenAI antes de qualquer anonimização.** Só use se isso for aceitável.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Utterance:
    """Um trecho contínuo de fala de uma pessoa."""

    speaker: str
    """``cliente``/``atendente`` no estéreo; rótulo do diarizador (``A``, ``B``) no mono."""
    start: float
    end: float
    text: str
    asr_logprob: float | None = None
    """Confiança média da transcrição (log-probabilidade; mais perto de 0 é melhor)."""


@dataclass
class Transcript:
    utterances: list[Utterance]
    channels: dict[str, np.ndarray] = field(default_factory=dict)
    """Sinal de cada falante (estéreo) ou ``{"mix": ...}`` (mono), a 16 kHz."""
    roles_known: bool = True
    """``False`` quando os rótulos ainda precisam virar cliente/atendente."""

    def signal_for(self, speaker: str) -> np.ndarray:
        return self.channels.get(speaker, self.channels.get("mix", np.zeros(1)))


class Transcriber(Protocol):
    def transcribe(self, path: Path) -> Transcript: ...


def load_audio(path: Path) -> list[np.ndarray]:
    """Canais a 16 kHz (um item no mono, dois no estéreo). Lê wav, flac, ogg e mp3.

    Usa soundfile/libsndfile em vez do decodificador do faster-whisper (incompatível com o
    PyAV 19) e dispensa o ffmpeg.
    """
    import librosa
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    channels = [
        librosa.resample(audio[:, c], orig_sr=sr, target_sr=SAMPLE_RATE)
        for c in range(audio.shape[1])
    ]
    if len(channels) == 2 and np.allclose(channels[0], channels[1], atol=1e-4):
        return channels[:1]  # "estéreo" com o mesmo sinal nos dois lados
    if len(channels) > 2:
        raise ValueError(f"{len(channels)} canais; esperado mono ou estéreo")
    return channels


def _whisper(model_size: str) -> Any:
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    return BatchedInferencePipeline(WhisperModel(model_size, device="cpu", compute_type="int8"))


@dataclass
class LocalStereoTranscriber:
    agent_channel: Literal["esquerdo", "direito"] = "esquerdo"
    model_size: str = field(
        default_factory=lambda: os.environ.get("CONVERSA_WHISPER_MODEL", "small")
    )
    language: str = "pt"
    max_pause: float = 0.6
    """Pausa (s) entre palavras que separa duas falas no mesmo canal."""
    batch_size: int = 8
    _model: Any = None

    def transcribe(self, path: Path) -> Transcript:
        channels = load_audio(path)
        if len(channels) != 2:
            raise ValueError(
                "áudio mono: sem um canal por pessoa não dá para saber quem falou localmente. "
                "Use a gravação estéreo da central ou o modo OpenAI (--mono-openai)."
            )
        left, right = channels
        agent, customer = (left, right) if self.agent_channel == "esquerdo" else (right, left)
        self._model = self._model or _whisper(self.model_size)
        utterances = [
            u
            for speaker, signal in (("atendente", agent), ("cliente", customer))
            for u in self._channel(speaker, signal)
        ]
        utterances.sort(key=lambda u: u.start)
        return Transcript(utterances, {"atendente": agent, "cliente": customer})

    def _channel(self, speaker: str, signal: np.ndarray) -> list[Utterance]:
        # Em lotes: cada trecho de fala (achado pelo VAD) é transcrito à parte. Numa ligação
        # de 5 min foi 3,4x mais rápido e errou 3,8% das palavras do cliente, contra 45% do
        # modo sequencial, que se perdia ao correr o canal inteiro (ver README).
        segments, _ = self._model.transcribe(
            signal,
            language=self.language,
            batch_size=self.batch_size,
            vad_filter=True,  # o canal fica em silêncio enquanto o outro fala
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=True,
        )
        words = [(w, s.avg_logprob) for s in segments for w in (s.words or []) if w.word.strip()]
        return split_on_pauses(speaker, words, self.max_pause)


def split_on_pauses(
    speaker: str, words: Sequence[tuple[Any, float | None]], max_pause: float
) -> list[Utterance]:
    """Quebra em falas onde há pausa maior que ``max_pause`` entre duas palavras.

    O Whisper às vezes junta num segmento só duas falas separadas por um longo silêncio
    (o tempo em que o outro falava, no outro canal). As palavras têm tempo próprio.
    """
    out: list[Utterance] = []
    current: list[tuple[Any, float | None]] = []

    def flush() -> None:
        if current:
            logprobs = [lp for _, lp in current if lp is not None]
            out.append(
                Utterance(
                    speaker,
                    float(current[0][0].start),
                    float(current[-1][0].end),
                    "".join(w.word for w, _ in current).strip(),
                    sum(logprobs) / len(logprobs) if logprobs else None,
                )
            )
            current.clear()

    for word, logprob in words:
        if current and word.start - current[-1][0].end > max_pause:
            flush()
        current.append((word, logprob))
    flush()
    return out


@dataclass
class OpenAIDiarizedTranscriber:
    model: str = "gpt-4o-transcribe-diarize"
    client: Any = None

    def transcribe(self, path: Path) -> Transcript:
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI()
        with path.open("rb") as f:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=f,
                response_format="diarized_json",
                chunking_strategy="auto",
                language="pt",
            )
        return Transcript(
            parse_diarized(response.segments), {"mix": load_audio(path)[0]}, roles_known=False
        )


def _field(segment: Any, key: str) -> Any:
    return segment[key] if isinstance(segment, dict) else getattr(segment, key)


def parse_diarized(segments: Sequence[Any]) -> list[Utterance]:
    out = []
    for s in segments:
        text = str(_field(s, "text")).strip()
        if text:
            out.append(
                Utterance(
                    str(_field(s, "speaker")),
                    float(_field(s, "start")),
                    float(_field(s, "end")),
                    text,
                )
            )
    return sorted(out, key=lambda u: u.start)
