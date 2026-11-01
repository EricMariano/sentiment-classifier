"""Prepara o MUStARD: áudio de cada fala-alvo + sinais de voz relativos ao personagem.

MUStARD (Castro et al., ACL 2019, https://github.com/soujanyaporia/MUStARD, MIT): 690 falas
de séries de TV em inglês, metade sarcásticas, com as falas anteriores como contexto e o
vídeo da fala. Usamos só o áudio da fala-alvo.

Os sinais (volume, tom, variação do tom, velocidade) são relativos ao **normal daquele
personagem** no dataset inteiro (mediana das falas dele), porque cada item é uma fala solta,
sem "começo da ligação". O rótulo de sarcasmo não é usado aqui.

Limites conhecidos, fixados antes de rodar o Jev:
- o clipe pode ter risada de plateia e silêncio nas bordas, que entram no volume e na
  velocidade (palavras / duração do clipe);
- "PERSON" (45 falas) junta personagens diferentes; a referência dele é a da série.

    python benchmarks/mustard/prepare.py            # extrai, mede e apaga o zip
"""

from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from statistics import median

import av
import numpy as np
import soundfile as sf

from sentiment_classifier.audio.acoustics import MIN_WORDS_FOR_RATE, RawSignals, describe, measure
from sentiment_classifier.audio.transcribe import SAMPLE_RATE

DATA = Path(__file__).resolve().parents[1] / "data" / "mustard"
REF_WORDS = "o normal deste personagem"
SAME = "voz dentro do normal deste personagem"


def decode(blob: bytes) -> np.ndarray:
    """Trilha de áudio de um mp4, mono a 16 kHz, sem ffmpeg."""
    resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    with av.open(io.BytesIO(blob)) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
    for out in resampler.resample(None):
        chunks.append(out.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype("float32")


def extract(zip_path: Path, audio_dir: Path, ids: set[str]) -> None:
    audio_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.startswith("utterances_final/") or not name.endswith(".mp4"):
                continue
            uid = Path(name).stem
            out = audio_dir / f"{uid}.flac"
            if uid in ids and not out.exists():
                sf.write(out, decode(z.read(name)), SAMPLE_RATE)


def relative(s: RawSignals, base: dict[str, float | None]) -> dict[str, float | str]:
    f: dict[str, float | str] = {}
    if base["rate"] and s.words >= MIN_WORDS_FOR_RATE:
        f["velocidade_vs_normal_pct"] = round((s.rate / base["rate"] - 1) * 100)
    if s.rms_db is not None and base["rms"] is not None:
        f["volume_vs_normal_db"] = round(s.rms_db - base["rms"], 1)
    if s.f0_hz and base["f0"]:
        f["tom_vs_normal_semitons"] = round(12 * math.log2(s.f0_hz / base["f0"]), 1)
    if s.f0_range_st is not None and base["f0_range"] is not None:
        f["variacao_tom_vs_normal_semitons"] = round(s.f0_range_st - base["f0_range"], 1)
    f["leitura"] = describe(f, ref="normal", ref_words=REF_WORDS, same=SAME)
    return f


def _median(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return median(known) if known else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manter-zip", action="store_true")
    args = parser.parse_args()

    labels = json.loads((DATA / "sarcasm_data.json").read_text())
    zip_path, audio_dir = DATA / "mmsd_raw_data.zip", DATA / "audio"
    if zip_path.exists():
        extract(zip_path, audio_dir, set(labels))

    raw: dict[str, RawSignals] = {}
    for uid, item in labels.items():
        path = audio_dir / f"{uid}.flac"
        if path.exists():
            signal, _ = sf.read(path, dtype="float32")
            raw[uid] = measure(signal, 0.0, len(signal) / SAMPLE_RATE, item["utterance"])
    print(f"{len(raw)} de {len(labels)} falas com áudio")

    groups: dict[str, list[RawSignals]] = defaultdict(list)
    for uid, s in raw.items():
        item = labels[uid]
        key = item["show"] if item["speaker"] == "PERSON" else item["speaker"]
        groups[key].append(s)
    bases = {
        key: {
            "rate": median(s.rate for s in ss),
            "rms": _median([s.rms_db for s in ss]),
            "f0": _median([s.f0_hz for s in ss]),
            "f0_range": _median([s.f0_range_st for s in ss]),
        }
        for key, ss in groups.items()
    }

    out = DATA / "mustard.jsonl"
    with out.open("w") as f:
        for uid, item in labels.items():
            if uid not in raw:
                continue
            key = item["show"] if item["speaker"] == "PERSON" else item["speaker"]
            row = {"id": uid, **item, "voz": relative(raw[uid], bases[key])}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"→ {out}")
    if zip_path.exists() and not args.manter_zip and len(raw) == len(labels):
        zip_path.unlink()
        print("zip apagado")


if __name__ == "__main__":
    main()
