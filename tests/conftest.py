from __future__ import annotations

from pathlib import Path

import pytest

from sentiment_classifier.models import Speaker, Turn, TurnReading

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def reading(
    index: int,
    speaker: Speaker,
    sentiment: float,
    label: str = "neutro",
    text: str = "",
    **flags: float,
) -> TurnReading:
    return TurnReading(
        turn=Turn(index, speaker, text or f"fala {index}"),
        sentiment=sentiment,
        sentiment_confidence=0.9,
        label=label,
        label_confidence=0.9,
        label_probabilities={label: 0.9},
        flags=flags,
    )


@pytest.fixture
def samples() -> Path:
    return SAMPLES
