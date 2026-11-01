"""Agregação sem LLM: curvas por pessoa, pontos de virada e alertas.

Tudo aqui é determinístico e testável. Os limiares são pontos de partida, não valores
calibrados: ajuste com conversas rotuladas da sua operação.
"""

from __future__ import annotations

from collections.abc import Sequence

from sentiment_classifier.models import (
    SPEAKERS,
    Alert,
    AlertKind,
    Dynamics,
    TurnFailure,
    TurningPoint,
    TurnReading,
)

TURN_DELTA = 1.0
"""Variação mínima (na escala 1–5) entre dois turnos seguidos do cliente."""
FLAG_THRESHOLD = 0.7
DRIFT_DROP = 1.5
"""Queda total do início ao fim que, terminando em 2 ou menos, pede escalada."""
MAX_KEY_TURNS = 12

_FLAG_ALERTS: dict[str, tuple[AlertKind, str]] = {
    "cancelamento": ("cancelamento", "cliente fala em cancelar ou trocar de empresa"),
    "pede_escalada": ("escalar", "cliente pede supervisor ou cita órgão externo"),
    "sarcasmo": ("sarcasmo", "fala positiva no texto, negativa no sentido"),
}


def analyze_dynamics(
    readings: Sequence[TurnReading],
    failures: Sequence[TurnFailure] = (),
    *,
    turn_delta: float = TURN_DELTA,
    flag_threshold: float = FLAG_THRESHOLD,
) -> Dynamics:
    curves = {
        speaker: [(r.turn.index, r.sentiment) for r in readings if r.turn.speaker == speaker]
        for speaker in SPEAKERS
    }
    customer = [r for r in readings if r.turn.speaker == "cliente"]

    turning_points = []
    for before, after in zip(customer, customer[1:], strict=False):
        delta = after.sentiment - before.sentiment
        if abs(delta) >= turn_delta:
            agent_turns = tuple(
                r.turn.index
                for r in readings
                if r.turn.speaker == "atendente"
                and before.turn.index < r.turn.index < after.turn.index
            )
            missing = tuple(
                f.turn.index
                for f in failures
                if before.turn.index < f.turn.index < after.turn.index
            )
            turning_points.append(
                TurningPoint(
                    before.turn.index, after.turn.index, round(delta, 2), agent_turns, missing
                )
            )

    alerts = [
        Alert(kind, r.turn.index, p, reason)
        for r in customer
        for flag, (kind, reason) in _FLAG_ALERTS.items()
        if (p := r.flags.get(flag, 0.0)) >= flag_threshold
    ]
    start = customer[0].sentiment if customer else None
    end = customer[-1].sentiment if customer else None
    if start is not None and end is not None and end <= 2 and start - end >= DRIFT_DROP:
        alerts.append(
            Alert(
                "escalar",
                customer[-1].turn.index,
                None,
                f"sentimento do cliente caiu de {start:.1f} para {end:.1f} e não se recuperou",
            )
        )

    key: set[int] = set()
    for tp in turning_points:
        key.update(range(tp.customer_before, tp.customer_after + 1))
    for alert in alerts:
        key.update({max(0, alert.turn_index - 1), alert.turn_index})
    # Se houver momentos demais, fica com os pontos de virada de maior variação.
    if len(key) > MAX_KEY_TURNS:
        key = set()
        for tp in sorted(turning_points, key=lambda t: -abs(t.delta)):
            span = set(range(tp.customer_before, tp.customer_after + 1))
            if len(key | span) > MAX_KEY_TURNS:
                break
            key |= span

    return Dynamics(
        curves=curves,
        turning_points=turning_points,
        alerts=alerts,
        customer_start=start,
        customer_end=end,
        key_turns=sorted(key),
        unread_turns=sorted(f.turn.index for f in failures),
    )
