from conftest import reading

from sentiment_classifier.dynamics import analyze_dynamics


def test_turning_points_point_at_agent_turns_in_between() -> None:
    dyn = analyze_dynamics(
        [
            reading(0, "cliente", 3.0),
            reading(1, "atendente", 2.0, "burocratico"),
            reading(2, "cliente", 1.5),
            reading(3, "atendente", 5.0, "resolutivo"),
            reading(4, "atendente", 4.0),
            reading(5, "cliente", 4.5),
        ]
    )
    assert [
        (t.customer_before, t.customer_after, t.delta, t.agent_turns) for t in dyn.turning_points
    ] == [
        (0, 2, -1.5, (1,)),
        (2, 5, 3.0, (3, 4)),
    ]
    assert [t.direction for t in dyn.turning_points] == ["piorou", "melhorou"]
    assert dyn.curves["atendente"] == [(1, 2.0), (3, 5.0), (4, 4.0)]
    assert dyn.key_turns == [0, 1, 2, 3, 4, 5]


def test_small_changes_are_not_turning_points() -> None:
    dyn = analyze_dynamics([reading(0, "cliente", 3.0), reading(1, "cliente", 3.9)])
    assert dyn.turning_points == [] and dyn.alerts == [] and dyn.key_turns == []


def test_alerts_from_flags_and_from_the_curve() -> None:
    dyn = analyze_dynamics(
        [
            reading(0, "cliente", 4.0),
            reading(1, "atendente", 3.0, solucao_concreta=0.9),
            reading(2, "cliente", 3.5, cancelamento=0.69),
            reading(3, "cliente", 2.0, cancelamento=0.7, sarcasmo=0.2),
        ]
    )
    kinds = [(a.kind, a.turn_index, a.probability) for a in dyn.alerts]
    # cancelamento só no limiar; a queda de 4.0 para 2.0 também pede escalada
    assert kinds == [("cancelamento", 3, 0.7), ("escalar", 3, None)]


def test_agent_flags_never_raise_customer_alerts() -> None:
    dyn = analyze_dynamics([reading(0, "atendente", 3.0, cancelamento=0.99)])
    assert dyn.alerts == [] and dyn.customer_start is None
