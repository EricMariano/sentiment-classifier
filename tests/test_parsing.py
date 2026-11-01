import pytest

from sentiment_classifier.parsing import parse_chat, turns_from_messages


def test_aliases_continuation_and_merge() -> None:
    turns = parse_chat(
        "[10:01] Cliente: oi\ncliente: tudo bem?\nOperador: Olá!\ncontinuação da fala\n\nC: certo"
    )
    assert [(t.index, t.speaker, t.text) for t in turns] == [
        (0, "cliente", "oi tudo bem?"),
        (1, "atendente", "Olá! continuação da fala"),
        (2, "cliente", "certo"),
    ]


def test_first_line_needs_speaker() -> None:
    with pytest.raises(ValueError, match="quem falou"):
        parse_chat("sem rótulo\nCliente: oi")


def test_empty() -> None:
    with pytest.raises(ValueError, match="vazia"):
        parse_chat("  \n")


def test_messages_format_merges_and_maps_roles() -> None:
    turns = turns_from_messages(
        [
            {"role": "user", "content": "oi"},
            {"role": "customer", "content": "tudo\nbem?"},
            {"role": "assistant", "content": "Olá!"},
        ]
    )
    assert [(t.speaker, t.text) for t in turns] == [
        ("cliente", "oi tudo bem?"),
        ("atendente", "Olá!"),
    ]


def test_unknown_role() -> None:
    with pytest.raises(ValueError, match="papel"):
        turns_from_messages([{"role": "robô", "content": "x"}])
