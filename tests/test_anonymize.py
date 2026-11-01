from sentiment_classifier.anonymize import redact_text, redact_turns
from sentiment_classifier.models import Turn


def test_structured_identifiers() -> None:
    text = (
        "CPF 123.456.789-09, CNPJ 12.345.678/0001-90, cartão 4111 1111 1111 1111, "
        "e-mail ana.silva+x@exemplo.com.br, tel (11) 98765-4321, CEP 01310-100, "
        "moro na Rua das Flores, 123"
    )
    out = redact_text(text)
    for token in ("[CPF]", "[CNPJ]", "[CARTAO]", "[EMAIL]", "[TELEFONE]", "[CEP]", "[ENDERECO]"):
        assert token in out, token
    for leak in ("789", "0001", "1111", "ana.silva", "98765", "01310", "Flores"):
        assert leak not in out, leak


def test_values_and_protocols_survive() -> None:
    out = redact_text("Estorno de R$ 89,90 em até 48 horas, protocolo 2026102412345.")
    assert out == "Estorno de R$ 89,90 em até 48 horas, protocolo 2026102412345."


def test_names_after_introductions_and_known_names() -> None:
    out = redact_text("Meu nome é Mariana Souza. Obrigado, mariana.", names=["Mariana"])
    assert "Mariana" not in out and "mariana" not in out
    assert out.count("[NOME]") == 2


def test_counts_per_type() -> None:
    result = redact_turns(
        [Turn(0, "cliente", "CPF 12345678909"), Turn(1, "cliente", "outro: 987.654.321-00")]
    )
    assert result.counts == {"CPF": 2}
    assert [t.text for t in result.turns] == ["CPF [CPF]", "outro: [CPF]"]


def test_introduced_name_is_redacted_everywhere() -> None:
    result = redact_turns(
        [
            Turn(0, "cliente", "Oi. Meu nome é Mariana Souza."),
            Turn(1, "atendente", "Obrigado, Mariana. Boa tarde!"),
        ]
    )
    assert [t.text for t in result.turns] == [
        "Oi. Meu nome é [NOME].",
        "Obrigado, [NOME]. Boa tarde!",
    ]
