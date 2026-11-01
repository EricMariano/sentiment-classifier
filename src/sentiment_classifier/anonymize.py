"""Remove dados pessoais antes de qualquer texto sair da máquina.

Roda sempre, antes do Jev e do LLM. É baseada em padrões, então pega formatos
previsíveis (CPF, CNPJ, cartão, e-mail, telefone, CEP, endereço com número) e nomes que
aparecem depois de expressões como "meu nome é". Nomes soltos no meio da frase passam:
para produção, combine com um reconhecedor de entidades (por exemplo, Microsoft Presidio
com um modelo em português) e revise uma amostra à mão.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace

from sentiment_classifier.models import Turn

_NAME = r"[A-ZÀ-Ý][a-zà-ÿ]+(?:\s+(?:d[aeo]s?\s+)?[A-ZÀ-Ý][a-zà-ÿ]+){0,3}"

# A ordem importa: padrões mais longos antes dos que poderiam casar com um pedaço deles.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("CNPJ", re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")),
    ("CARTAO", re.compile(r"\b(?:\d{4}[ .-]?){3}\d{1,4}\b")),
    ("CPF", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    (
        "TELEFONE",
        re.compile(r"(?:\+?55\s?)?(?:\(?\b\d{2}\)?\s?)?\b9?\d{4}[ -]?\d{4}\b"),
    ),
    ("CEP", re.compile(r"\b\d{5}-\d{3}\b")),
    (
        "ENDERECO",
        re.compile(
            r"\b(?:[Rr]ua|R\.|[Aa]venida|[Aa]v\.|[Tt]ravessa|[Aa]lameda|[Ee]strada|[Rr]odovia|"
            r"[Pp]raça)\s+[^,\n]{2,60},?\s*(?:n[º°o.]?\s*)?\d{1,6}\b"
        ),
    ),
)
_INTRO = re.compile(
    r"\b((?i:meu nome é|me chamo|aqui é (?:[oa] )?|sou (?:[oa] )|falo com (?:[oa] )?|"
    r"em nome de|titular é (?:[oa] )?)\s*)(" + _NAME + ")"
)


_KEEP = "\x00"


def _luhn(digits: str) -> bool:
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d) * (2 if i % 2 else 1)
        total += n - 9 if n > 9 else n
    return total % 10 == 0


def _replacer(label: str):  # type: ignore[no-untyped-def]
    """Cartão só é trocado se passar no dígito verificador, para poupar protocolos."""

    def replace(match: re.Match[str]) -> str:
        if label == "CARTAO" and not _luhn(re.sub(r"\D", "", match.group(0))):
            return match.group(0) + _KEEP
        return f"[{label}]"

    return replace


@dataclass(frozen=True)
class Redaction:
    turns: list[Turn]
    counts: dict[str, int]
    """Quantas substituições de cada tipo foram feitas, para mostrar no relatório."""


def redact_text(text: str, names: Iterable[str] = (), counts: Counter[str] | None = None) -> str:
    counts = counts if counts is not None else Counter()
    for label, pattern in _PATTERNS:
        text, n = pattern.subn(_replacer(label), text)
        counts[label] += n - text.count(_KEEP)
        text = text.replace(_KEEP, "")

    def _intro(match: re.Match[str]) -> str:
        counts["NOME"] += 1
        return f"{match.group(1)}[NOME]"

    text = _INTRO.sub(_intro, text)
    for name in sorted({n.strip() for n in names if n.strip()}, key=len, reverse=True):
        text, n = re.subn(rf"\b{re.escape(name)}\b", "[NOME]", text, flags=re.IGNORECASE)
        counts["NOME"] += n
    return text


def introduced_names(text: str) -> set[str]:
    """Nomes ditos numa apresentação ("meu nome é Ana Lima"): o nome completo e o primeiro."""
    found = set()
    for match in _INTRO.finditer(text):
        full = match.group(2)
        found |= {full, full.split()[0]}
    return found


def redact_turns(turns: Iterable[Turn], names: Iterable[str] = ()) -> Redaction:
    """Anonimiza cada turno. ``names`` são nomes conhecidos (do CRM, por exemplo).

    Um nome dito numa apresentação é procurado na conversa inteira, então o "Obrigado,
    Ana" do atendente também é trocado.
    """
    turns = list(turns)
    names = list(names)
    for t in turns:
        names += introduced_names(t.text)
    counts: Counter[str] = Counter()
    redacted = [replace(t, text=redact_text(t.text, names, counts)) for t in turns]
    return Redaction(redacted, {k: v for k, v in counts.items() if v})
