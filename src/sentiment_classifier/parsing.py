"""Transforma um chat em texto (``Cliente: ...`` / ``Atendente: ...``) em turnos."""

from __future__ import annotations

import re

from sentiment_classifier.models import Speaker, Turn

_ALIASES: dict[str, Speaker] = {
    "cliente": "cliente",
    "c": "cliente",
    "usuario": "cliente",
    "usuário": "cliente",
    "atendente": "atendente",
    "a": "atendente",
    "agente": "atendente",
    "operador": "atendente",
    "operadora": "atendente",
    "suporte": "atendente",
}
_LINE = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?([\wÀ-ÿ]+)\s*:\s*(.*)$")


_ROLES: dict[str, Speaker] = {
    "customer": "cliente",
    "user": "cliente",
    "human": "cliente",
    "cliente": "cliente",
    "agent": "atendente",
    "assistant": "atendente",
    "system": "atendente",
    "atendente": "atendente",
}


def turns_from_messages(messages: list[dict[str, str]]) -> list[Turn]:
    """Converte ``[{"role": ..., "content": ...}]`` (formato comum dos datasets) em turnos.

    Papéis aceitos: customer/user/human/cliente e agent/assistant/system/atendente. Mensagens
    seguidas do mesmo papel viram um turno, como em ``parse_chat``.
    """
    lines = []
    for m in messages:
        role = _ROLES.get(str(m.get("role", "")).lower())
        if role is None:
            raise ValueError(f"papel desconhecido: {m.get('role')!r}")
        lines.append(f"{role}: {' '.join(str(m.get('content', '')).split())}")
    return parse_chat("\n".join(lines))


def parse_chat(raw: str) -> list[Turn]:
    """Lê uma conversa em que cada fala começa com ``Falante:``.

    Linhas sem rótulo continuam a fala anterior. Falas seguidas da mesma pessoa viram um
    turno só, porque no chat é comum alguém mandar várias mensagens curtas em sequência.
    Um horário entre colchetes antes do rótulo (``[10:02] Cliente: ...``) é ignorado.
    """
    turns: list[tuple[Speaker, list[str]]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        match = _LINE.match(line)
        speaker = _ALIASES.get(match.group(1).lower()) if match else None
        if match and speaker:
            text = match.group(2).strip()
            if turns and turns[-1][0] == speaker:
                turns[-1][1].append(text)
            else:
                turns.append((speaker, [text]))
        elif turns:
            turns[-1][1].append(line.strip())
        else:
            raise ValueError(f"a primeira fala não diz quem falou: {line.strip()!r}")
    if not turns:
        raise ValueError("conversa vazia")
    return [
        Turn(index=i, speaker=speaker, text=" ".join(p for p in parts if p))
        for i, (speaker, parts) in enumerate(turns)
    ]
