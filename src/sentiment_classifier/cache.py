"""Cache das respostas do Jev, para não pagar duas vezes pela mesma pergunta.

A chave é um hash SHA-256 do pedido (estado anonimizado + perguntas + modelo). No disco
fica só o hash e a resposta (probabilidades), nunca o texto da conversa. Reanalisar a mesma
conversa, rodar o lote de novo ou trocar só o resumo fica instantâneo e de graça.

Mudou a pergunta, o contexto ou a voz? O hash muda e o Jev é chamado de novo.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from langchain_typesafe import ClassifierRequest, ClassifierResponse

VERSION = 1
"""Suba quando o significado de uma resposta mudar sem o pedido mudar."""


def request_key(request: ClassifierRequest, model: str) -> str:
    questions = {k: q.model_dump(mode="json") for k, q in request["questions"].items()}
    payload = {"v": VERSION, "model": model, "state": request["state"], "questions": questions}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class ResponseCache:
    """SQLite em disco, seguro para várias threads."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS jev (key TEXT PRIMARY KEY, response TEXT)")
        self._lock = threading.Lock()

    def get(self, key: str) -> ClassifierResponse | None:
        with self._lock:
            row = self._db.execute("SELECT response FROM jev WHERE key = ?", (key,)).fetchone()
        return ClassifierResponse.model_validate_json(row[0]) if row else None

    def put(self, key: str, response: ClassifierResponse) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO jev VALUES (?, ?)", (key, response.model_dump_json())
            )
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM jev").fetchone()[0])


def model_name(client: Any) -> str:
    return str(getattr(client, "model", "") or "")
