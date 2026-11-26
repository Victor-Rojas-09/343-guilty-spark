"""Dobles de prueba compartidos para el servicio y la API de chat."""

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from spark_rag.rag.chain import RagResult
from spark_rag.rag.history import ChatTurn
from spark_rag.rag.mood import Mood
from spark_rag.rag.sources import RetrievedChunk

CHUNK = RetrievedChunk(
    content="Chakas fue un humano de Erde-Tyrene.",
    similarity=0.81,
    metadata={"source": "history/dossier.pdf", "category": "history", "page": 3},
)


class InMemoryStore:
    def __init__(self) -> None:
        self.messages: dict[UUID, list[dict[str, Any]]] = {}
        self.fail = False

    def exists(self, session_id: UUID) -> bool:
        if self.fail:
            raise ConnectionError("postgresql://user:secret@db")
        return session_id in self.messages

    def recent_messages(self, session_id: UUID, limit: int) -> list[dict[str, Any]]:
        return self.messages.get(session_id, [])[-limit:] if limit > 0 else []

    def save_exchange(self, **kwargs: Any) -> None:
        if self.fail:
            raise ConnectionError("postgresql://user:secret@db")
        rows = self.messages.setdefault(kwargs["session_id"], [])
        rows.append({"role": "user", "content": kwargs["user_message"]})
        rows.append(
            {
                "role": "assistant",
                "content": kwargs["assistant_message"],
                "mood": kwargs["mood"],
                "sources": kwargs["sources"],
                "prompt_version": kwargs["prompt_version"],
                "model": kwargs["model"],
            }
        )


class FakeChain:
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = [CHUNK] if chunks is None else chunks
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None

    def invoke(
        self,
        question: str,
        *,
        history: Sequence[ChatTurn] = (),
        retrieval_query: str | None = None,
    ) -> RagResult:
        self.calls.append(
            {
                "question": question,
                "history": list(history),
                "retrieval_query": retrieval_query,
            }
        )
        if self.error is not None:
            raise self.error
        return RagResult(
            reply=f"Respuesta a: {question}", mood=Mood.THINKING, chunks=self.chunks
        )
