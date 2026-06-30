"""Persistent local vector store backed by sqlite-vec (Phase 3).

Why sqlite-vec over the existing FAISS backend, for the local build:

* **No external service and no separate index file** — vectors live in the same
  on-disk SQLite database as everything else, so storage is one file, writes are
  transactional, and memory **survives restarts** (the FAISS backend keeps its
  documents in RAM only).
* **Pure-local embeddings** — defaults to the :class:`OllamaEmbedder`
  (``nomic-embed-text``), never a hosted embedding API.
* **Tiny footprint** — the extension is a few hundred KB; the embedding model is
  the only sizeable resident piece and it is counted against the VRAM budget.

FAISS remains available for large in-memory indexes; sqlite-vec is the default
*persistent* choice. Requires ``sqlite-vec`` + ``numpy`` and a Python sqlite3
built with extension loading enabled (checked at construction with a clear error).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from openjarvis.core.events import EventType, get_event_bus
from openjarvis.core.registry import MemoryRegistry
from openjarvis.tools.storage._stubs import MemoryBackend, RetrievalResult
from openjarvis.tools.storage.embeddings import Embedder


@MemoryRegistry.register("sqlite_vec")
class SqliteVecMemory(MemoryBackend):
    """Persistent dense-retrieval backend using the sqlite-vec extension.

    Cosine similarity via the ``vec0`` virtual table over L2-normalised vectors
    (the embedders normalise rows, so distance ordering == cosine ordering).
    """

    backend_id: str = "sqlite_vec"

    def __init__(
        self,
        *,
        db_path: str | Path = ":memory:",
        embedder: Optional[Embedder] = None,
        dim: Optional[int] = None,
    ) -> None:
        try:
            import sqlite_vec
        except ImportError as exc:
            raise ImportError(
                "sqlite-vec is required for SqliteVecMemory. Install it with: "
                "uv sync --extra memory-sqlite-vec  (or pip install sqlite-vec)."
            ) from exc

        if embedder is None:
            # Local Ollama embeddings by default — never a hosted API.
            from openjarvis.tools.storage.embeddings import OllamaEmbedder

            embedder = OllamaEmbedder()
        self._embedder = embedder
        self._sqlite_vec = sqlite_vec

        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        try:
            self._conn.enable_load_extension(True)
        except AttributeError as exc:  # pragma: no cover - platform-specific
            raise RuntimeError(
                "This Python's sqlite3 was built without extension loading, "
                "which sqlite-vec needs. Use the FAISS backend instead, or a "
                "Python with loadable SQLite extensions."
            ) from exc
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        # Defer the vec0 table until we know the embedding dimension (probing the
        # embedder may hit the network, so do it lazily on first store/query).
        self._dim = dim
        self._vec_ready = False
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS documents ("
            "  doc_id TEXT PRIMARY KEY, content TEXT, source TEXT, metadata TEXT"
            ")"
        )
        self._conn.commit()
        # If the db already has a vec table (reopened), mark ready.
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_docs'"
        ).fetchone()
        if row is not None:
            self._vec_ready = True

    # ------------------------------------------------------------------

    def _ensure_vec_table(self) -> int:
        if self._vec_ready:
            return self._dim or 0
        if self._dim is None:
            self._dim = int(self._embedder.dim())
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_docs USING vec0("
            f"  doc_id TEXT PRIMARY KEY, embedding float[{self._dim}]"
            f")"
        )
        self._conn.commit()
        self._vec_ready = True
        return self._dim

    def _vec(self, text: str) -> bytes:
        arr = self._embedder.embed([text])
        return self._sqlite_vec.serialize_float32([float(x) for x in arr[0]])

    def store(
        self,
        content: str,
        *,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        self._ensure_vec_table()
        doc_id = uuid.uuid4().hex
        meta = metadata if metadata is not None else {}
        self._conn.execute(
            "INSERT INTO documents (doc_id, content, source, metadata) "
            "VALUES (?, ?, ?, ?)",
            (doc_id, content, source, json.dumps(meta)),
        )
        self._conn.execute(
            "INSERT INTO vec_docs (doc_id, embedding) VALUES (?, ?)",
            (doc_id, self._vec(content)),
        )
        self._conn.commit()
        get_event_bus().publish(
            EventType.MEMORY_STORE,
            {"backend": self.backend_id, "doc_id": doc_id, "source": source},
        )
        return doc_id

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        if query.strip() and self._vec_ready:
            rows = self._conn.execute(
                "SELECT v.doc_id, v.distance, d.content, d.source, d.metadata "
                "FROM vec_docs v JOIN documents d ON d.doc_id = v.doc_id "
                "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
                (self._vec(query), top_k),
            ).fetchall()
            for _doc_id, distance, content, source, meta_json in rows:
                # vec0 returns L2 distance on normalised vectors; map to a
                # bounded similarity score (1 at distance 0).
                score = 1.0 / (1.0 + float(distance))
                results.append(
                    RetrievalResult(
                        content=content,
                        score=score,
                        source=source or "",
                        metadata=json.loads(meta_json) if meta_json else {},
                    )
                )
        get_event_bus().publish(
            EventType.MEMORY_RETRIEVE,
            {"backend": self.backend_id, "query": query, "num_results": len(results)},
        )
        return results

    def delete(self, doc_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        existed = cur.rowcount > 0
        if self._vec_ready:
            self._conn.execute("DELETE FROM vec_docs WHERE doc_id = ?", (doc_id,))
        self._conn.commit()
        return existed

    def clear(self) -> None:
        self._conn.execute("DELETE FROM documents")
        if self._vec_ready:
            self._conn.execute("DELETE FROM vec_docs")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


__all__ = ["SqliteVecMemory"]
