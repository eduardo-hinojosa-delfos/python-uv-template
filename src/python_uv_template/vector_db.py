"""
vector_db.py
Contenedor de utilidades para indexación y recuperación en ChromaDB.
Incluye saneamiento de metadatos y adaptadores de búsqueda (similaridad y MMR).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_chroma import Chroma


def _strip_invisible(s: str) -> str:
    """Normaliza a NFKC, elimina caracteres de ancho cero y espacios no estándar."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    s = s.replace("\u00A0", " ").replace("\u2007", " ").replace("\u202F", " ")
    return s.strip()


def _sanitize_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convierte metadatos a tipos JSON-serializables, limpia valores de texto
    y garantiza las claves 'materia' y 'status'.
    """
    out: Dict[str, Any] = {}
    for k, v in (md or {}).items():
        key = str(k)
        if isinstance(v, (str, int, float, bool)) or v is None:
            val = v
        else:
            val = str(v)
        if isinstance(val, str) or val is None:
            val = _strip_invisible(val or "")
        out[key] = val

    out["materia"] = _strip_invisible(out.get("materia", "")) or "Sin información"
    out["status"] = _strip_invisible(out.get("status", "")) or "Sin información"
    return out


class ChromaVectorDB:
    """
    Fachada para una colección Chroma persistente con embeddings dados.
    Ofrece operaciones de indexación, persistencia, reinicio y recuperación.
    """

    def __init__(
        self,
        embeddings,
        collection_name: str = "tc_uru",
        persist_directory: str = "chroma_db",
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._embeddings = embeddings

        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=self.persist_directory,
        )

    # ------- Indexación -------
    def index(self, documents: Sequence[Document], ids: Optional[Sequence[str]] = None) -> List[str]:
        """
        Indexa documentos tras sanear metadatos. Devuelve los IDs asignados.
        """
        docs: List[Document] = []
        for d in documents:
            md = _sanitize_metadata(dict(d.metadata or {}))
            docs.append(Document(page_content=d.page_content, metadata=md))

        return self.store.add_documents(docs, ids=list(ids) if ids else None)

    def persist(self) -> None:
        """Persiste la base de datos en disco si el cliente lo soporta."""
        client = getattr(self.store, "_client", None)
        if client and hasattr(client, "persist"):
            try:
                client.persist()
            except Exception:
                pass

    def reset(self) -> None:
        """Elimina la colección y la re-crea vacía con la misma configuración."""
        try:
            if hasattr(self.store, "reset_collection"):
                self.store.reset_collection()
            else:
                self.store.delete_collection()
        except Exception:
            pass

        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=self.persist_directory,
        )
        _ = self.store._collection  # fuerza inicialización

    # ------- Búsqueda -------
    def similarity_search(
        self,
        query: str,
        k: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Recupera por similitud coseno los k documentos más relevantes."""
        return self.store.similarity_search(query, k=k, filter=metadata_filter)

    def mmr_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Recupera con MMR priorizando diversidad semántica."""
        return self.store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, filter=metadata_filter
        )

    def as_retriever(self, k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None):
        """Expone un retriever LangChain con parámetros de búsqueda preconfigurados."""
        search_kwargs = {"k": k}
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter
        return self.store.as_retriever(search_kwargs=search_kwargs)