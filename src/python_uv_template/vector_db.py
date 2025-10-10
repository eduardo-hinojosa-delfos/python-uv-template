from __future__ import annotations
from typing import List, Optional, Dict, Any, Sequence
import re
import unicodedata

from langchain_core.documents import Document
from langchain_chroma import Chroma


def _strip_invisible(s: str) -> str:
    """Quita ZWSP/BOM y normaliza espacios raros; NFKC para unificar signos."""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)  # zero-width
    s = s.replace("\u00A0", " ").replace("\u2007", " ").replace("\u202F", " ")
    return s.strip()


def _sanitize_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    """Convierte metadatos a tipos JSON-serializables y asegura materia/status."""
    out: Dict[str, Any] = {}

    # Copiamos todo pero forzamos a str los que no sean tipos simples
    for k, v in (md or {}).items():
        key = str(k)
        if isinstance(v, (str, int, float, bool)) or v is None:
            val = v
        else:
            # evita objetos no serializables (Path, datetime, etc.)
            val = str(v)

        if isinstance(val, str) or val is None:
            val = _strip_invisible(val or "")
        out[key] = val

    # Asegura claves obligatorias con valor limpio
    materia = _strip_invisible(out.get("materia", "")) or "Sin información"
    status = _strip_invisible(out.get("status", "")) or "Sin información"
    out["materia"] = materia
    out["status"]  = status

    return out


class ChromaVectorDB:
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
        Indexa documentos asegurando que 'materia' y 'status' se graben siempre
        como strings limpios (o 'Sin información').
        """
        docs: List[Document] = []
        for i, d in enumerate(documents):
            md = _sanitize_metadata(dict(d.metadata or {}))
            docs.append(Document(page_content=d.page_content, metadata=md))

        return self.store.add_documents(docs, ids=list(ids) if ids else None)

    def persist(self) -> None:
        client = getattr(self.store, "_client", None)
        if client and hasattr(client, "persist"):
            try:
                client.persist()
            except Exception:
                pass

    def reset(self) -> None:
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
        return self.store.similarity_search(query, k=k, filter=metadata_filter)

    def mmr_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        return self.store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, filter=metadata_filter
        )

    def as_retriever(self, k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None):
        search_kwargs = {"k": k}
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter
        return self.store.as_retriever(search_kwargs=search_kwargs)
