# src/python_uv_template/vector_db.py
"""
Capa de acceso a la Vector DB (implementación Chroma).
Al migrar a OpenSearch, mantén la MISMA interfaz pública y cambia la implementación interna.

Interfaz pública:
- ChromaVectorDB(embeddings, collection_name, persist_directory)
- index(documents, ids=None)
- similarity_search(query, k=5, metadata_filter=None)
- mmr_search(query, k=5, fetch_k=20, metadata_filter=None)
- as_retriever(k=5, metadata_filter=None)
- persist()
- reset()  # borra la colección
- store  # acceso al VectorStore subyacente (si lo necesitas)
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Sequence

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma


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

        # Crea o carga la colección persistente
        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=self.persist_directory,
        )

    # ------- Indexación -------
    def index(self, documents: Sequence[Document], ids: Optional[Sequence[str]] = None) -> List[str]:
        """
        Indexa documentos (y opcionalmente IDs estables). Devuelve los ids.
        """
        # Asegúrate de que los metadatos clave estén presentes (status, materia, etc.)
        return self.store.add_documents(list(documents), ids=list(ids) if ids else None)

    def persist(self) -> None:
        """Persiste en disco."""
        self.store.persist()

    def reset(self) -> None:
        """Elimina la colección (¡cuidado!)."""
        try:
            self.store.delete_collection()
        except Exception:
            # Fallback: recrear el objeto de store
            self.store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self._embeddings,
                persist_directory=self.persist_directory,
            )

    # ------- Búsqueda -------
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Búsqueda por similitud con filtro de metadatos (AND).
        Ejemplo filter: {"status": "Mantiene", "materia": "Créditos hipotecarios"}
        """
        return self.store.similarity_search(query, k=k, filter=metadata_filter)

    def mmr_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Búsqueda MMR (diversidad). Útil para evitar mucha redundancia.
        """
        return self.store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, filter=metadata_filter
        )

    # ------- Retriever (LangChain) -------
    def as_retriever(
        self, k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None
    ):
        """
        Devuelve un retriever compatible con LC. Puedes pasar search_kwargs con k y filter.
        """
        search_kwargs = {"k": k}
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter
        return self.store.as_retriever(search_kwargs=search_kwargs)
