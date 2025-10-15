"""
vector_db.py
Contenedor de utilidades para indexación y recuperación en ChromaDB.
Incluye saneamiento de metadatos y adaptadores de búsqueda (similaridad y MMR).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple
from math import sqrt

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
    def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
        """Distancia coseno = 1 - cos_sim(a, b)."""
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na == 0.0 or nb == 0.0:
            # Si alguna norma es 0, no se puede definir coseno; devolvemos distancia neutra 1.0
            return 1.0
        return 1.0 - (dot / (sqrt(na) * sqrt(nb)))


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
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Igual que similarity_search pero devuelve (Document, score).
        El score es la distancia nativa que reporta Chroma (típicamente cosine distance: menor es mejor).
        """
        # 1) Si el wrapper de LangChain ya lo trae, úsalo.
        try:
            return self.store.similarity_search_with_score(query, k=k, filter=metadata_filter)
        except AttributeError:
            pass  # caemos al plan B

        # 2) Plan B: llamar a la colección nativa de Chroma y armar los Documents + distancias.
        coll = getattr(self.store, "_collection", None)
        if coll is None:
            # Fallback extremo: sin distancias (compatibilidad)
            docs = self.store.similarity_search(query, k=k, filter=metadata_filter)
            return [(d, None) for d in docs]  # type: ignore[list-item]

        res = coll.query(
            query_texts=[query],
            n_results=k,
            where=metadata_filter or {},
            include=["documents", "metadatas", "distances", "ids"],
        )
        # res["documents"], res["metadatas"], res["distances"] son listas por query (aquí 1).
        docs_out: List[Tuple[Document, float]] = []
        docs_list = (res.get("documents") or [[]])[0]
        metas_list = (res.get("metadatas") or [[]])[0]
        dists_list = (res.get("distances") or [[]])[0]

        for txt, md, dist in zip(docs_list, metas_list, dists_list):
            # md ya viene saneado de indexación
            doc = Document(page_content=txt or "", metadata=dict(md or {}))
            docs_out.append((doc, float(dist)))
        return docs_out

    def mmr_search_with_score(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Selecciona documentos con MMR (diversidad) y devuelve (Document, score).
        El score se calcula como distancia coseno entre el embedding de query y el embedding del documento
        (usando primero embeddings almacenados en Chroma; si no están, se recalculan).
        """
        # 1) Selección MMR usando el wrapper (no devuelve score).
        docs = self.store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, filter=metadata_filter
        )
        if not docs:
            return []

        # 2) Embedding del query
        q_vec = self._embeddings.embed_query(query)

        # 3) Intentar recuperar embeddings de los docs seleccionados desde la colección
        coll = getattr(self.store, "_collection", None)
        out: List[Tuple[Document, float]] = []

        # Construimos lista de IDs si vienen en metadata
        ids: List[str] = []
        for d in docs:
            mid = (d.metadata or {}).get("id")
            if isinstance(mid, str) and mid:
                ids.append(mid)
            else:
                ids.append("")  # placeholder para mantener el índice

        doc_embeds_by_id: Dict[str, List[float]] = {}
        if coll and any(ids):
            # Filtrar vacíos
            ids_to_fetch = [i for i in ids if i]
            if ids_to_fetch:
                try:
                    got = coll.get(ids=ids_to_fetch, include=["embeddings"])
                    # got["ids"] -> lista en el mismo orden que embeddings
                    for gid, gemb in zip(got.get("ids", []), got.get("embeddings", [])):
                        if gid and gemb:
                            doc_embeds_by_id[str(gid)] = list(map(float, gemb))
                except Exception:
                    # Si falla, seguimos con recálculo
                    pass

        # 4) Para cada doc: usar embedding almacenado si está; si no, recalcular.
        for d, doc_id in zip(docs, ids):
            emb: Optional[List[float]] = None
            if doc_id and doc_id in doc_embeds_by_id:
                emb = doc_embeds_by_id[doc_id]
            else:
                # Recalcular embedding del contenido como fallback
                try:
                    emb_list = self._embeddings.embed_documents([d.page_content or ""])
                    emb = list(map(float, emb_list[0])) if emb_list else None
                except Exception:
                    emb = None
            # 5) Distancia coseno (si no podemos calcular, devolver None)
            score = _cosine_distance(q_vec, emb) if emb else None  # type: ignore[arg-type]
            out.append((d, score if score is not None else 1.0))

        return out