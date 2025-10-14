# python_uv_template/main.py
import os
import sys
import subprocess
import calendar
import re
from typing import List, Dict, Any, Optional

def run_streamlit():
    """Lanza esta app con `streamlit run` (entry-point para `uv run main`)."""
    script_path = os.path.join(os.path.dirname(__file__), "main.py")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", script_path], check=True)
    except KeyboardInterrupt:
        print("\nInterfaz detenida por el usuario.")
    except Exception as e:
        print(f"\nError al iniciar Streamlit: {e}")
        sys.exit(1)

def app():
    """Construye la UI de Streamlit y orquesta enriquecimiento → retrieval → generación."""
    import json
    from functools import lru_cache

    import streamlit as st
    from dotenv import load_dotenv

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.chat_models import init_chat_model
    from langchain.prompts import PromptTemplate
    from langchain_core.documents import Document

    from python_uv_template.vector_db import ChromaVectorDB

    load_dotenv()
    st.set_page_config(page_title="RAG Jurídico", page_icon="⚖️", layout="wide")

    DEFAULT_CHROMA_DIR = "chroma_db"
    DEFAULT_COLLECTION = "tc_uru"
    DEFAULT_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_LLM_MODEL = "gpt-4o-mini"

    @lru_cache(maxsize=1)
    def build_enrichment_prompt() -> PromptTemplate:
        """Devuelve el prompt para que el LLM extraiga/expanda filtros y consulta."""
        return PromptTemplate(
            input_variables=["question"],
            template=(
                "Eres un agente de jurisprudencia experto en documentos legales.\n"
                "Objetivo: Expande/extrae filtros útiles para recuperar evidencia correcta.\n\n"
                "Devuelve SOLO un JSON con esta estructura EXACTA:\n"
                "{\n"
                '  "expanded_query": "pregunta reescrita",\n'
                '  "keywords": ["k1","k2"],\n'
                '  "entities": ["entidad1","entidad2"],\n'
                '  "filters": { "materia": "", "status": "", "date": "", "expediente": "", "carpeta": ""}\n'
                "}\n"
                "- `filters.date` en ISO: YYYY, YYYY-MM o YYYY-MM-DD.\n"
                "- Si no hay valor, usar vacío.\n\n"
                "Pregunta:\n{question}\n"
            ),
        )

    def _ym_last_day(year: int, month: int) -> int:
        """Último día del mes (utilidad básica de fechas)."""
        return calendar.monthrange(year, month)[1]

    def build_chroma_where(md: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Construye objeto `where` de Chroma a partir de metadatos (eq simples)."""
        if not md:
            return None
        clauses = []
        if md.get("materia"):
            clauses.append({"materia": {"$eq": str(md["materia"]).strip()}})
        if md.get("status"):
            clauses.append({"status": {"$eq": str(md["status"]).strip()}})
        if md.get("expediente"):
            clauses.append({"expediente": {"$eq": str(md["expediente"]).strip()}})
        if md.get("source_dir"):
            clauses.append({"source_dir": {"$eq": str(md["source_dir"]).strip()}})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def build_date_predicate(date_filter_str: str):
        """Crea un predicado para filtrar localmente por fecha (YYYY|YYYY-MM|YYYY-MM-DD)."""
        s = (date_filter_str or "").strip()
        if not s:
            return lambda _: True
        m_year = re.fullmatch(r"(\d{4})", s)
        if m_year:
            year = m_year.group(1)
            return lambda d: (d or "").startswith(f"{year}-")
        m_year_month = re.fullmatch(r"(\d{4})-(\d{2})", s)
        if m_year_month:
            year, month = m_year_month.group(1), m_year_month.group(2)
            return lambda d: (d or "").startswith(f"{year}-{month}-")
        m_full = re.fullmatch(r"\d{4}-\d{2}-\d{2}", s)
        if m_full:
            return lambda d: (d or "") == s
        m = re.search(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?", s)
        if m:
            if m.group(2) is None:
                year = m.group(1)
                return lambda d: (d or "").startswith(f"{year}-")
            if m.group(3) is None:
                year, month = m.group(1), m.group(2)
                return lambda d: (d or "").startswith(f"{year}-{month}-")
            return lambda d: (d or "") == f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        return lambda _: True

    @lru_cache(maxsize=1)
    def build_generation_prompt() -> PromptTemplate:
        """Devuelve el prompt para síntesis final usando solo el contexto recuperado."""
        return PromptTemplate(
            input_variables=["question", "context"],
            template=(
                "Eres un agente de jurisprudencia.\n"
                "Responde breve y directo usando SOLO los documentos recuperados.\n"
                "Si no hay información suficiente, responde: \"No se encontró información suficiente\".\n\n"
                "Pregunta:\n{question}\n\n"
                "Documentos recuperados:\n{context}\n\n"
                "Respuesta concisa:\n"
            ),
        )

    def format_docs_for_prompt(docs: List[Document], max_chars_per_doc: int = 1800) -> str:
        """Formatea docs y metadatos para el prompt del generador."""
        blocks = []
        for doc in docs:
            md = doc.metadata or {}
            content = (doc.page_content or "").strip().replace("\r\n", "\n")
            if len(content) > max_chars_per_doc:
                content = content[:max_chars_per_doc] + "…"
            block = (
                "[METADATOS]\n"
                f"Fecha: {md.get('date','')}\n"
                f"Expediente: {md.get('expediente','')}\n"
                f"Entrada: {md.get('entrada','')}\n"
                f"Materia: {md.get('materia','')}\n"
                f"Status: {md.get('status','')}\n"
                f"Archivo: {md.get('source_file','')}\n"
                f"Sección: {md.get('section','')}\n"
                f"Carpeta: {md.get('source_dir','')}\n\n"
                f"[CONTENIDO]\n{content}"
            )
            blocks.append(block)
        return "\n\n".join(blocks)

    def snippet(text: str, n: int = 900) -> str:
        """Acorta texto largo para vistas previas en la UI."""
        t = (text or "").strip().replace("\r\n", "\n")
        return t if len(t) <= n else t[:n] + "…"

    def _norm_status(val: str) -> str:
        """Normaliza variantes de ‘status’ a valores canónicos del índice."""
        mapping = {
            "mantiene": "Mantiene",
            "observa": "Observa",
            "observado": "Observado",
            "no formula": "No formula",
            "noformula": "No formula",
            "levanta": "Levanta",
        }
        key = (val or "").strip().lower()
        return mapping.get(key, val)

    def build_metadata_filter(filt: Dict[str, Any]) -> Dict[str, Any]:
        """Convierte filtros del usuario en metadatos normalizados (para where/cliente)."""
        md: Dict[str, Any] = {}
        if not isinstance(filt, dict):
            return md
        if filt.get("materia"):
            md["materia"] = str(filt["materia"]).strip()
        if filt.get("status"):
            md["status"] = _norm_status(filt["status"])
        if filt.get("date"):
            md["date"] = str(filt["date"]).strip()
        if filt.get("expediente"):
            md["expediente"] = str(filt["expediente"]).strip()
        if filt.get("carpeta"):
            md["source_dir"] = str(filt["carpeta"]).strip()
        return md

    @st.cache_resource(show_spinner=False)
    def get_embeddings(model_name: str):
        """Carga embeddings HF cacheados para todo el ciclo de vida de la app."""
        return HuggingFaceEmbeddings(model_name=model_name)

    @st.cache_resource(show_spinner=False)
    def get_llm(model_name: str):
        """Inicializa el modelo de chat (OpenAI) con baja temperatura."""
        try:
            return init_chat_model(model_name, model_provider="openai", model_kwargs={"temperature": 0})
        except TypeError:
            return init_chat_model(model_name, model_provider="openai")

    @st.cache_resource(show_spinner=False)
    def get_vdb(embeddings, collection_name: str, persist_directory: str):
        """Crea/abre la colección Chroma y realiza un warmup ligero."""
        vdb = ChromaVectorDB(
            embeddings=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        try:
            _ = vdb.similarity_search("__warmup__", k=1)
        except Exception:
            pass
        return vdb

    with st.sidebar:
        st.subheader("Parámetros")
        collection = st.text_input("Colección", value=DEFAULT_COLLECTION)
        chroma_dir = st.text_input("Chroma dir", value=DEFAULT_CHROMA_DIR)
        embeddings_model = st.text_input("Embeddings HF", value=DEFAULT_EMBEDDINGS)
        llm_model = st.text_input("LLM (OpenAI)", value=DEFAULT_LLM_MODEL)
        top_k = st.slider("k documentos", min_value=1, max_value=5, value=1, step=1)
        use_mmr = st.toggle("Usar MMR", value=False)
        show_debug = st.toggle("Depuración", value=False)
        if st.button("Limpiar historial"):
            st.session_state.pop("history", None)
            st.rerun()

    with st.spinner("Inicializando…"):
        if not os.environ.get("OPENAI_API_KEY"):
            st.warning("OPENAI_API_KEY no está definido.")
        embeddings = get_embeddings(embeddings_model)
        llm = get_llm(llm_model)
        vdb = get_vdb(embeddings, collection, chroma_dir)

    enrich_prompt = build_enrichment_prompt()
    gen_prompt = build_generation_prompt()

    if "history" not in st.session_state:
        st.session_state["history"] = []

    st.title("⚖️ Chat RAG — Base Chroma")
    user_question = st.chat_input("Escribe tu pregunta…")

    def enrich_question(q: str) -> Dict[str, Any]:
        """Llama al LLM para extraer JSON (expanded_query, keywords, entities, filters)."""
        raw_text = llm.invoke(enrich_prompt.format(question=q)).content or ""
        txt = raw_text.strip()
        if txt.startswith("```"):
            lines = txt.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            txt = "\n".join(lines).strip()
        try:
            data = json.loads(txt)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", txt)
            if m:
                try:
                    data = json.loads(m.group(0))
                except Exception:
                    data = {}
            else:
                data = {}
        if not isinstance(data, dict):
            data = {}
        expanded = data.get("expanded_query") if isinstance(data.get("expanded_query"), str) else q
        keywords = data.get("keywords") if isinstance(data.get("keywords"), list) else []
        entities = data.get("entities") if isinstance(data.get("entities"), list) else []
        filters = data.get("filters") if isinstance(data.get("filters"), dict) else {
            "materia": "", "status": "", "date": "", "expediente": ""
        }
        parts = [expanded or q]
        if keywords:
            parts.append(" ".join([str(k) for k in keywords if k]))
        if entities:
            parts.append(" ".join([str(e) for e in entities if e]))
        final_query = " ".join([p for p in parts if p]).strip()
        md_filter = build_metadata_filter(filters)
        chroma_where = build_chroma_where(md_filter)
        return {
            "data": {
                "expanded_query": expanded,
                "keywords": keywords,
                "entities": entities,
                "filters": filters,
                "_raw": raw_text,
            },
            "final_query": final_query,
            "metadata_filter": md_filter,
            "chroma_where": chroma_where,
        }

    def answer_once(q: str, k: int):
        """Ejecuta un ciclo de pregunta: enrich → retrieve (con filtro) → generar → guardar en historial."""
        enriched = enrich_question(q)
        final_q = enriched["final_query"]
        md_filter = enriched["metadata_filter"]
        chroma_where = build_chroma_where(md_filter)
        fetch_more = max(k * 5, 20)
        if use_mmr:
            candidates = vdb.mmr_search(
                final_q,
                k=fetch_more,
                fetch_k=max(fetch_more * 2, 40),
                metadata_filter=chroma_where or None,
            )
        else:
            candidates = vdb.similarity_search(
                final_q, k=fetch_more, metadata_filter=chroma_where or None
            )
        date_pred = build_date_predicate(md_filter.get("date", ""))
        filtered = [d for d in candidates if date_pred((d.metadata or {}).get("date", ""))]
        docs = (filtered[:k] if filtered else candidates[:k])
        ctx = format_docs_for_prompt(docs)
        msg = gen_prompt.format(question=q, context=ctx)
        resp = llm.invoke(msg)
        ans = (resp.content or "").strip()
        st.session_state["history"].insert(
            0,
            {
                "q": q,
                "a": ans,
                "docs": docs,
                "enriched": enriched["data"],
                "filter": md_filter,
                "where": chroma_where,
                "final_query": final_q,
            },
        )

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            with st.spinner("Buscando y redactando…"):
                answer_once(user_question, k=top_k)

    for turn in st.session_state["history"]:
        with st.chat_message("user"):
            st.markdown(turn["q"])
        with st.chat_message("assistant"):
            st.markdown(turn["a"])
            with st.expander("📎 Documentos utilizados (ver)"):
                for i, d in enumerate(turn["docs"], start=1):
                    md = d.metadata or {}
                    st.markdown(f"**{i}. {md.get('source_file','(sin nombre)')} — {md.get('section','')}**")
                    st.caption(
                        f"Fecha: {md.get('date','')} • Expediente: {md.get('expediente','')} • "
                        f"Entrada: {md.get('entrada','')} • Materia: {md.get('materia','')} • "
                        f"Status: {md.get('status','')}"
                    )
                    st.code(snippet(d.page_content, 900), language="markdown")
                    st.divider()

            with st.expander("🔎 Ver pregunta enriquecida"):
                enriched = turn.get("enriched", {}) or {}
                st.write("**Expanded query:**")
                st.code(enriched.get("expanded_query", "") or "—", language="text")
                st.write("**Final query usada en retrieval:**")
                st.code(turn.get("final_query", "") or "—", language="text")

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Keywords**")
                    kws = enriched.get("keywords", []) or []
                    st.write(", ".join(map(str, kws)) or "—")
                with col2:
                    st.write("**Entities**")
                    ents = enriched.get("entities", []) or []
                    st.write(", ".join(map(str, ents)) or "—")

                st.write("**Filtros aplicados**")
                st.json(turn.get("filter", {}) or {})

                st.write("**Where (Chroma) aplicado**")
                st.json(turn.get("where", {}) or {})

            if show_debug:
                with st.expander("🔎 Depuración"):
                    st.write("**Consulta final enriquecida (texto para retrieval):**")
                    st.code(turn.get("final_query", ""), language="text")
                    st.write("**Filtro de metadatos aplicado:**")
                    st.json(turn.get("filter", {}))
                    st.write("**JSON crudo de enriquecimiento (LLM):**")
                    st.json(turn.get("enriched", {}))

if __name__ == "__main__":
    app()