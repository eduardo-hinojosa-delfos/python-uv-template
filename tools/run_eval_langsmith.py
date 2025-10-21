# tools/run_eval_langsmith.py
# -- coding: utf-8 --
"""
Evalúa tu RAG contra un dataset de LangSmith (p.ej. 'uru-qa-v1') sin importar funciones de main.py.

Uso:
    uv run python tools/run_eval_langsmith.py --dataset uru-qa-v1 --k 10
"""

import argparse
import os
import re
import unicodedata
from typing import Dict, Any, List, Optional, Tuple

from dotenv import load_dotenv

# LangSmith
from langsmith import Client
from langsmith.run_helpers import traceable

# LangChain / LLMs / Embeddings / Docs
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Tu wrapper a Chroma
from python_uv_template.vector_db import ChromaVectorDB


# =========================
# Config por defecto
# =========================
DEFAULT_CHROMA_DIR = "chroma_db"
DEFAULT_COLLECTION = "tc_uru"
DEFAULT_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-4o-mini"


# =========================
# Utilidades de normalización / métricas
# =========================
def norm_txt(t: str) -> str:
    t = (t or "").strip().lower()
    t = unicodedata.normalize("NFKD", t)
    t = re.sub(r"\s+", " ", t)
    return t

def metric_exact_or_contains(pred: str, exp: str) -> float:
    p = norm_txt(pred)
    e = norm_txt(exp)
    if not e:
        return 0.0
    if p == e:
        return 1.0
    if e in p:
        return 0.9
    return 0.0

def metric_must_contain(pred: str, phrases: List[str]) -> float:
    if not phrases:
        return 1.0
    p = norm_txt(pred)
    ok = sum(1 for ph in phrases if norm_txt(ph) in p)
    return ok / len(phrases)


# =========================
# Prompts (idénticos a tu app)
# =========================
def build_enrichment_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["question"],
        template=(
            "Tu rol es el de un Analista Legal especializado en estructuras jurisprudenciales.\n"
            "Tu tarea es orientar la recuperación de evidencia *dentro de las secciones formales* de resoluciones/jurisprudencias, "
            "limitándote únicamente a estas cuatro secciones:\n\n"
            "1) VISTO → Origen del expediente/asunto; identificación del caso, entidad/persona involucrada y materia del reclamo/solicitud.\n"
            "2) RESULTANDO → Hechos y antecedentes, cronología de actuaciones, resoluciones previas y fundamentos fácticos.\n"
            "3) CONSIDERANDO → Fundamentos jurídicos y análisis normativo; razones de derecho e interpretación de normas.\n"
            "4) ATENTO → Enumeración de normas legales y fundamentos normativos que soportan la resolución; puente hacia la decisión final.\n\n"
            "Comportamiento esperado:\n"
            "- Si la pregunta del usuario *corresponde claramente* a alguna de esas secciones, identifica cuál.\n"
            "- Si *no corresponde explícitamente* a esas secciones (p.ej., decisión final como RESUELVE/DICTAMINA), deja la sección vacía.\n"
            "- No inventes contenido jurídico ni emitas juicios legales: tu función es orientar la ubicación estructural.\n\n"
            "Devuelve SOLO un JSON con esta estructura EXACTA (sin texto extra):\n"
            "{{\n"
            '  "expanded_query": "pregunta reescrita",\n'
            '  "keywords": ["k1","k2"],\n'
            '  "entities": ["entidad1","entidad2"],\n'
            '  "filters": {{\n'
            '    "materia": "",\n'
            '    "status": "",\n'
            '    "date": "",\n'
            '    "expediente": "",\n'
            '    "carpeta": "",\n'
            '    "seccion": ""\n'
            "  }}\n"
            "}}\n\n"
            "Instrucciones para filters.seccion:\n"
            '- Usa SOLO uno de: "VISTO", "RESULTANDO", "CONSIDERANDO", "ATENTO".\n'
            '- Si no aplica, usa cadena vacía "". \n'
            "- filters.date en ISO (YYYY, YYYY-MM o YYYY-MM-DD); vacío si no hay.\n\n"
            "Pregunta del usuario:\n"
            "{question}\n"
        ),
    )

def build_generation_prompt() -> PromptTemplate:
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


# =========================
# Helpers de metadatos/fechas (copiados de tu app)
# =========================
def _norm_section(val: str) -> str:
    if not val:
        return ""
    if val in ("VISTO", "RESULTANDO", "CONSIDERANDO", "ATENTO"):
        return val
    key = (val or "").strip().lower()
    mapping = {
        "visto": "VISTO", "vistos": "VISTO",
        "resultando": "RESULTANDO", "resultandos": "RESULTANDO",
        "considerando": "CONSIDERANDO", "considerandos": "CONSIDERANDO",
        "atento": "ATENTO", "atentos": "ATENTO",
    }
    return mapping.get(key, val)

def _norm_status(val: str) -> str:
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
    if filt.get("seccion"):
        md["section"] = _norm_section(str(filt["seccion"]))
    return md

def build_chroma_where(md: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
    if md.get("section"):
        clauses.append({"section": {"$eq": str(md["section"]).strip()}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}

def build_date_predicate(date_filter_str: str):
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

def format_docs_for_prompt(docs: List[Document], max_chars_per_doc: int = 1800) -> str:
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


# =========================
# Recursos (embeddings, LLM, VDB)
# =========================
def get_embeddings(model_name: str = DEFAULT_EMBEDDINGS):
    return HuggingFaceEmbeddings(model_name=model_name)

def get_llm(model_name: str = DEFAULT_LLM_MODEL):
    try:
        return init_chat_model(model_name, model_provider="openai", model_kwargs={"temperature": 0})
    except TypeError:
        return init_chat_model(model_name, model_provider="openai")

def get_vdb(embeddings, collection_name: str = DEFAULT_COLLECTION, persist_directory: str = DEFAULT_CHROMA_DIR):
    vdb = ChromaVectorDB(
        embeddings=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    try:
        _ = vdb.similarity_search("_warmup_", k=1)
    except Exception:
        pass
    return vdb


# =========================
# RAG core: rag_infer (standalone)
# =========================
def rag_infer(
    question: str,
    *,
    k: int,
    enrich_enabled: bool,
    use_mmr: bool,
    vdb,
    llm,
    enrich_prompt,
    gen_prompt,
) -> Dict[str, Any]:
    import json as _json

    # Enriquecimiento robusto
    try:
        if enrich_enabled:
            raw_text = llm.invoke(enrich_prompt.format(question=question)).content or ""
            txt = raw_text.strip()
            if txt.startswith(""):
                lines = txt.splitlines()
                if lines and lines[0].startswith(""):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                txt = "\n".join(lines).strip()
            try:
                data = _json.loads(txt)
            except Exception:
                m = re.search(r"\{[\s\S]*\}", txt)
                data = _json.loads(m.group(0)) if m else {}
        else:
            data = {}
        if not isinstance(data, dict):
            data = {}
        expanded = data.get("expanded_query") if isinstance(data.get("expanded_query"), str) else question
        keywords = data.get("keywords") if isinstance(data.get("keywords"), list) else []
        entities = data.get("entities") if isinstance(data.get("entities"), list) else []
        filters = data.get("filters") if isinstance(data.get("filters"), dict) else {
            "materia": "", "status": "", "date": "", "expediente": "", "carpeta": "", "seccion": ""
        }
        parts = [expanded or question]
        if keywords:
            parts.append(" ".join([str(k_) for k_ in keywords if k_]))
        if entities:
            parts.append(" ".join([str(e_) for e_ in entities if e_]))
        final_q = " ".join([p for p in parts if p]).strip()
        md_filter = build_metadata_filter(filters)
        chroma_where = build_chroma_where(md_filter)
        enriched_data = {
            "expanded_query": expanded,
            "keywords": keywords,
            "entities": entities,
            "filters": filters,
            "_raw": "(ok)" if enrich_enabled else "(sin enriquecimiento: fallback)",
        }
    except Exception:
        final_q = question
        md_filter = {}
        chroma_where = None
        enriched_data = {
            "expanded_query": question,
            "keywords": [],
            "entities": [],
            "filters": {"materia": "", "status": "", "date": "", "expediente": "", "carpeta": "", "seccion": ""},
            "_raw": "(sin enriquecimiento: error/fallback)",
        }

    # Retrieval
    fetch_more = max(k * 5, 20)
    if use_mmr:
        try:
            pairs = vdb.mmr_search_with_score(
                final_q, k=fetch_more, fetch_k=max(fetch_more * 2, 40), metadata_filter=chroma_where or None
            )
        except AttributeError:
            _docs = vdb.mmr_search(
                final_q, k=fetch_more, fetch_k=max(fetch_more * 2, 40), metadata_filter=chroma_where or None
            )
            pairs = [(d, None) for d in _docs]
    else:
        try:
            pairs = vdb.similarity_search_with_score(final_q, k=fetch_more, metadata_filter=chroma_where or None)
        except AttributeError:
            _docs = vdb.similarity_search(final_q, k=fetch_more, metadata_filter=chroma_where or None)
            pairs = [(d, None) for d in _docs]

    date_pred = build_date_predicate(md_filter.get("date", ""))
    filtered_pairs = [(d, s) for (d, s) in pairs if date_pred((d.metadata or {}).get("date", ""))]

    docs_with_scores: List[Tuple[Document, Optional[float]]] = (filtered_pairs[:k] if filtered_pairs else pairs[:k])
    docs = [d for (d, _) in docs_with_scores]
    ctx = format_docs_for_prompt(docs)

    # Generación
    msg = gen_prompt.format(question=question, context=ctx)
    resp = llm.invoke(msg)
    answer = (resp.content or "").strip()

    return {
        "answer": answer,
        "docs_with_scores": docs_with_scores,
        "final_query": final_q,
        "filter": md_filter,
        "where": chroma_where,
        "enriched": enriched_data,
    }


# =========================
# Evaluación LangSmith
# =========================
@traceable(run_type="chain", name="Eval-RAG-Juridico")
def evaluate_example(question: str, expected: str, meta: Dict[str, Any], *, k: int) -> Dict[str, Any]:
    enrich_prompt = build_enrichment_prompt()
    gen_prompt = build_generation_prompt()
    embeddings = get_embeddings(DEFAULT_EMBEDDINGS)
    llm = get_llm(DEFAULT_LLM_MODEL)
    vdb = get_vdb(embeddings, DEFAULT_COLLECTION, DEFAULT_CHROMA_DIR)

    out = rag_infer(
        question, k=k, enrich_enabled=True, use_mmr=False,
        vdb=vdb, llm=llm, enrich_prompt=enrich_prompt, gen_prompt=gen_prompt
    )
    pred = out["answer"]

    m1 = metric_exact_or_contains(pred, expected)
    musts = meta.get("must_contain", [])
    m2 = metric_must_contain(pred, musts)

    return {
        "prediction": pred,
        "expected": expected,
        "scores": {"exact_or_contains": m1, "must_contain": m2, "combined": (m1 * 0.6 + m2 * 0.4)},
        "meta": meta,
    }


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Nombre del dataset en LangSmith (p.ej., uru-qa-v1)")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    client = Client()
    ds = client.read_dataset(dataset_name=args.dataset)
    print(f"Evaluando dataset: {ds.name}")
    examples = list(client.list_examples(dataset_id=ds.id))
    print(f"Ejemplos: {len(examples)}")

    wins = 0
    for ex in examples:
        q = ex.inputs.get("question", "")
        expected = ex.outputs.get("answer", "")
        meta = ex.metadata or {}

        result = evaluate_example(q, expected, meta, k=args.k)

        combined = result["scores"]["combined"]
        passed = combined >= 0.8
        wins += 1 if passed else 0

        client.create_feedback(
            key="combined_score",
            score=combined,
            comment=f"exact_or_contains={result['scores']['exact_or_contains']:.2f}, must_contain={result['scores']['must_contain']:.2f}",
        )
        client.create_feedback(
            key="passed",
            score=1.0 if passed else 0.0,
            comment="pass/fail threshold 0.8"
        )

        print(f"- Q: {q[:70]}...")
        print(f"  pred: {result['prediction'][:120]}...")
        print(f"  expected: {expected}")
        print(f"  scores: {result['scores']}")
        print(f"  PASS={passed}\n")

    print(f"Resumen: {wins}/{len(examples)} PASSED")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelado por el usuario.")
    except Exception as e:
        print(f"Error: {e}")
        raise