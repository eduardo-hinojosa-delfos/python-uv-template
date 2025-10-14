"""
qa_pipeline.py
Consulta una base ChromaDB ya persistida y ejecuta recuperación + generación de respuesta.
No reindexa.
"""

import os
import getpass
from typing import List

from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import START, StateGraph

from python_uv_template.vector_db import ChromaVectorDB


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    references: str


def build_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["question", "context"],
        template="""
Responde de forma breve y directa utilizando EXCLUSIVAMENTE la información de los documentos recuperados.

Cada documento puede contener dos partes:
1) Metadatos (fecha, expediente, entrada, materia, status).
2) Contenido textual.

Instrucciones:
1) Revisa primero los METADATOS.
2) Si no basta, usa el CONTENIDO.
3) Si piden fundamentos/causas: prioriza CONSIDERANDO; si no está, revisa “EL TRIBUNAL ACUERDA”/“ATENTO” con referencias al CONSIDERANDO; por último RESULTANDO.
4) Para fundamentos o citas, usa comillas con texto EXACTO.
5) Si no hay información suficiente, responde: "No se encontró información suficiente".
6) No agregues información externa.

Pregunta:
{question}

Documentos recuperados (metadatos y contenido):
{context}

Respuesta concisa:

""",
    )


def main() -> None:
    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vdb = ChromaVectorDB(
        embeddings=embeddings,
        collection_name="tc_uru",
        persist_directory="chroma_db",
    )

    prompt = build_prompt()

    def retrieve(state: State):
        folder = (os.getenv("RAG_SOURCE_DIR") or "").strip()
        mf = {"source_dir": {"$eq": folder}} if folder else None
        docs = vdb.similarity_search(state["question"], k=3, metadata_filter=mf)
        return {"context": docs}

    def generate(state: State):
        docs_content = "\n\n".join(
            f"[METADATOS]\n"
            f"Fecha: {doc.metadata.get('date','')}\n"
            f"Expediente: {doc.metadata.get('expediente','')}\n"
            f"Entrada: {doc.metadata.get('entrada','')}\n"
            f"Materia: {doc.metadata.get('materia','')}\n"
            f"Status: {doc.metadata.get('status','')}\n"
            f"Archivo: {doc.metadata.get('source_file','')}\n"
            f"Sección: {doc.metadata.get('section','')}\n"
            f"Carpeta: {doc.metadata.get('source_dir','')}\n\n"
            f"[CONTENIDO]\n{doc.page_content}"
            for doc in state["context"]
        )
        message = prompt.format(question=state["question"], context=docs_content)
        response = llm.invoke(message)

        refs = "\n\n".join(
            f"- Archivo: {doc.metadata.get('source_file','')} | "
            f"Sección: {doc.metadata.get('section','')} | "
            f"Fecha: {doc.metadata.get('date','')} | "
            f"Materia: {doc.metadata.get('materia','')} | "
            f"Status: {doc.metadata.get('status','')} | "
            f"Carpeta: {doc.metadata.get('source_dir','')}"
            for doc in state["context"]
        )
        return {"answer": response.content.strip(), "references": refs}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    print("\n=== QA sobre VDB (Chroma) ===")
    question = input("Escribe tu pregunta: ").strip()
    result = graph.invoke({"question": question})

    print(f"\nRespuesta concisa:\n{result.get('answer')}\n")
    show_refs = input("¿Quieres ver las referencias? (s/n): ").strip().lower()
    if show_refs == "s":
        print("\n=== Referencias ===\n")
        print(result.get("references", "No se encontraron referencias."))


if __name__ == "__main__":
    main()