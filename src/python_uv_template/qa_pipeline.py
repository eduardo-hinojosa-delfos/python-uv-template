"""
qa_pipeline.py
Abre la vdb persistida en Chroma y realiza SOLO retrieval + respuesta.
No re-indexa. Ideal para correr muchas veces.
"""

import os
import getpass
from dotenv import load_dotenv
from typing_extensions import TypedDict, List

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
Responde de forma breve y directa a la siguiente pregunta utilizando EXCLUSIVAMENTE la información de los documentos recuperados.

Cada documento puede contener dos partes:
1) Metadatos (p. ej.: fecha, expediente, entrada, materia, status).
2) Contenido textual del documento.

INSTRUCCIONES DE ANÁLISIS:
1. Primero revisa la información en los **METADATOS** para ver si allí se encuentra la respuesta.
2. Si la información no está completamente en los metadatos, revisa el **CONTENIDO TEXTUAL**.
3. Si la pregunta pide **fundamentos, causas, motivos o argumentos**, aplica esta prioridad de búsqueda:
   - **Primero** busca la respuesta en la sección **CONSIDERANDO**.
   - **Si no se encuentra**, revisa si en **“EL TRIBUNAL ACUERDA”** o **“ATENTO”** hay referencias explícitas al CONSIDERANDO (por ejemplo: “(Considerando N°4)”) y dirígete a ese considerando citado.
   - **Como última opción**, busca en **RESULTANDO** o secciones equivalentes que puedan contener hechos o antecedentes relacionados.
4. Si la pregunta pide fundamentos o citas, **usa citas textuales EXACTAS entre comillas**, copiando el texto tal cual aparece en el documento (sin parafrasear ni resumir).
5. Si no hay información suficiente o clara para responder con certeza, responde exactamente:  
   👉 `"No se encontró información suficiente"`.
6. No agregues información externa, interpretaciones ni inferencias fuera del contexto proporcionado.

Pregunta:
{question}

Documentos recuperados (metadatos y contenido):
{context}

Respuesta concisa:

""",
    )


def main() -> None:
    load_dotenv()

    # API key para el LLM
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    # LLM y embeddings (usa el mismo modelo de embeddings que en ingest)
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Conecta con la vdb YA PERSISTIDA (no reindexa)
    vdb = ChromaVectorDB(
        embeddings=embeddings,
        collection_name="tc_uru",
        persist_directory="chroma_db",
    )

    prompt = build_prompt()

    # --------- nodos del grafo ---------
    def retrieve(state: State):
        # Puedes incorporar filtros por metadatos: e.g., filter={"status": "Mantiene"}
        docs = vdb.similarity_search(state["question"], k=5)
        #docs = vdb.mmr_search(state["question"], k=5)
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
            f"Sección: {doc.metadata.get('section','')}\n\n"
            f"[CONTENIDO]\n{doc.page_content}"
            for doc in state["context"]
        )
        message = prompt.format(question=state["question"], context=docs_content)
        response = llm.invoke(message)

        refs = "\n\n".join(
            f"- **Archivo:** {doc.metadata.get('source_file','')} | "
            f"Sección: {doc.metadata.get('section','')} | "
            f"Fecha: {doc.metadata.get('date','')} | "
            f"Materia: {doc.metadata.get('materia','')} | "
            f"Status: {doc.metadata.get('status','')}"
            for doc in state["context"]
        )
        return {"answer": response.content.strip(), "references": refs}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    # --------- demo interactiva ---------
    print("\n=== QA sobre VDB persistida (Chroma) ===")
    question = input("Escribe tu pregunta: ").strip()
    result = graph.invoke({"question": question})

    print(f"\nRespuesta concisa:\n{result.get('answer')}\n")
    show_refs = input("¿Quieres ver las referencias? (s/n): ").strip().lower()
    if show_refs == "s":
        print("\n=== Referencias ===\n")
        print(result.get("references", "No se encontraron referencias."))


if __name__ == "__main__":
    main()
