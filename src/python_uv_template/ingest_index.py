"""
ingest_index.py
Carga .txt, chunking legal (LegalChunkerLC) y indexación única en Chroma (persistente).
Ejecuta este archivo una sola vez (o cuando quieras reindexar).
"""

import os
import getpass
from dotenv import load_dotenv
from typing import List
from python_uv_template.ingesta import process_pdfs

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from python_uv_template.chunking_langchain import LegalChunkerLC
from python_uv_template.vector_db import ChromaVectorDB


def main() -> None:
    load_dotenv()
    pdf_folder = "pdf_documents"
    extracted_folder = "extracted_text"
    
    # API key (para otros pasos del pipeline si la necesitas luego)
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    # Embeddings (mantén el mismo modelo entre ingesta y QA)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    results = process_pdfs(pdf_folder, extracted_folder)


    # 1) Carga de documentos .txt
    loader = DirectoryLoader(
        "extracted_text",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    
    
    
    docs = loader.load()
    print(f"[Ingest] Archivos leídos: {len(docs)}")

    # 2) Chunking legal
    chunker = LegalChunkerLC(
        chunk_size=1000,
        chunk_overlap=150,
        section_subchunking="none",  # o "recursive"
    )

    chunked_docs: List[Document] = []
    for d in docs:
        text = d.page_content
        file_name = d.metadata.get("source", "documento.txt")
        doc_meta = chunker._extract_document_metadata(text)
        chunked_docs.extend(chunker.create_documents_from_text(text, file_name, doc_meta))

    print(f"[Ingest] Chunks generados: {len(chunked_docs)}")

    if not chunked_docs:
        print("[Ingest] No hay chunks para indexar. Revisa los .txt o el chunker.")
        return

    # 3) Indexación y persistencia en Chroma
    vdb = ChromaVectorDB(
        embeddings=embeddings,
        collection_name="tc_uru",
        persist_directory="chroma_db",
    )

    # Limpia colección anterior si quieres reindexar desde cero:
    # vdb.reset()

    vdb.index(chunked_docs)  # opcionalmente pasa IDs estables
    vdb.persist()
    print("[Ingest] Indexación completa y persistida en 'chroma_db' (colección 'tc_uru').")


if __name__ == "__main__":
    main()
