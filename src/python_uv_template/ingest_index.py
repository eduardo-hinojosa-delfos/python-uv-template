"""
ingest_index.py
Ingesta de .txt, segmentación legal (LegalChunkerLC), persistencia de chunks en disco
y indexación en ChromaDB (persistente).
"""

import os
import getpass
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model

from python_uv_template.ingesta import process_pdfs
from python_uv_template.chunking_langchain import LegalChunkerLC
from python_uv_template.vector_db import ChromaVectorDB


def main(
    *,
    pdf_folder: str = "pdf_documents",
    extracted_folder: str = "extracted_text",
    chunks_folder: str = "chunks",
    persist_chunks: bool = True,
) -> None:
    """Orquesta extracción, chunking, persistencia de chunks e indexación en ChromaDB."""
    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    process_pdfs(pdf_folder, extracted_folder)

    loader = DirectoryLoader(
        extracted_folder,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"[Ingest] Archivos leídos: {len(docs)}")

    chunker = LegalChunkerLC(
        chunk_size=1000,
        chunk_overlap=150,
        section_subchunking="none",
        llm=llm,
    )

    chunked_docs: List[Document] = []
    for d in docs:
        text = d.page_content
        file_name = d.metadata.get("source", "documento.txt")
        doc_meta = chunker._extract_document_metadata(text)
        chunked_docs.extend(
            chunker.create_documents_from_text(text, file_name, doc_meta)
        )

    print(f"[Ingest] Chunks generados: {len(chunked_docs)}")

    if not chunked_docs:
        print("[Ingest] No hay chunks para indexar. Verifique los .txt o la configuración del chunker.")
        return

    if persist_chunks:
        try:
            chunker.persist_documents(chunked_docs, output_folder=chunks_folder)
            print(f"[Ingest] Chunks persistidos en '{chunks_folder}'.")
        except Exception as e:
            print(f"[WARN] No se pudieron persistir los chunks en disco: {e}")

    vdb = ChromaVectorDB(
        embeddings=embeddings,
        collection_name="tc_uru",
        persist_directory="chroma_db",
    )

    vdb.reset()
    vdb.index(chunked_docs)
    vdb.persist()
    print("[Ingest] Indexación completa y persistida en 'chroma_db' (colección 'tc_uru').")


if __name__ == "__main__":
    main()