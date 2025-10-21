# tools/seed_langsmith_dataset.py
# -*- coding: utf-8 -*-
"""
Crea/actualiza un dataset en LangSmith con ejemplos Q&A desde eval/uru_qa_dataset.jsonl

Uso:
    uv run python tools/seed_langsmith_dataset.py --name uru-qa-v1 --file eval/uru_qa_dataset.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import os

from dotenv import load_dotenv

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def get_or_create_dataset(client, name: str, description: str):
    """Reutiliza si existe; si no, crea."""
    try:
        ds = client.read_dataset(dataset_name=name)
        return ds, False
    except Exception:
        ds = client.create_dataset(dataset_name=name, description=description)
        return ds, True

def main():
    load_dotenv()  # asegura que leemos .env
    from langsmith import Client

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Nombre del dataset en LangSmith (p.ej., uru-qa-v1)")
    parser.add_argument("--file", required=True, help="Ruta del JSONL con los ejemplos")
    args = parser.parse_args()

    # Diagnóstico mínimo
    have_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    print(f"[LangSmith] API key presente: {'sí' if have_key else 'no'}")
    print(f"[LangSmith] Endpoint: {os.getenv('LANGCHAIN_ENDPOINT') or os.getenv('LANGSMITH_ENDPOINT') or '(default)'}")

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {file_path}")

    # Si quieres forzar explícito (opcional):
    client = Client(
        api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        api_key=os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY"),
    )

    dataset, created = get_or_create_dataset(client, args.name, "Q&A de jurisprudencia Uruguay (RAG Jurídico)")
    print(f"Dataset listo: {dataset.name} (id={dataset.id})  [{'CREATED' if created else 'EXISTS'}]")

    rows = load_jsonl(file_path)
    for i, row in enumerate(rows, 1):
        inp = row.get("input", {})
        exp = row.get("expected", "")
        meta = row.get("meta", {})

        client.create_example(
            inputs=inp,
            outputs={"answer": exp},
            metadata=meta,
            dataset_id=dataset.id,   # <-- atributo, no dict
        )
        print(f"[{i}] ejemplo agregado: {inp.get('question','(sin pregunta)')[:80]}...")

    print("Listo. Revisa el dataset en LangSmith.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelado por el usuario.")
    except Exception as e:
        print(f"Error: {e}")
        raise
