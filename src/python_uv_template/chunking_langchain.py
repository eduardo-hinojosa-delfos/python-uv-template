# legal_chunker_langchain.py
"""
Legal chunking en LangChain replicando la lógica de LlamaIndex.

Flujo:
1) Limpieza (pie de página y continuidad de líneas).
2) Detección de secciones legales (VISTO, RESULTANDO, CONSIDERANDO, ATENTO).
3) (Opcional) Sub-chunking por sección con RecursiveCharacterTextSplitter.
4) Extracción de metadatos (fecha, expediente, entrada, fecha_entrada).
5) Genera List[Document] (LangChain) y persiste a /chunks con cabecera de metadatos.

Parámetros clave:
- chunk_size, chunk_overlap: sólo aplican si section_subchunking="recursive".
- section_subchunking: "none" (1 chunk por sección, comportamiento por defecto) o "recursive".
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Si tienes disponible tu normalizador, perfecto; si no, se hace fallback grácil.
try:
    from python_uv_template.ingesta import normalize_date_es
except Exception:
    def normalize_date_es(_: str) -> Optional[str]:
        return None


@dataclass
class ChunkMetadata:
    section: str
    file: str
    chunk_index: int
    word_count: int
    char_count: int


class LegalChunkerLC:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        section_subchunking: str = "none",  # "none" (equivalente a tu LlamaIndex actual) o "recursive"
    ):
        self.section_patterns = {
            "VISTO": r"VISTO\s*:?",
            "RESULTANDO": r"RESULTANDO\s*:?",
            "CONSIDERANDO": r"CONSIDERANDO\s*:?",
            "ATENTO": r"ATENTO\s*:?",
        }
        self.section_subchunking = section_subchunking
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ".", " "],
        )

    # ---------- Limpiezas ----------
    def _strip_trailing_footer(self, text: str) -> str:
        """
        Elimina pie de página final si hay múltiples saltos y rutas/URLs.
        """
        pattern = re.compile(
            r"(\n\s*){2,}(?P<footer>([A-Za-z]:\\|/|\\\\|file:|https?://).+)$",
            re.IGNORECASE | re.DOTALL,
        )
        m = pattern.search(text)
        if m:
            return text[: m.start()].rstrip()
        return text

    def _merge_line_breaks_for_continuity(self, text: str) -> str:
        """
        Une saltos de línea intra-párrafo para continuidad, preservando dobles saltos como separadores de párrafo.
        Respeta abreviaturas y siglas punteadas.
        """
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        t = re.sub(r"(\w+)-\n(\w+)", r"\1\2", t)  # une palabras cortadas por guion al final de línea

        lines = t.split("\n")
        merged: List[str] = []
        buffer = ""
        abbrev_tokens = {
            "art.", "arts.", "cap.", "caps.", "pág.", "págs.", "sr.", "sra.", "dr.", "dra.",
            "lic.", "ing.", "etc.", "no.", "nº.", "núm.", "núm", "num.", "u$s.", "usd.",
        }
        dotted_acronym_re = re.compile(r"(?:[A-ZÁÉÍÓÚÑ]\.){2,}[A-ZÁÉÍÓÚÑ]?\.?$")

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if buffer:
                    merged.append(buffer)
                    buffer = ""
                merged.append("")
                continue

            if buffer:
                prev_end = buffer[-1] if buffer else ""
                starts_section = bool(re.match(
                    r"^(VISTO|RESULTANDO|CONSIDERANDO|ATENTO)\b", stripped, re.IGNORECASE
                ))
                prev_last_token = buffer.split()[-1].lower() if buffer.split() else ""
                next_starts_lower = bool(re.match(r"^[a-záéíóúñ]", stripped))
                ends_with_dotted_acronym = bool(dotted_acronym_re.search(buffer))
                is_abbrev = prev_last_token in abbrev_tokens

                should_merge = False
                if not starts_section:
                    if prev_end not in ".:;)":
                        should_merge = True
                    else:
                        if prev_end == "." and (ends_with_dotted_acronym or is_abbrev or next_starts_lower):
                            should_merge = True

                if should_merge:
                    buffer += " " + stripped
                else:
                    merged.append(buffer)
                    buffer = stripped
            else:
                buffer = stripped

        if buffer:
            merged.append(buffer)

        return "\n\n".join([p for p in merged if p != ""]) if merged else t

    # ---------- Extracción de metadatos de documento ----------
    def _extract_document_metadata(self, text: str) -> Dict[str, str]:
        # Mantiene la lógica original para fecha, expediente y entrada
        head = text[:3000]
        tail = text[-3000:]  # <-- NUEVO: también analizamos los últimos 3000 caracteres
        date_normalized = normalize_date_es(head) or ""

        expediente = ""
        m_exp = re.search(
            r"\b(Expediente|Carpeta)\b\s*N[°ºoO]?\s*[:\-]?\s*([\w./-]+)",
            head,
            re.IGNORECASE,
        )
        if m_exp:
            expediente = m_exp.group(2).strip()

        entrada = ""
        fecha_entrada = ""
        m_ent_fecha = re.search(
            r"\bEntrada\b\s*N[°ºoO]?\s*:\s*([\w./-]+)\s+DE\s+FECHA\s*:\s*(\d{1,2}/\d{1,2}/\d{4})",
            head,
            re.IGNORECASE,
        )
        if m_ent_fecha:
            entrada = m_ent_fecha.group(1).strip()
            dmy = m_ent_fecha.group(2)
            try:
                d, m, y = dmy.split("/")
                fecha_entrada = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
            except Exception:
                fecha_entrada = dmy
        else:
            m_ent = re.search(
                r"\b(Entrada|N[úu]mero\s+de\s+entrada)\b\s*(?:N[°ºoO])?\s*[:\-]*\s*([\w./-]+)",
                head,
                re.IGNORECASE,
            )
            if m_ent:
                entrada = m_ent.group(2).strip()

        if not fecha_entrada:
            fecha_entrada = "N/A"

        # --- NUEVO: Buscar MATERIA y STATUS al principio y al final ---
        materia = "Sin información"
        status = "Sin información"

        # Primero buscamos en todo el texto (por si están en medio)
        # Priorizamos tail para capturar datos al final
        search_area = tail + "\n" + head

        m_materia = re.search(
            r"\bMATERIA\s*[:\-]\s*(.+)", search_area, re.IGNORECASE
        )
        if m_materia:
            materia = m_materia.group(1).strip()

        m_status = re.search(
            r"\bSTATUS\s*[:\-]\s*(.+)", search_area, re.IGNORECASE
        )
        if m_status:
            status = m_status.group(1).strip()

        return {
            "date": date_normalized,
            "expediente": expediente,
            "entrada": entrada,
            "fecha_entrada": fecha_entrada,
            "materia": materia,
            "status": status,
        }


    # ---------- Normalización de ATENTO / "EL TRIBUNAL ACUERDA" ----------
    def _process_atento_acuerda_block(self, section_text: str) -> str:
        m = re.search(r"EL\s+TRIBUNAL\s+ACUERDA\s*:?", section_text, re.IGNORECASE)
        if not m:
            return section_text

        start = m.start()
        end = m.end()
        before = section_text[:start].rstrip()
        after = section_text[end:].lstrip("\n ")

        lines = after.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        list_lines: List[str] = []
        list_pattern = re.compile(
            r"^\s*(?:[-•*]\s+|\d+\)|\d+\.\s+|[IVXLCM]+\)\s+)", re.IGNORECASE
        )

        in_list = False
        for line in lines:
            if not line.strip():
                if in_list:
                    list_lines.append("")
                continue
            if list_pattern.match(line):
                in_list = True
                list_lines.append(line.rstrip())
                continue
            if in_list:
                break
            continue

        header_norm = "EL TRIBUNAL ACUERDA:"
        body = "\n".join(list_lines).strip()
        acordado = header_norm + "\n\n" + body if body else header_norm
        return (before + "\n\n" + acordado).strip()

    # ---------- Detección de secciones ----------
    def find_sections(self, text: str) -> List[Dict[str, str]]:
        cleaned = self._strip_trailing_footer(text)
        cleaned = self._merge_line_breaks_for_continuity(cleaned)

        header_regex = re.compile(
            r"^\s*(?:[IVXLCM]+\)|\d+\)|[-•]\s*)?\s*(VISTO|RESULTANDO|CONSIDERANDO|ATENTO)\s*:?",
            re.IGNORECASE | re.MULTILINE,
        )
        matches = [(m.group(1).upper(), m.start()) for m in header_regex.finditer(cleaned)]

        if not matches:
            return [{"name": "DOCUMENTO", "text": cleaned.strip()}]

        sections: List[Dict[str, str]] = []
        for i, (name, start) in enumerate(matches):
            end = matches[i + 1][1] if i + 1 < len(matches) else len(cleaned)
            section_text = cleaned[start:end].rstrip()
            sections.append({"name": name, "text": section_text})
        return sections

    # ---------- Creación de Documents ----------
    def create_documents_from_text(self, text: str, file_name: str, doc_meta: Dict[str, str]) -> List[Document]:
        sections = self.find_sections(text)
        documents: List[Document] = []

        for section in sections:
            section_text = section["text"]
            if section["name"].upper() == "ATENTO":
                section_text = self._process_atento_acuerda_block(section_text)

            if self.section_subchunking == "recursive":
                # sub-chunking por sección (si quieres volver al docstring original)
                # Nota: preservamos metadata y enumeramos chunk_index por sección
                sub_docs = self.splitter.create_documents([section_text])
                for idx, d in enumerate(sub_docs):
                    md = {
                        "section": section["name"],
                        "source_file": file_name,
                        "chunk_index": idx,
                        "word_count": len(d.page_content.split()),
                        "char_count": len(d.page_content),
                        "date": doc_meta.get("date", ""),
                        "expediente": doc_meta.get("expediente", ""),
                        "entrada": doc_meta.get("entrada", ""),
                        "fecha_entrada": doc_meta.get("fecha_entrada", ""),
                    }
                    documents.append(Document(page_content=d.page_content, metadata=md))
            else:
                # 1 chunk por sección (comportamiento actual equivalente a tu LlamaIndex)
                md = {
                    "section": section["name"],
                    "source_file": file_name,
                    "chunk_index": 0,
                    "word_count": len(section_text.split()),
                    "char_count": len(section_text),
                    "date": doc_meta.get("date", ""),
                    "expediente": doc_meta.get("expediente", ""),
                    "entrada": doc_meta.get("entrada", ""),
                    "fecha_entrada": doc_meta.get("fecha_entrada", ""),
                }
                documents.append(Document(page_content=section_text, metadata=md))

        return documents

    # ---------- Persistencia a /chunks ----------
    def persist_documents(self, documents: List[Document], output_folder: str = "chunks") -> None:
        out_path = Path(output_folder)
        out_path.mkdir(exist_ok=True)

        for doc in documents:
            section = str(doc.metadata.get("section", "SECCION")).lower().replace(" ", "-")
            base_name = Path(str(doc.metadata.get("source_file", "documento"))).stem
            idx = int(doc.metadata.get("chunk_index", 0))
            out_name = f"{base_name}_{section}_{idx:03d}.txt"
            out_file = out_path / out_name
            with open(out_file, "w", encoding="utf-8") as f:
                f.write("=== METADATOS ===\n")
                f.write(f"Archivo: {doc.metadata.get('source_file','')}\n")
                f.write(f"Sección: {doc.metadata.get('section','')}\n")
                f.write(f"Chunk: {idx}\n")
                f.write(f"Palabras: {doc.metadata.get('word_count',0)}\n")
                f.write(f"Caracteres: {doc.metadata.get('char_count',0)}\n")
                f.write(f"Fecha: {doc.metadata.get('date','')}\n")
                f.write(f"Expediente: {doc.metadata.get('expediente','')}\n")
                f.write(f"Entrada: {doc.metadata.get('entrada','')}\n")
                f.write(f"Fecha entrada: {doc.metadata.get('fecha_entrada','')}\n")
                f.write("\n=== CONTENIDO ===\n")
                f.write(doc.page_content)

    # ---------- Pipeline principal ----------
    def process_txt_folder(
        self,
        txt_folder: str = "extracted_text",
        output_folder: str = "chunks",
        persist: bool = True,
    ) -> List[Document]:
        txt_path = Path(txt_folder)
        txt_files = sorted(list(txt_path.glob("*.txt")))

        if not txt_files:
            print("No se encontraron archivos .txt para chunking.")
            return []

        print(f"\nProcesando {len(txt_files)} archivos TXT...\n")
        all_docs: List[Document] = []

        for txt_file in txt_files:
            try:
                text = txt_file.read_text(encoding="utf-8")
            except Exception as e:
                print(f"[WARN] No se pudo leer {txt_file.name}: {e}")
                continue

            try:
                doc_meta = self._extract_document_metadata(text)
                docs = self.create_documents_from_text(text, txt_file.name, doc_meta)
                all_docs.extend(docs)
                print(f"{txt_file.name}: {len(docs)} chunk(s) generado(s).")
            except Exception as e:
                print(f"[ERROR] Falló el chunking en {txt_file.name}: {e}")

        print(f"\nTotal de chunks generados: {len(all_docs)}")

        if persist:
            try:
                self.persist_documents(all_docs, output_folder=output_folder)
            except Exception as e:
                print(f"[WARN] No se pudieron persistir los chunks en disco: {e}")

        return all_docs


# ---------- Función de conveniencia ----------
def generate_langchain_chunks_from_txt(
    txt_folder: str = "extracted_text",
    output_folder: str = "chunks",
    section_subchunking: str = "none",  # "none" (igual a tu implementación actual) o "recursive"
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    persist: bool = True,
) -> List[Document]:
    """
    Construye y ejecuta el pipeline de chunking en LangChain.
    """
    chunker = LegalChunkerLC(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        section_subchunking=section_subchunking,
    )
    return chunker.process_txt_folder(
        txt_folder=txt_folder,
        output_folder=output_folder,
        persist=persist,
    )


# ---------- CLI / main ----------
if __name__ == "__main__":
    try:
        # Por defecto conserva 1 chunk por sección (como tu LlamaIndex actual).
        # Si quieres sub-chunking por sección, usa section_subchunking="recursive".
        docs = generate_langchain_chunks_from_txt(
            txt_folder="extracted_text",
            output_folder="chunks",
            section_subchunking="none",
            chunk_size=1000,
            chunk_overlap=150,
            persist=True,
        )
        print(f"OK: {len(docs)} Document(s) generados.")
    except Exception as e:
        print(f"[FATAL] Error en ejecución: {e}")
