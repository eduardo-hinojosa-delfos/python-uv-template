"""
Segmentación legal en LangChain con lógica compatible con esquemas comunes en documentos jurídicos.

Flujo general:
1) Limpieza (pie de página y continuidad de líneas).
2) Detección de secciones: VISTO, RESULTANDO, CONSIDERANDO, ATENTO.
3) Sub-segmentación opcional por sección mediante RecursiveCharacterTextSplitter.
4) Extracción de metadatos (fecha, expediente, entrada, fecha_entrada, materia, status).
5) Creación de List[Document] y persistencia en /chunks con cabecera de metadatos.

Parámetros:
- section_subchunking: "none" (1 chunk por sección) o "recursive".
- chunk_size y chunk_overlap: aplican si section_subchunking="recursive".
"""

import re
import unicodedata
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

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
    """
    Constructor de chunks legales a partir de texto plano, con limpieza previa,
    detección de secciones y enriquecimiento de metadatos.
    """
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        section_subchunking: str = "none",
        llm=None,
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
        self.llm = llm

    # Limpiezas
    def _strip_trailing_footer(self, text: str) -> str:
        """Suprime pies de página finales con múltiples saltos y rutas/URLs."""
        pattern = re.compile(
            r"(\n\s*){2,}(?P<footer>([A-Za-z]:\\|/|\\\\|file:|https?://).+)$",
            re.IGNORECASE | re.DOTALL,
        )
        m = pattern.search(text)
        return text[: m.start()].rstrip() if m else text

    def _merge_line_breaks_for_continuity(self, text: str) -> str:
        """
        Unifica saltos de línea intra-párrafo preservando separadores dobles,
        y evita cortes por guion al final de línea.
        """
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        t = re.sub(r"(\w+)-\n(\w+)", r"\1\2", t)

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

    def _normalize_for_metadata(self, s: str) -> str:
        """
        Normaliza texto para extracción de metadatos:
        NFKC, supresión de zero-width, espacios no estándar, guiones y saltos.
        """
        if not s:
            return s
        s = unicodedata.normalize("NFKC", s)
        s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
        s = s.replace("\u00A0", " ").replace("\u2007", " ").replace("\u202F", " ")
        s = s.replace("–", "-").replace("—", "-").replace("−", "-").replace("：", ":")
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"[ \t]+\n", "\n", s)
        s = re.sub(r"\n[ \t]+", "\n", s)
        s = re.sub(r"[ \t]+:[ \t]+", ": ", s)
        return s

    def _first_acuerda_item(self, text_norm: str) -> Optional[str]:
        """Obtiene el primer ítem tras 'EL TRIBUNAL ACUERDA' cuando hay lista."""
        m = re.search(r"EL\s+TRIBUNAL\s+ACUERDA\s*:?", text_norm, re.IGNORECASE)
        if not m:
            return None
        after = text_norm[m.end():]
        lines = after.split("\n")
        list_re = re.compile(r"^\s*(?:[-•*]\s+|\d+\)|\d+\.\s+|[IVXLCM]+\)\s+)", re.IGNORECASE)
        for line in lines:
            if not line.strip():
                continue
            if list_re.match(line):
                return line.strip()
        return None

    def _classify_status_with_llm(self, first_item: str) -> Optional[str]:
        """
        Clasifica el primer ítem del bloque ACUERDA en:
        Mantiene | Observa | No formula | Levanta.
        """
        if not self.llm or not first_item:
            return None

        prompt = f"""
Eres un clasificador estricto. Lee el siguiente enunciado y devuelve SOLO una palabra EXACTA:
Mantiene | Observa | No formula | Levanta.

Enunciado:
{first_item}

Respuesta:
""".strip()

        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            label = (resp.content or "").strip().lower().strip('"').strip("'")
            label = re.sub(r"\s+", " ", label)
            mapping = {
                "mantiene": "Mantiene",
                "observa": "Observa",
                "no formula": "No formula",
                "noformula": "No formula",
                "levanta": "Levanta",
            }
            return mapping.get(label, None)
        except Exception:
            return None

    def _extract_document_metadata(self, text: str) -> Dict[str, str]:
        """Extrae metadatos clave del encabezado y deduce materia/status con reglas y LLM opcional."""
        text_norm = self._normalize_for_metadata(text)
        head = text_norm[:3000]

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
            d, m, y = m_ent_fecha.group(2).split("/")
            try:
                fecha_entrada = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
            except Exception:
                fecha_entrada = m_ent_fecha.group(2)
        else:
            m_ent = re.search(
                r"\b(Entrada|N[úu]mero\s+de\s+entrada)\b\s*(?:N[°ºoO])?\s*[:\-]*\s*([\w./-]+)",
                head,
                re.IGNORECASE,
            )
            if m_ent:
                entrada = m_ent.group(2).strip()
            fecha_entrada = fecha_entrada or "N/A"

        materia = "Sin información"
        status = "Sin información"

        line_materia = re.search(r"(?im)^\s*Materia\s*[:\-]\s*(.+?)\s*$", text_norm)
        if line_materia:
            val = line_materia.group(1).strip()
            materia = val if val else "Sin información"

        line_status = re.search(r"(?im)^\s*Status\s*[:\-]\s*(.+?)\s*$", text_norm)
        if line_status:
            val = line_status.group(1).strip()
            status = val if val else "Sin información"

        if materia == "Sin información" or status == "Sin información":
            _DELIMS = r"(?:\/\/|;|\||,|\n{2,}|$|\bStatus\b|\bMateria\b)"
            materia_pat = re.compile(
                rf"\bMateria\s*[:\-]?\s*(.+?)(?=\s*{_DELIMS})",
                flags=re.IGNORECASE | re.DOTALL,
            )

            def _clean_value(v: Optional[str]) -> str:
                if v is None:
                    return "Sin información"
                v = re.sub(r"\s*(?:\/\/|;|\||,)\s*$", "", v.strip())
                return v if re.search(r"[A-Za-zÁÉÍÓÚáéíóúÑñ0-9]", v) else "Sin información"

            if materia == "Sin información":
                mm = materia_pat.search(text_norm)
                if mm:
                    materia = _clean_value(mm.group(1))

            if status == "Sin información":
                first_item = self._first_acuerda_item(text_norm)
                inferred = self._classify_status_with_llm(first_item)
                if inferred:
                    status = inferred

        return {
            "date": date_normalized,
            "expediente": expediente,
            "entrada": entrada,
            "fecha_entrada": fecha_entrada,
            "materia": materia,
            "status": status,
        }

    def _process_atento_acuerda_block(self, section_text: str) -> str:
        """Estandariza el bloque 'EL TRIBUNAL ACUERDA' y conserva el listado inmediatamente posterior."""
        m = re.search(r"EL\s+TRIBUNAL\s+ACUERDA\s*:?", section_text, re.IGNORECASE)
        if not m:
            return section_text

        start, end = m.start(), m.end()
        before = section_text[:start].rstrip()
        after = section_text[end:].lstrip("\n ")

        lines = after.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        list_lines: List[str] = []
        list_pattern = re.compile(r"^\s*(?:[-•*]\s+|\d+\)|\d+\.\s+|[IVXLCM]+\)\s+)", re.IGNORECASE)

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

        header_norm = "EL TRIBUNAL ACUERDA:"
        body = "\n".join(list_lines).strip()
        acordado = header_norm + "\n\n" + body if body else header_norm
        return (before + "\n\n" + acordado).strip()

    # Detección de secciones
    def find_sections(self, text: str) -> List[Dict[str, str]]:
        """Identifica y devuelve secciones principales del documento ya limpiado."""
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

    # Construcción de Documents
    def create_documents_from_text(self, text: str, file_name: str, doc_meta: Dict[str, str]) -> List[Document]:
        """
        Crea documentos por sección con metadatos consolidados. Asigna chunk_index
        secuencial a lo largo de TODO el archivo (no se reinicia por sección).
        """
        sections = self.find_sections(text)
        documents: List[Document] = []
        running_idx = 0

        for section in sections:
            section_text = section["text"]
            if section["name"].upper() == "ATENTO":
                section_text = self._process_atento_acuerda_block(section_text)

            if self.section_subchunking == "recursive":
                sub_docs = self.splitter.create_documents([section_text])
                for d in sub_docs:
                    md = {
                        "section": section["name"],
                        "source_file": file_name,
                        "chunk_index": running_idx,
                        "word_count": len(d.page_content.split()),
                        "char_count": len(d.page_content),
                        "date": doc_meta.get("date", ""),
                        "expediente": doc_meta.get("expediente", ""),
                        "entrada": doc_meta.get("entrada", ""),
                        "fecha_entrada": doc_meta.get("fecha_entrada", ""),
                        "materia": doc_meta.get("materia", "Sin información"),
                        "status": doc_meta.get("status", "Sin información"),
                    }
                    documents.append(Document(page_content=d.page_content, metadata=md))
                    running_idx += 1
            else:
                md = {
                    "section": section["name"],
                    "source_file": file_name,
                    "chunk_index": running_idx,
                    "word_count": len(section_text.split()),
                    "char_count": len(section_text),
                    "date": doc_meta.get("date", ""),
                    "expediente": doc_meta.get("expediente", ""),
                    "entrada": doc_meta.get("entrada", ""),
                    "fecha_entrada": doc_meta.get("fecha_entrada", ""),
                    "materia": doc_meta.get("materia", "Sin información"),
                    "status": doc_meta.get("status", "Sin información"),
                }
                documents.append(Document(page_content=section_text, metadata=md))
                running_idx += 1

        return documents


    # Persistencia
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
                f.write(f"Materia: {doc.metadata.get('materia','Sin información')}\n")
                f.write(f"Status: {doc.metadata.get('status','Sin información')}\n")
                f.write("\n=== CONTENIDO ===\n")
                f.write(doc.page_content)


    # Pipeline principal
    def process_txt_folder(
        self,
        txt_folder: str = "extracted_text",
        output_folder: str = "chunks",
        persist: bool = True,
    ) -> List[Document]:
        """Ejecuta el pipeline completo sobre una carpeta de .txt y devuelve la lista de documentos."""
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


def generate_langchain_chunks_from_txt(
    txt_folder: str = "extracted_text",
    output_folder: str = "chunks",
    section_subchunking: str = "none",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    persist: bool = True,
) -> List[Document]:
    """Construye y ejecuta el pipeline de chunking en LangChain para una carpeta de textos."""
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


if __name__ == "__main__":
    try:
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
