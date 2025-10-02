"""Sistema de procesamiento de documentos PDF con extracción de metadatos legales."""

import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

# ----------------------------
# Utilidades de normalización
# ----------------------------

MONTHS_ES = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "setiembre": 9,
    "septiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}


def normalize_date_es(text: str) -> str | None:
    """
    Normaliza fechas del tipo: 'Montevideo, 28 de setiembre de 2022.' → '2022-09-28'
    Retorna None si no logra parsear.
    """
    if not text:
        return None
    t = text.strip().lower().replace(".", "")
    # Buscar patrón: [día] de [mes] de [año]
    m = re.search(r"(\d{1,2})\s+de\s+([a-záéíóú]+)\s+de\s+(\d{4})", t, re.IGNORECASE)
    if not m:
        return None
    day = int(m.group(1))
    month_name = m.group(2)
    year = int(m.group(3))
    month = MONTHS_ES.get(month_name, None)
    if not month:
        return None
    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except Exception:
        return None


# -----------------------------------------
# Patrones para campos y encabezados legales
# -----------------------------------------

SECTION_HEADERS = [
    "VISTO",
    "RESULTANDO",
    "CONSIDERANDO",
    "ATENTO",
    "EL TRIBUNAL ACUERDA",
    "PROYECTO DE RESOLUCION",
    "PROYECTO DE RESOLUCIÓN",
]

SECTION_REGEX = r"(?P<h>VISTO|RESULTANDO|CONSIDERANDO|ATENTO|EL TRIBUNAL ACUERDA|PROYECTO DE RESOLUCI[ÓO]N)\s*:?"

CARPETA_REGEX = r"CARPETA\s*N[º°:]?\s*:?[\s]*([0-9\-\w]+)"
ENTRADA_REGEX = r"ENTRADA\s*N[º°:]?\s*:?[\s]*([\w\/\-]+)"
FOLIO_REGEX = r"FOLIO\s*N[º°:]?\s*:?[\s]*([^\n]+)"
# La fecha suele ir precedida por 'Montevideo, ...'
FECHA_LINE_REGEX = r"Montevideo,\s*([^\n]+)"


@dataclass
class SectionSpan:
    name: str
    start: int
    end: int
    page_from: int | None = None
    page_to: int | None = None


class PDFProcessor:
    """Procesador de documentos PDF usando PyMuPDF con extracción de metadatos."""

    def __init__(self, pdf_folder: str = "pdf_documents") -> None:
        """
        Inicializar el procesador de PDFs.

        Args:
            pdf_folder: Carpeta donde se almacenan los PDFs
        """
        self.pdf_folder = Path(pdf_folder)
        self.pdf_folder.mkdir(exist_ok=True)
        print(f"Carpeta de PDFs: {self.pdf_folder.absolute()}")

    # -------------------------
    # Descubrimiento de archivos
    # -------------------------
    def get_pdf_files(self) -> list[Path]:
        """Obtener lista de archivos PDF en la carpeta."""
        pdf_files = list(self.pdf_folder.glob("*.pdf"))
        print(f"Encontrados {len(pdf_files)} archivos PDF")
        return pdf_files

    # --------------------------------------------
    # Extracción por página: texto, header y footer
    # --------------------------------------------
    def _extract_page_text_header_footer(
        self, page: fitz.Page, header_y: float = 60.0, footer_y_margin: float = 60.0
    ) -> dict[str, Any]:
        """
        Extrae texto total de la página y separa posibles header/footer por posición Y.
        Args:
            page: objeto de página de PyMuPDF
            header_y: umbral de Y (px) para considerar 'header'
            footer_y_margin: margen desde el final de la página para 'footer'
        """
        page_height = page.rect.height
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, ...)
        header_parts, body_parts, footer_parts = [], [], []

        for b in blocks:
            x0, y0, x1, y1, text, *_ = b
            text = text.strip()
            if not text:
                continue
            if y1 <= header_y:
                header_parts.append(text)
            elif y0 >= (page_height - footer_y_margin):
                footer_parts.append(text)
            else:
                body_parts.append(text)

        header_text = "\n".join(header_parts).strip()
        footer_text = "\n".join(footer_parts).strip()
        # Texto completo de la página (puede preferirse get_text("text") si se quiere orden de lectura)
        page_text = page.get_text("text")

        return {
            "header_text": header_text or None,
            "footer_text": footer_text or None,
            "page_text": page_text or "",
        }

    # ------------------------
    # Extracción de "campos duros"
    # ------------------------
    def _extract_fields_from_text(self, full_text: str) -> dict[str, str | None]:
        """
        Extrae campos como CARPETA, ENTRADA, FOLIO y FECHA (normalizada).
        """
        m_carpeta = re.search(CARPETA_REGEX, full_text, re.IGNORECASE)
        m_entrada = re.search(ENTRADA_REGEX, full_text, re.IGNORECASE)
        m_folio = re.search(FOLIO_REGEX, full_text, re.IGNORECASE)
        m_fecha = re.search(FECHA_LINE_REGEX, full_text, re.IGNORECASE)

        fecha_norm = None
        if m_fecha:
            fecha_norm = normalize_date_es(m_fecha.group(1))

        return {
            "carpeta": m_carpeta.group(1).strip() if m_carpeta else None,
            "entrada": m_entrada.group(1).strip() if m_entrada else None,
            "folio": m_folio.group(1).strip() if m_folio else None,
            "fecha": fecha_norm,
            "fecha_raw": m_fecha.group(1).strip() if m_fecha else None,
        }

    # ------------------------
    # Segmentación por secciones legales
    # ------------------------
    def _split_sections(self, full_text: str) -> list[SectionSpan]:
        """
        Divide el texto completo en spans por encabezados tipo:
        VISTO, RESULTANDO, CONSIDERANDO, ATENTO, EL TRIBUNAL ACUERDA, PROYECTO DE RESOLUCIÓN.
        """
        it = list(re.finditer(SECTION_REGEX, full_text, re.IGNORECASE))
        if not it:
            return []

        spans: list[SectionSpan] = []
        for i, m in enumerate(it):
            start = m.start()
            end = it[i + 1].start() if i + 1 < len(it) else len(full_text)
            name = m.group("h").upper()
            # Normalizar acentos en el nombre para consistencia
            name = (
                name.replace("Ó", "O")
                .replace("É", "E")
                .replace("Í", "I")
                .replace("Á", "A")
                .replace("Ú", "U")
            )
            spans.append(SectionSpan(name=name, start=start, end=end))
        return spans

    # ------------------------
    # Enriquecimiento de metadatos
    # ------------------------
    def _build_enriched_metadata(
        self, base_meta: dict[str, Any], sections: list[SectionSpan]
    ) -> dict[str, Any]:
        """
        Crea un diccionario de metadatos enriquecidos para reportar y/o persistir.
        """
        enriched = dict(base_meta)
        enriched["sections_found"] = [s.name for s in sections] if sections else []
        return enriched

    # -------------------------------------
    # Entrada principal: extraer texto + meta
    # -------------------------------------
    def extract_text_from_pdf(self, pdf_path: Path) -> dict[str, Any]:
        """
        Extraer texto y metadatos enriquecidos de un archivo PDF.

        Args:
            pdf_path: Ruta al archivo PDF

        Returns:
            Diccionario con información del PDF, texto, metadatos y secciones
        """
        try:
            doc = fitz.open(pdf_path)

            pdf_info: dict[str, Any] = {
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "total_pages": len(doc),
                "pages": [],
                "full_text": "",
                "metadata_pdf": doc.metadata or {},  # metadatos PDF estándar
                "metadata_enriched": {},  # metadatos legales enriquecidos
                "sections": [],  # spans y textos por sección
            }

            print(f"Procesando: {pdf_path.name} ({len(doc)} páginas)")

            # Extraer por página (texto + header/footer)
            full_text_parts: list[str] = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_data = self._extract_page_text_header_footer(page)

                text = page_data["page_text"]
                page_info = {
                    "page_number": page_num + 1,
                    "text": text,
                    "char_count": len(text),
                    "word_count": len(text.split()) if text else 0,
                    "header_text": page_data["header_text"],
                    "footer_text": page_data["footer_text"],
                }
                pdf_info["pages"].append(page_info)
                full_text_parts.append(text)

                print(
                    f"  Página {page_num + 1}: "
                    f"{page_info['char_count']} caracteres, {page_info['word_count']} palabras"
                )

            # Texto completo
            full_text = "\n\n".join(full_text_parts)
            pdf_info["full_text"] = full_text
            pdf_info["total_chars"] = len(full_text)
            pdf_info["total_words"] = len(full_text.split())

            # Extraer campos duros del texto completo
            fields = self._extract_fields_from_text(full_text)

            # Dividir en secciones legales
            sections = self._split_sections(full_text)

            # Adjuntar slices de texto por sección
            sections_payload = []
            for s in sections:
                sections_payload.append(
                    {
                        "name": s.name,
                        "start": s.start,
                        "end": s.end,
                        "text": full_text[s.start : s.end].strip(),
                    }
                )
            pdf_info["sections"] = sections_payload

            # Construir metadata enriquecida reportable
            base_meta = {
                "doc_id": fields.get("carpeta") or Path(pdf_path).stem,
                "carpeta": fields.get("carpeta"),
                "entrada": fields.get("entrada"),
                "folio": fields.get("folio"),
                "fecha": fields.get("fecha"),
                "fecha_raw": fields.get("fecha_raw"),
                "source": pdf_path.name,
            }
            pdf_info["metadata_enriched"] = self._build_enriched_metadata(
                base_meta, sections
            )

            doc.close()

            print(
                f"Texto extraído: {pdf_info['total_chars']} caracteres, "
                f"{pdf_info['total_words']} palabras"
            )
            return pdf_info

        except Exception as e:
            print(f"Error procesando {pdf_path.name}: {e}")
            return {
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "error": str(e),
                "success": False,
            }

    # -----------------------
    # Procesamiento en lote
    # -----------------------
    def process_all_pdfs(self) -> list[dict[str, Any]]:
        """
        Procesar todos los PDFs en la carpeta y mostrar metadatos extraídos.
        """
        pdf_files = self.get_pdf_files()
        if not pdf_files:
            print("No se encontraron archivos PDF en la carpeta")
            return []

        results: list[dict[str, Any]] = []

        print(f"\nIniciando procesamiento de {len(pdf_files)} archivos PDF")
        print("=" * 80)

        for pdf_file in pdf_files:
            print(f"\nProcesando: {pdf_file.name}")
            result = self.extract_text_from_pdf(pdf_file)
            results.append(result)

            if "error" in result:
                print(f"  Error: {result['error']}")
                continue

            # Mostrar metadatos enriquecidos por cada PDF
            meta = result.get("metadata_enriched", {})
            print("  Metadatos extraídos:")
            print(f"    doc_id   : {meta.get('doc_id')}")
            print(f"    carpeta  : {meta.get('carpeta')}")
            print(f"    entrada  : {meta.get('entrada')}")
            print(f"    folio    : {meta.get('folio')}")
            print(f"    fecha    : {meta.get('fecha')} (raw: {meta.get('fecha_raw')})")
            print(f"    source   : {meta.get('source')}")
            print(
                f"    secciones: {', '.join(meta.get('sections_found', [])) or 'N/D'}"
            )

        return results

    # -----------------------
    # Guardado del texto plano
    # -----------------------
    def save_extracted_text(
        self, pdf_info: dict[str, Any], output_folder: str = "extracted_text"
    ) -> None:
        """
        Guardar el texto extraído en archivos de texto.
        """
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)

        if "error" in pdf_info:
            print(f"Saltando {pdf_info['file_name']} debido a error")
            return

        text_filename = Path(pdf_info["file_name"]).stem + ".txt"
        text_file_path = output_path / text_filename

        try:
            with open(text_file_path, "w", encoding="utf-8") as f:
                f.write(f"Archivo: {pdf_info['file_name']}\n")
                f.write(f"Páginas: {pdf_info['total_pages']}\n")
                f.write(f"Caracteres: {pdf_info['total_chars']}\n")
                f.write(f"Palabras: {pdf_info['total_words']}\n")
                f.write("=" * 60 + "\n")
                f.write("Metadatos PDF (estándar):\n")
                for k, v in (pdf_info.get("metadata_pdf") or {}).items():
                    f.write(f"  - {k}: {v}\n")
                f.write("Metadatos enriquecidos:\n")
                for k, v in (pdf_info.get("metadata_enriched") or {}).items():
                    f.write(f"  - {k}: {v}\n")
                f.write("=" * 60 + "\n\n")
                f.write(pdf_info["full_text"])

            print(f"Texto guardado en: {text_file_path}")

        except Exception as e:
            print(f"Error guardando texto de {pdf_info['file_name']}: {e}")

    # -----------------------
    # Resumen
    # -----------------------
    def print_summary(self, results: list[dict[str, Any]]) -> None:
        """Mostrar resumen del procesamiento."""
        print("\n" + "=" * 80)
        print("RESUMEN DEL PROCESAMIENTO")
        print("=" * 80)

        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]

        print(f"Archivos procesados exitosamente: {len(successful)}")
        print(f"Archivos con errores: {len(failed)}")

        if successful:
            total_pages = sum(r["total_pages"] for r in successful)
            total_chars = sum(r["total_chars"] for r in successful)
            total_words = sum(r["total_words"] for r in successful)

            print(f"Total de páginas procesadas: {total_pages}")
            print(f"Total de caracteres extraídos: {total_chars:,}")
            print(f"Total de palabras extraídas: {total_words:,}")

        if failed:
            print("\nArchivos con errores:")
            for result in failed:
                print(f"  - {result['file_name']}: {result['error']}")


def main() -> int:
    """
    Función principal para procesamiento de PDFs.
    """
    print("Sistema de Procesamiento de PDFs")
    print("=" * 80)

    # Puedes cambiar la carpeta aquí o pasarla por CLI más adelante
    processor = PDFProcessor(pdf_folder="pdf_documents")

    pdf_files = processor.get_pdf_files()
    if not pdf_files:
        print("\nPara usar este sistema:")
        print(
            f"  1) Coloca archivos PDF en la carpeta: {processor.pdf_folder.absolute()}"
        )
        print("  2) Ejecuta nuevamente el programa")
        print("\nLa carpeta se ha creado automáticamente si no existía.")
        return 0

    results = processor.process_all_pdfs()

    print("\nGuardando texto extraído...")
    for result in results:
        if "error" not in result:
            processor.save_extracted_text(result)

    processor.print_summary(results)

    print("\nProcesamiento completado.")
    print("Texto extraído guardado en: ./extracted_text/")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error fatal: {e}")
        sys.exit(1)
