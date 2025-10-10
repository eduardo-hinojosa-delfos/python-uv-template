"""
pdf_extractor.py
Extracción estructurada de texto desde documentos PDF y almacenamiento en formato .txt.
Incluye normalización de fechas en español y filtrado básico de encabezados y pies de página.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz

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


def normalize_date_es(text: str) -> Optional[str]:
    """Convierte fechas en español con el formato 'd de mes de yyyy' al formato ISO (YYYY-MM-DD)."""
    m = re.search(r"(\d{1,2})\s+de\s+([a-záéíóú]+)\s+de\s+(\d{4})", text.lower())
    if not m:
        return None
    day, month_name, year = int(m.group(1)), m.group(2), int(m.group(3))
    month = MONTHS_ES.get(month_name)
    return datetime(year, month, day).strftime("%Y-%m-%d") if month else None


class PDFTextExtractor:
    """
    Clase para la extracción de texto de documentos PDF, excluyendo encabezados y pies de página.
    Los textos extraídos se almacenan en archivos .txt con el mismo nombre base.
    """

    def __init__(self, pdf_folder: str = "pdf_documents", output_folder: str = "extracted_text") -> None:
        self.pdf_folder = Path(pdf_folder)
        self.output_folder = Path(output_folder)
        self.pdf_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)

    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extrae texto del PDF eliminando márgenes superiores e inferiores."""
        doc = fitz.open(pdf_path)
        print(f"Procesando {pdf_path.name} ({len(doc)} páginas)...")

        text_segments: List[str] = []
        for page in doc:
            height = page.rect.height
            for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
                if not text or not text.strip():
                    continue
                if y1 <= 60 or y0 >= (height - 60):
                    continue
                text_segments.append(text.strip())

        combined_text = "\n".join(text_segments).strip()

        combined_text = re.sub(
            r"(?i)(Materia\s*:\s*[^\n/]+)\s*//\s*(Status\s*:\s*[^\n]+)",
            lambda m: f"{m.group(1).strip()}\n{m.group(2).strip()}",
            combined_text,
        )

        print(f"  → Texto extraído ({len(combined_text)} caracteres).")
        return {"file": pdf_path.name, "text": combined_text, "chars": len(combined_text)}

    def save_text(self, file_name: str, text: str) -> None:
        """Guarda el texto extraído en formato .txt con codificación UTF-8."""
        txt_path = self.output_folder / f"{Path(file_name).stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  Archivo guardado en {txt_path}")

    def process_all_pdfs(self) -> List[Dict[str, Any]]:
        """Procesa todos los archivos PDF disponibles en el directorio y devuelve el resumen de extracción."""
        pdf_files = list(self.pdf_folder.glob("*.pdf"))
        if not pdf_files:
            print("No se encontraron archivos PDF en el directorio especificado.")
            return []

        results: List[Dict[str, Any]] = []
        for pdf in pdf_files:
            data = self.extract_text_from_pdf(pdf)
            self.save_text(data["file"], data["text"])
            results.append(data)
        return results


def process_pdfs(pdf_folder: str = "pdf_documents", output_folder: str = "extracted_text") -> List[Dict[str, Any]]:
    """Función auxiliar para ejecutar la extracción completa de texto desde un conjunto de PDFs."""
    extractor = PDFTextExtractor(pdf_folder, output_folder)
    return extractor.process_all_pdfs()
