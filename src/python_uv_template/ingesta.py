"""Módulo para extraer texto limpio de documentos PDF y guardarlos como .txt"""

import fitz
import re
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

MONTHS_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "setiembre": 9, "septiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}

def normalize_date_es(text: str):
    m = re.search(r"(\d{1,2})\s+de\s+([a-záéíóú]+)\s+de\s+(\d{4})", text.lower())
    if not m:
        return None
    day, month_name, year = int(m.group(1)), m.group(2), int(m.group(3))
    month = MONTHS_ES.get(month_name)
    return datetime(year, month, day).strftime("%Y-%m-%d") if month else None


class PDFTextExtractor:
    def __init__(self, pdf_folder="pdf_documents", output_folder="extracted_text"):
        self.pdf_folder = Path(pdf_folder)
        self.output_folder = Path(output_folder)
        self.pdf_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)

    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extrae texto limpio (sin headers ni footers) de un PDF."""
        doc = fitz.open(pdf_path)
        print(f"Procesando {pdf_path.name} ({len(doc)} páginas)...")

        full_text_parts = []
        for page in doc:
            height = page.rect.height
            blocks = page.get_text("blocks")
            for x0, y0, x1, y1, text, *_ in blocks:
                if not text.strip():
                    continue
                if y1 <= 60 or y0 >= (height - 60):
                    continue
                full_text_parts.append(text.strip())

        full_text = "\n".join(full_text_parts).strip()

        full_text = re.sub(
            r"(?i)(Materia\s*:\s*[^\n/]+)\s*//\s*(Status\s*:\s*[^\n]+)",
            lambda m: f"{m.group(1).strip()}\n{m.group(2).strip()}",
            full_text,
        )
        
        print(f"  → Extraído {len(full_text)} caracteres.")

        return {"file": pdf_path.name, "text": full_text, "chars": len(full_text)}

    def save_text(self, file_name: str, text: str):
        """Guarda el texto en un archivo .txt."""
        txt_path = self.output_folder / f"{Path(file_name).stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  ✅ Texto guardado en {txt_path}")

    def process_all_pdfs(self) -> List[Dict[str, Any]]:
        """Procesa todos los PDFs del directorio."""
        pdf_files = list(self.pdf_folder.glob("*.pdf"))
        if not pdf_files:
            print("⚠️ No se encontraron PDFs en la carpeta.")
            return []

        results = []
        for pdf in pdf_files:
            data = self.extract_text_from_pdf(pdf)
            self.save_text(data["file"], data["text"])
            results.append(data)
        return results


def process_pdfs(pdf_folder="pdf_documents", output_folder="extracted_text"):
    extractor = PDFTextExtractor(pdf_folder, output_folder)
    return extractor.process_all_pdfs()
