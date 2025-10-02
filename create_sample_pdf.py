"""Script para crear un PDF de ejemplo para probar el sistema."""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pathlib import Path

def create_sample_pdf():
    """Crear un PDF de ejemplo con contenido sobre Uruguay."""
    
    # Crear carpeta si no existe
    pdf_folder = Path("pdf_documents")
    pdf_folder.mkdir(exist_ok=True)
    
    # Crear el PDF
    pdf_path = pdf_folder / "uruguay_ejemplo.pdf"
    
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    
    # Página 1
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 100, "Uruguay - Información General")
    
    c.setFont("Helvetica", 12)
    y = height - 150
    
    content_page1 = [
        "Uruguay es un país ubicado en América del Sur, entre Argentina y Brasil.",
        "Su capital es Montevideo, que es también la ciudad más poblada del país.",
        "",
        "Datos básicos:",
        "• Superficie: 176.215 km²",
        "• Población: aproximadamente 3.5 millones de habitantes",
        "• Idioma oficial: Español",
        "• Moneda: Peso uruguayo",
        "",
        "Historia:",
        "Uruguay obtuvo su independencia en 1825. El país ha tenido una historia",
        "democrática estable y es conocido por sus políticas progresistas.",
    ]
    
    for line in content_page1:
        c.drawString(100, y, line)
        y -= 20
    
    # Nueva página
    c.showPage()
    
    # Página 2
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 100, "Cultura y Economía")
    
    c.setFont("Helvetica", 12)
    y = height - 150
    
    content_page2 = [
        "Cultura:",
        "Uruguay es famoso por el tango, el fútbol y su rica tradición literaria.",
        "El mate es la bebida nacional y forma parte importante de la cultura social.",
        "",
        "Economía:",
        "La economía uruguaya se basa principalmente en:",
        "• Agricultura y ganadería",
        "• Servicios financieros",
        "• Turismo",
        "• Tecnología",
        "",
        "El país es conocido por su producción de carne y productos lácteos",
        "de alta calidad, y ha emergido como un hub tecnológico en la región.",
    ]
    
    for line in content_page2:
        c.drawString(100, y, line)
        y -= 20
    
    c.save()
    print(f"✅ PDF de ejemplo creado: {pdf_path}")

if __name__ == "__main__":
    try:
        create_sample_pdf()
    except ImportError:
        print("⚠️  Para crear el PDF de ejemplo, instala reportlab:")
        print("   uv add reportlab")
    except Exception as e:
        print(f"❌ Error creando PDF: {e}")
