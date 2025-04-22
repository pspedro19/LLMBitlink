#!/usr/bin/env python3
"""
Script simple para probar la lectura de documentos PDF en app2.
"""
import os
import sys
from pathlib import Path

# Añadir el directorio raíz al PATH para poder importar los módulos de app2
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))

try:
    from app2.ingestion.document_reader import DocumentReader
    from app2.core.config.config import Config
    print("Módulos importados correctamente")
except ImportError as e:
    print(f"Error al importar módulos: {e}")
    sys.exit(1)

# Inicializar el lector de documentos
reader = DocumentReader()
config = Config()

# Verificar directorios
print(f"Directorio base: {base_dir}")
print(f"Directorio KB_DIR: {config.KB_DIR}")
pdf_dir = config.KB_DIR / "pdf" / "curaçao_information"
print(f"Directorio de PDFs: {pdf_dir}")
print(f"¿Existe el directorio? {pdf_dir.exists()}")

# Si el directorio existe, listar subdirectorios
if pdf_dir.exists():
    print("\nSubdirectorios encontrados:")
    for subdir in pdf_dir.glob("*"):
        if subdir.is_dir():
            print(f"\n- {subdir.name}")
            # Listar PDFs en este subdirectorio
            pdfs = list(subdir.glob("*.pdf"))
            print(f"  PDFs encontrados: {len(pdfs)}")
            
            if pdfs:
                # Probar con el primer PDF
                pdf_path = pdfs[0]
                print(f"\n  Probando: {pdf_path.name}")
                try:
                    result = reader.read_document(str(pdf_path))
                    print(f"  ✓ Lectura exitosa! Formato: {result['format']}")
                    print(f"  ✓ Páginas: {result.get('pages', 'N/A')}")
                    
                    content = result['content']
                    content_preview = content[:150].replace('\n', ' ')
                    print(f"  ✓ Contenido (primeros 150 caracteres): {content_preview}...")
                    print(f"  ✓ Longitud del contenido: {len(content)} caracteres")
                    
                    if 'metadata' in result:
                        print(f"  ✓ Metadatos: {result['metadata']}")
                except Exception as e:
                    print(f"  ✗ Error al leer PDF: {e}")
                    
                    # Intentar verificar si el archivo es accesible
                    print(f"  Verificando acceso al archivo...")
                    try:
                        size = os.path.getsize(pdf_path)
                        print(f"  ✓ Tamaño del archivo: {size} bytes")
                        
                        # Intentar abrir el archivo directamente
                        with open(pdf_path, 'rb') as f:
                            header = f.read(5)
                            print(f"  ✓ Primeros 5 bytes: {header}")
                    except Exception as access_error:
                        print(f"  ✗ Error al acceder al archivo: {access_error}")
else:
    print("\nEl directorio de PDFs no existe. Verificando estructura de directorios:")
    print(f"- app2 existe: {(base_dir / 'app2').exists()}")
    print(f"- KB_DIR existe: {config.KB_DIR.exists()}")
    print(f"- documents existe: {(config.KB_DIR).exists()}")
    print(f"- pdf existe: {(config.KB_DIR / 'pdf').exists()}")
    
    # Intentar crear la estructura de directorios si no existe
    try:
        print("\nCreando estructura de directorios:")
        os.makedirs(config.KB_DIR / "pdf", exist_ok=True)
        os.makedirs(config.KB_DIR / "txt", exist_ok=True)
        print("✓ Directorios creados correctamente")
        
        # Crear un archivo de texto de prueba
        test_file = config.KB_DIR / "txt" / "test_document.txt"
        content = "Este es un documento de prueba para app2.\nCreado por test_pdf_simple.py"
        with open(test_file, 'w') as f:
            f.write(content)
        print(f"✓ Archivo de prueba creado: {test_file}")
        
        # Leer el archivo de texto
        print("\nLeyendo archivo de texto de prueba:")
        try:
            result = reader.read_document(str(test_file))
            print(f"✓ Lectura exitosa! Formato: {result['format']}")
            print(f"✓ Contenido: {result['content']}")
        except Exception as e:
            print(f"✗ Error al leer archivo de texto: {e}")
    except Exception as e:
        print(f"✗ Error al crear directorios: {e}")

# Intentar verificar PyPDF2
try:
    import PyPDF2
    print(f"\nPyPDF2 instalado: versión {PyPDF2.__version__}")
except ImportError:
    print("\nPyPDF2 no está instalado. Intenta ejecutar: pip install PyPDF2")

print("\nTest completado.")