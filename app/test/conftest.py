import os
import sys
from pathlib import Path

# Obtener el directorio raíz del proyecto (donde está la carpeta core)
ROOT_DIR = Path(__file__).parent.parent

# Agregar el directorio raíz al PYTHONPATH
sys.path.insert(0, str(ROOT_DIR))