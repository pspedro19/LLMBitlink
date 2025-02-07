import os
import sys

# Obtener el directorio raíz del proyecto (un nivel arriba del directorio test)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Agregar el directorio raíz al PYTHONPATH
sys.path.insert(0, project_root)