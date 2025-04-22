#!/usr/bin/env python3
import os
import sys
import asyncio

# Configurar el path para resolver problemas de importación
os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app3'))

# Importar el módulo main de app4
from app4.main import main

if __name__ == "__main__":
    # Añadir argumento CLI si no está presente
    if len(sys.argv) == 1 or "--cli" not in sys.argv:
        sys.argv.append("--cli")
    
    # Ejecutar la función principal
    asyncio.run(main())
