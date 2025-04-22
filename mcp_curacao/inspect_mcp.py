#!/usr/bin/env python3

import mcp

# Mostrar información sobre el módulo client
print("Información sobre módulo mcp.client:")
import mcp.client
print(dir(mcp.client))

# Probar importaciones específicas
print("\nProbando importaciones específicas:")
try:
    from mcp.client import ClientSession
    print("ClientSession importado correctamente")
    print(dir(ClientSession))
except ImportError as e:
    print(f"Error importando ClientSession: {e}")

try:
    from mcp.stdio_client import StdioClient
    print("StdioClient importado correctamente")
    print(dir(StdioClient))
except ImportError as e:
    print(f"Error importando StdioClient: {e}")