#!/usr/bin/env python3
"""
Cliente básico para los servidores MCP de Curaçao
Este cliente usa subprocess para interactuar con los servidores MCP
"""

import os
import sys
import subprocess
import json
import shutil
import time

# Obtener ancho del terminal
term_width, _ = shutil.get_terminal_size()

# Colores ANSI
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
BLUE = "\033[34m"
CYAN = "\033[36m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"

def print_header():
    """Imprime el encabezado del cliente"""
    title = "Asistente Turístico de Curaçao - Cliente Básico"
    print("\n" + "=" * term_width)
    print(f"{BOLD}{GREEN}{title.center(term_width)}{RESET}")
    print("=" * term_width)
    
    print(f"\n{CYAN}¡Bienvenido al asistente turístico de Curaçao!{RESET}")
    print("Este cliente básico interactúa con los servidores MCP para Curaçao.")
    print("Puedo ayudarte a planificar tu viaje con recomendaciones personalizadas.")
    
    print("\nEscribe tu consulta y presiona Enter. Escribe '/salir' para terminar.")
    print("=" * term_width + "\n")

def run_mcp_call(server, tool, **kwargs):
    """
    Ejecuta una llamada a un servidor MCP usando subprocess
    
    Args:
        server: ID del servidor MCP
        tool: Nombre de la herramienta a llamar
        kwargs: Parámetros para la herramienta
        
    Returns:
        Resultado de la llamada o None si falla
    """
    try:
        # Construir comando
        cmd = ["mcp", "call", server, tool]
        
        # Agregar parámetros
        for key, value in kwargs.items():
            cmd.extend(["--" + key, json.dumps(value)])
        
        # Ejecutar comando
        print(f"{DIM}Ejecutando: {' '.join(cmd)}{RESET}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Verificar resultado
        if result.returncode != 0:
            print(f"{RED}Error al llamar al servidor MCP: {result.stderr}{RESET}")
            return None
        
        # Parsear resultado
        return json.loads(result.stdout)
    except Exception as e:
        print(f"{RED}Error al llamar al servidor MCP: {str(e)}{RESET}")
        return None

def main():
    """Función principal del cliente básico"""
    print_header()
    
    # Crear sesión
    session_result = run_mcp_call("curacao-orchestrator-server", "create_user_session")
    if not session_result:
        print(f"{RED}Error al crear sesión.{RESET}")
        return
    
    session_id = session_result.get("session_id")
    print(f"{DIM}Sesión creada: {session_id}{RESET}")
    
    # Loop principal
    while True:
        try:
            # Solicitar entrada
            user_input = input(f"\n{BOLD}{GREEN}Tú:{RESET} ")
            
            # Verificar si es comando de salida
            if user_input.lower() in ['/salir', '/exit', '/quit']:
                print(f"\n{GREEN}¡Gracias por usar el asistente turístico de Curaçao! ¡Hasta pronto!{RESET}\n")
                break
            
            # Verificar si está vacío
            if not user_input.strip():
                continue
            
            # 1. Extraer preferencias del mensaje
            print(f"{DIM}Procesando mensaje...{RESET}")
            preferences = run_mcp_call("curacao-orchestrator-server", "extract_preferences_from_message", message=user_input)
            
            # 2. Actualizar preferencias
            if preferences:
                run_mcp_call("curacao-orchestrator-server", "update_user_preferences", 
                           session_id=session_id, preferences=preferences)
            
            # 3. Obtener contexto actualizado
            user_context = run_mcp_call("curacao-orchestrator-server", "get_user_context", 
                                     session_id=session_id)
            
            # 4. Determinar información faltante
            missing_info = run_mcp_call("curacao-orchestrator-server", "determine_missing_information", 
                                      session_id=session_id)
            
            # 5. Generar recomendaciones si no falta información
            recommendations = []
            if not missing_info:
                prefs = user_context.get("preferences", {})
                
                # Buscar recomendaciones en Excel
                print(f"{DIM}Buscando recomendaciones...{RESET}")
                excel_results = run_mcp_call("curacao-excel-server", "get_recommendations",
                                           interests=prefs.get("interests", []),
                                           budget=prefs.get("budget"),
                                           duration=prefs.get("duration"),
                                           locations=prefs.get("locations", ["Curaçao"]),
                                           limit=5)
                
                # Si hay recomendaciones, buscar información en RAG
                rag_results = {}
                if excel_results:
                    places = [rec.get("name", "") for rec in excel_results[:2]]
                    if places:
                        query = f"Información sobre {', '.join(places)} en Curaçao"
                        print(f"{DIM}Consultando RAG: {query}{RESET}")
                        rag_results = run_mcp_call("curacao-rag-server", "search_knowledge",
                                                query=query, top_k=3)
                        
                        # Enriquecer recomendaciones
                        recommendations = run_mcp_call("curacao-orchestrator-server", "enrich_recommendations",
                                                   recommendations=excel_results, rag_results=rag_results)
                    else:
                        recommendations = excel_results
            
            # 6. Registrar mensaje en historial
            run_mcp_call("curacao-orchestrator-server", "add_message_to_history",
                       session_id=session_id, sender="user", message=user_input)
            
            # 7. Generar respuesta
            print(f"{DIM}Generando respuesta...{RESET}")
            response_data = run_mcp_call("curacao-response-server", "generate_response",
                                      query=user_input,
                                      user_context=user_context,
                                      recommendations=recommendations,
                                      rag_info=rag_results)
            
            # 8. Registrar respuesta en historial
            if response_data:
                run_mcp_call("curacao-orchestrator-server", "add_message_to_history",
                           session_id=session_id, sender="assistant", 
                           message=response_data.get("response", ""))
            
            # 9. Mostrar respuesta
            if response_data and "response" in response_data:
                response_text = response_data["response"]
                print(f"\n{BOLD}{BLUE}Asistente:{RESET}")
                for line in response_text.split("\n"):
                    print(f"    {line}")
            else:
                print(f"\n{RED}Error: No se pudo generar una respuesta.{RESET}")
            
        except KeyboardInterrupt:
            print(f"\n\n{GREEN}¡Hasta pronto!{RESET}\n")
            break
        except Exception as e:
            print(f"\n{RED}Error inesperado: {str(e)}{RESET}\n")
            continue  # Continuar con la siguiente iteración

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{GREEN}¡Hasta pronto!{RESET}\n")
    except Exception as e:
        print(f"\n{RED}Error inesperado: {str(e)}{RESET}\n")