# mcp_curacao/launch.py
import subprocess
import time
import os
import signal
import sys
import argparse

def start_server(name, script_path):
    """Inicia un servidor MCP en un proceso separado"""
    print(f"Iniciando servidor MCP: {name}...")
    process = subprocess.Popen(
        ["python", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    # Esperar un momento para que el servidor se inicialice
    time.sleep(1)
    if process.poll() is not None:
        print(f"Error al iniciar {name}. Código de salida: {process.poll()}")
        stdout, stderr = process.communicate()
        print(f"Salida: {stdout}")
        print(f"Error: {stderr}")
        return None
    return process

def main():
    """Inicia todos los servidores y el cliente"""
    parser = argparse.ArgumentParser(description='Lanzador de Asistente MCP-RAG para Curaçao')
    
    parser.add_argument('--no-client', action='store_true',
                      help='Solo inicia los servidores, no el cliente')
    parser.add_argument('--client', choices=['simulated', 'improved', 'basic'],
                      default='simulated',
                      help='Tipo de cliente a usar (default: simulated)')
    parser.add_argument('--dev', action='store_true',
                      help='Modo desarrollo (muestra más logs)')
    
    args = parser.parse_args()
    
    # Obtener directorio base
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Rutas a los scripts de servidores
    server_scripts = [
        ("Servidor Excel", os.path.join(base_dir, "servers/excel_server.py")),
        ("Servidor RAG", os.path.join(base_dir, "servers/rag_server.py")),
        ("Servidor Orquestador", os.path.join(base_dir, "servers/orchestrator_server.py")),
        ("Servidor de Respuestas", os.path.join(base_dir, "servers/response_server.py"))
    ]
    
    # Lista para controlar los procesos
    processes = []
    
    try:
        # Iniciar servidores (solo si no usamos el cliente mejorado que los inicia)
        if args.client != 'improved':
            for name, script in server_scripts:
                process = start_server(name, script)
                if process:
                    processes.append((name, process))
                else:
                    # Si falla un servidor, terminar todos los procesos lanzados
                    print("Error al iniciar servidores. Abortando...")
                    for n, p in processes:
                        p.terminate()
                    return 1
            
            print("\nTodos los servidores MCP iniciados correctamente.")
        
        # Iniciar cliente si se solicita
        if not args.no_client:
            if args.client == 'simulated':
                print("\nIniciando cliente simulado...")
                client_path = os.path.join(base_dir, "simulate_client.py")
                client_process = subprocess.Popen([
                    "python", client_path
                ])
            elif args.client == 'basic':
                print("\nIniciando cliente básico...")
                client_path = os.path.join(base_dir, "basic_client.py")
                client_process = subprocess.Popen([
                    "python", client_path
                ])
            elif args.client == 'improved':
                print("\nIniciando cliente mejorado...")
                client_path = os.path.join(base_dir, "client/improved_client.py")
                client_process = subprocess.Popen([
                    "python", client_path
                ])
            
            # Esperar a que el cliente termine
            client_process.wait()
        else:
            print("\nServidores iniciados sin cliente. Presiona Ctrl+C para detener.")
            # Mantener los servidores activos
            while True:
                time.sleep(1)
        
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario. Cerrando servidores...")
    finally:
        # Cerrar todos los procesos
        for name, process in processes:
            print(f"Cerrando {name}...")
            process.terminate()
        
        print("Todos los servidores cerrados.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())