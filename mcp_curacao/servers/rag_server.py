# mcp_curacao/servers/rag_server.py
from mcp.server.fastmcp import FastMCP
import sys
import os
import asyncio
import logging
import time
from typing import Dict, Any

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag-server")

# Añadir directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAG_APP_DIR, RAG_SERVER_ID, DB_CONFIG

# Añadir directorio de app2 al path para importar módulos
sys.path.insert(0, RAG_APP_DIR)

# Inicializar servidor MCP
mcp = FastMCP(RAG_SERVER_ID)

# Variables globales
rag_pipeline = None
rag_cache = {}  # Caché simple para consultas recientes

# Timeout para operaciones de base de datos
TIMEOUT_SECONDS = 10

async def query_needs_rag(query: str) -> bool:
    """
    Determina si una consulta necesita usar el RAG o puede ser respondida directamente.
    
    Args:
        query: Consulta del usuario
        
    Returns:
        True si la consulta necesita RAG, False si no
    """
    # Consultas genéricas que no necesitan RAG
    generic_greetings = [
        "hola", "hello", "hi", "buenos días", "buenas tardes",
        "cómo estás", "how are you", "saludos", "qué tal",
        "adiós", "bye", "hasta luego", "gracias", "thank you",
    ]
    
    # Verificar si es un saludo simple
    if query.lower().strip().strip("?!.,") in generic_greetings:
        logger.info(f"Consulta genérica detectada, no se usará RAG: '{query}'")
        return False
    
    # Palabras clave que indican que es probable que necesite RAG
    rag_keywords = [
        "curaçao", "curacao", "playas", "beaches", "atracciones", "attractions",
        "hotel", "restaurante", "restaurant", "actividades", "activities",
        "tourism", "turismo", "isla", "island", "caribe", "caribbean",
        "willemstad", "playa", "beach", "museo", "museum", "snorkel",
        "buceo", "diving", "historia", "history", "precio", "price"
    ]
    
    # Verificar si contiene palabras clave relacionadas con Curaçao
    for keyword in rag_keywords:
        if keyword.lower() in query.lower():
            logger.info(f"Consulta relacionada con el dominio, se usará RAG: '{query}'")
            return True
    
    # Para otros mensajes, usar modelo para decidir
    if len(query.split()) > 5:  # Si tiene más de 5 palabras, podría ser una pregunta compleja
        return True
    
    # Por defecto, no usar RAG
    logger.info(f"Consulta no categorizada, no se usará RAG: '{query}'")
    return False

@mcp.tool()
async def initialize_rag() -> Dict[str, Any]:
    """
    Inicializa el sistema RAG.
    
    Returns:
        Estado de la inicialización
    """
    global rag_pipeline
    
    try:
        # Importar componentes RAG
        from app2.core.config.config import Config
        from app2.core.pipelines.search_pipeline import SearchPipeline
        
        # Inicializar configuración
        config = Config()
        # Usar configuración de base de datos del archivo config.py
        config.DB_HOST = DB_CONFIG["host"]
        config.DB_PORT = DB_CONFIG["port"]
        config.DB_USER = DB_CONFIG["user"]
        config.DB_PASSWORD = DB_CONFIG["password"]
        config.DB_NAME = DB_CONFIG["database"]
        config.TIMEOUT_SECONDS = TIMEOUT_SECONDS
        
        # Inicializar pipeline
        rag_pipeline = SearchPipeline(config=config)
        logger.info("Sistema RAG inicializado correctamente")
        
        # Hacer una consulta de prueba para verificar que todo funciona
        try:
            test_results = await asyncio.wait_for(
                rag_pipeline.search(
                    query_text="Curaçao",
                    mode="hybrid",
                    top_k=1,
                    strategy="relevance"
                ),
                timeout=TIMEOUT_SECONDS
            )
            logger.info(f"Consulta de prueba exitosa: {len(test_results.get('chunks', []))} resultados")
        except Exception as e:
            logger.warning(f"La consulta de prueba no tuvo éxito: {e}")
        
        return {
            "status": "success",
            "message": "RAG pipeline inicializado correctamente"
        }
    except Exception as e:
        logger.error(f"Error al inicializar RAG: {e}")
        return {
            "status": "error",
            "message": f"Error al inicializar RAG: {str(e)}"
        }

@mcp.tool()
async def get_server_health() -> Dict[str, Any]:
    """
    Verifica el estado de salud del servidor RAG.
    
    Returns:
        Estado de salud del servidor
    """
    global rag_pipeline
    
    health = {
        "status": "healthy" if rag_pipeline else "not_initialized",
        "pipeline_initialized": rag_pipeline is not None,
        "cache_size": len(rag_cache),
        "timestamp": time.time()
    }
    
    # Verificar si el pipeline funciona correctamente
    if rag_pipeline:
        try:
            # Intenta una consulta simple para verificar
            await asyncio.wait_for(
                rag_pipeline.search(
                    query_text="test",
                    mode="hybrid",
                    top_k=1,
                    strategy="relevance"
                ),
                timeout=5
            )
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
    
    return health

@mcp.tool()
async def search_knowledge(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Busca información en la base de conocimientos usando RAG con mejoras.
    
    Args:
        query: Texto de la consulta
        top_k: Número máximo de resultados
        
    Returns:
        Resultados de la búsqueda con texto relevante
    """
    global rag_pipeline, rag_cache
    
    # Verificar si la consulta necesita RAG
    needs_rag = await query_needs_rag(query)
    if not needs_rag:
        return {
            "query": query,
            "context": "Esta consulta no requiere información específica sobre Curaçao.",
            "chunks": [],
            "chunk_count": 0,
            "needs_rag": False
        }
    
    # Verificar caché
    cache_key = f"{query.lower()}_{top_k}"
    if cache_key in rag_cache and (time.time() - rag_cache[cache_key]["timestamp"]) < 3600:  # 1 hora TTL
        logger.info(f"Resultado encontrado en caché para: '{query}'")
        return rag_cache[cache_key]["result"]
    
    # Inicializar RAG si es necesario
    if not rag_pipeline:
        logger.info("RAG no inicializado. Inicializando...")
        result = await initialize_rag()
        if result["status"] == "error":
            return {
                "status": "error",
                "message": "No se pudo inicializar el sistema RAG",
                "error_details": result["message"]
            }
    
    try:
        logger.info(f"Buscando en RAG: '{query}'")
        
        # Realizar búsqueda híbrida con timeout
        try:
            results = await asyncio.wait_for(
                rag_pipeline.search(
                    query_text=query,
                    mode="hybrid",
                    top_k=top_k,
                    strategy="relevance"
                ),
                timeout=TIMEOUT_SECONDS
            )
            
            # Si no hay resultados con búsqueda híbrida, intentar con búsqueda semántica
            if len(results.get('chunks', [])) == 0:
                logger.info(f"Sin resultados en búsqueda híbrida, intentando búsqueda semántica para: '{query}'")
                semantic_results = await asyncio.wait_for(
                    rag_pipeline.search(
                        query_text=query,
                        mode="semantic",
                        top_k=top_k,
                        strategy="relevance"
                    ),
                    timeout=TIMEOUT_SECONDS
                )
                
                if len(semantic_results.get('chunks', [])) > 0:
                    logger.info(f"Búsqueda semántica exitosa: {len(semantic_results.get('chunks', []))} chunks encontrados")
                    results = semantic_results
        except asyncio.TimeoutError:
            logger.error(f"Timeout en búsqueda RAG para '{query}'")
            return {
                "status": "error",
                "message": "La búsqueda tardó demasiado tiempo",
                "query": query
            }
        
        logger.info(f"Búsqueda completada: {len(results.get('chunks', []))} chunks encontrados")
        
        # Extraer contexto y chunks
        context = results.get("context", "")
        chunks = results.get("chunks", [])
        
        # Estructurar respuesta
        response = {
            "query": query,
            "context": context,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "needs_rag": True
        }
        
        # Guardar en caché
        rag_cache[cache_key] = {
            "timestamp": time.time(),
            "result": response
        }
        
        # Limpiar caché si es demasiado grande
        if len(rag_cache) > 100:
            # Eliminar las entradas más antiguas
            oldest_keys = sorted(rag_cache.keys(), 
                                key=lambda k: rag_cache[k]["timestamp"])[:50]
            for key in oldest_keys:
                del rag_cache[key]
        
        return response
    except Exception as e:
        logger.error(f"Error en búsqueda RAG: {e}")
        
        # Intentar recuperar el pipeline si parece un error de conexión
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            logger.warning("Posible error de conexión, intentando reinicializar el pipeline...")
            rag_pipeline = None
            # No reinicializamos aquí para evitar recursión, lo hará en la próxima solicitud
        
        return {
            "status": "error",
            "message": f"Error en búsqueda RAG: {str(e)}",
            "query": query
        }

@mcp.tool()
async def search_about_place(place_name: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Busca información específica sobre un lugar turístico.
    
    Args:
        place_name: Nombre del lugar
        top_k: Número máximo de resultados
        
    Returns:
        Información específica sobre el lugar
    """
    # Ampliar la consulta para obtener mejores resultados
    query = f"Información sobre {place_name} en Curaçao características ubicación actividades"
    
    # Realizar búsqueda con la función principal, que ya implementa la lógica hybrid → semantic
    results = await search_knowledge(query, top_k)
    
    # Añadir nombre del lugar a los resultados
    results["place_name"] = place_name
    
    return results

@mcp.tool()
async def get_popular_places(category: str = None, limit: int = 5) -> Dict[str, Any]:
    """
    Obtiene lugares populares por categoría.
    
    Args:
        category: Categoría de lugares (playas, restaurantes, etc.)
        limit: Límite de resultados
        
    Returns:
        Lista de lugares populares
    """
    # Construir consulta según categoría
    if category:
        query = f"Lugares populares de {category} en Curaçao top mejores recomendados"
    else:
        query = "Lugares populares imprescindibles visitar Curaçao top atracciones"
    
    # Realizar búsqueda
    results = await search_knowledge(query, limit)
    
    # Estructurar resultados en formato adecuado para lugares
    places = []
    
    # Si hay chunks, extraer lugares mencionados
    chunks = results.get("chunks", [])
    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        
        # Extraer posibles nombres de lugares (simplificado)
        lines = chunk_text.split("\n")
        for line in lines:
            if ":" in line or "-" in line:
                parts = line.split(":", 1) if ":" in line else line.split("-", 1)
                if len(parts) == 2:
                    name = parts[0].strip()
                    description = parts[1].strip()
                    if name and len(name) < 50:  # Evitar líneas muy largas
                        places.append({
                            "name": name,
                            "description": description
                        })
    
    # Si no se encontraron lugares estructurados, extraer del contexto
    if not places and "context" in results:
        context = results["context"]
        places.append({
            "name": "Lugares recomendados",
            "description": context
        })
    
    return {
        "category": category,
        "places": places,
        "total": len(places),
        "query": query
    }

# Inicialización y funciones de mantenimiento
async def periodic_health_check():
    """Verifica periódicamente la salud del sistema y reinicia si es necesario."""
    global rag_pipeline
    
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutos
            
            # Verificar salud
            health = await get_server_health()
            
            # Si el servidor no está saludable, reintentar inicialización
            if health["status"] != "healthy":
                logger.warning(f"Verificación de salud fallida: {health}")
                rag_pipeline = None
                await initialize_rag()
        except Exception as e:
            logger.error(f"Error en verificación periódica: {e}")

# Inicializar RAG al arrancar
async def startup():
    logger.info("Iniciando servidor MCP RAG...")
    await initialize_rag()
    # Iniciar monitoreo de salud en background
    asyncio.create_task(periodic_health_check())

# Ejecutar inicialización
asyncio.run(startup())

if __name__ == "__main__":
    logger.info(f"Servidor MCP RAG iniciado: {RAG_SERVER_ID}")
    mcp.serve()