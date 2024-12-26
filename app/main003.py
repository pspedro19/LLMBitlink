"""
API FastAPI para Sistema de Reconocimiento de Entidades Nombradas en Bienes Raíces

Este módulo implementa una API RESTful para el procesamiento de consultas de bienes raíces
utilizando técnicas avanzadas de NLP. Integra el sistema EnhancedNER con FastAPI para
proporcionar endpoints de búsqueda, análisis y sugerencias.

Características principales:
- Procesamiento asíncrono de consultas
- Soporte para múltiples formatos de respuesta (JSON/HTML)
- Sistema de caché y rate limiting
- Gestión de conexiones a base de datos
- Logging completo
- Manejo de archivos estáticos
- Estadísticas del sistema

Author: [Tu nombre]
Version: 2.0.0
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import sqlite3
import logging
import aiosqlite
import aiohttp
from datetime import datetime
import os
from dotenv import load_dotenv
import asyncio
from functools import lru_cache
import json
from contextlib import asynccontextmanager
import uuid
import time
from app.enhanced_ner import EnhancedNER

class ChatMessage(BaseModel):
    """
    Modelo para mensajes de chat y consultas de usuario.
    
    Attributes:
        user_input (str): Texto de entrada del usuario.
        response_format (Optional[str]): Formato de respuesta ('html' o 'json').
        context (Optional[Dict[str, Any]]): Contexto adicional para el análisis.
        language (Optional[str]): Idioma del mensaje.
    """
    user_input: str = Field(..., description="Texto de entrada del usuario")
    response_format: Optional[str] = Field(
        default="html",
        description="Formato de respuesta: 'html' o 'json'"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Contexto adicional para el análisis"
    )
    language: Optional[str] = Field(
        default="es",
        description="Idioma del mensaje"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_input": "Busco apartamento en Madrid",
                "response_format": "html",
                "context": {"max_price": 300000},
                "language": "es"
            }
        }
    }

class PropertyFilter(BaseModel):
    """
    Modelo para filtros de búsqueda de propiedades.
    
    Attributes:
        location (Optional[str]): Ubicación deseada.
        min_price/max_price (Optional[float]): Rango de precios.
        property_type (Optional[str]): Tipo de propiedad.
        min_area/max_area (Optional[float]): Rango de área en metros cuadrados.
        bedrooms (Optional[int]): Número mínimo de dormitorios.
        features (Optional[List[str]]): Características deseadas.
    """
    location: Optional[str] = None
    min_price: Optional[float] = Field(default=None, ge=0)
    max_price: Optional[float] = Field(default=None, ge=0)
    property_type: Optional[str] = None
    min_area: Optional[float] = Field(default=None, ge=0)
    max_area: Optional[float] = Field(default=None, ge=0)
    bedrooms: Optional[int] = Field(default=None, ge=0)
    features: Optional[List[str]] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "location": "Madrid",
                "min_price": 100000,
                "max_price": 500000,
                "property_type": "apartment",
                "min_area": 50,
                "max_area": 150,
                "bedrooms": 2,
                "features": ["parking", "pool"]
            }
        }
    }

class AnalysisResponse(BaseModel):
    """
    Modelo para respuestas de análisis de texto.
    
    Attributes:
        status (str): Estado de la operación ('success' o 'error').
        entities (Optional[Dict[str, Any]]): Entidades identificadas.
        properties (Optional[List[Dict[str, Any]]]): Propiedades encontradas.
        error (Optional[str]): Mensaje de error si ocurrió alguno.
        suggestions (Optional[List[str]]): Sugerencias de búsqueda.
        processing_time (Optional[float]): Tiempo de procesamiento en segundos.
    """
    status: str
    entities: Optional[Dict[str, Any]] = None
    properties: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    suggestions: Optional[List[str]] = None
    processing_time: Optional[float] = None
    
class APIConfig:
    """
    Configuración de la API y gestión de recursos.
    
    Esta clase maneja la configuración inicial de la API, incluyendo:
    - Carga de variables de entorno
    - Validación de configuración
    - Creación de directorios necesarios
    - Configuración de logging
    """
    
    def __init__(self):
        """
        Inicializa la configuración de la API.
        
        Raises:
            ValueError: Si faltan variables de entorno requeridas.
        """
        load_dotenv()
        self.validate_config()
        self.setup_directories()
        self.setup_logging()
    
    def validate_config(self):
        """
        Valida las variables de entorno requeridas.
        
        Raises:
            ValueError: Si falta alguna variable de entorno obligatoria.
        """
        required_vars = ['OPENAI_API_KEY', 'DB_PATH']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    def setup_directories(self):
        """Crea los directorios necesarios para logs y archivos multimedia."""
        os.makedirs('logs', exist_ok=True)
        os.makedirs(os.getenv('MEDIA_DIR', 'images'), exist_ok=True)
    
    def setup_logging(self):
        """
        Configura el sistema de logging.
        
        Returns:
            logging.Logger: Logger configurado para la aplicación.
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(f'logs/api_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger("FastAPI-NER")

class RealEstateAPI:
    """
    Implementación principal de la API de bienes raíces.
    
    Esta clase maneja la inicialización de todos los componentes necesarios
    y la configuración de los endpoints de la API.
    
    Attributes:
        config (APIConfig): Configuración de la API.
        logger (logging.Logger): Logger para registro de eventos.
        ner_system (EnhancedNER): Sistema de reconocimiento de entidades.
        db (aiosqlite.Connection): Conexión a la base de datos.
        session (aiohttp.ClientSession): Sesión HTTP para requests.
        app (FastAPI): Aplicación FastAPI.
    """
    
    def __init__(self):
        """Inicializa la API y sus componentes."""
        self.config = APIConfig()
        self.logger = self.config.setup_logging()
        self.ner_system = None
        self.db = None
        self.session = None
        self.app = self.create_app()
        
    async def initialize_session(self):
        """
        Inicializa la sesión HTTP asincrónicamente.
        
        Returns:
            aiohttp.ClientSession: Sesión HTTP iniciada.
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
            
    async def initialize_db(self):
        """
        Inicializa la conexión a la base de datos asincrónicamente.
        
        Returns:
            aiosqlite.Connection: Conexión a la base de datos.
        """
        if not self.db:
            self.db = await aiosqlite.connect(os.getenv("DB_PATH"))
            self.db.row_factory = aiosqlite.Row
        return self.db

    async def get_db(self):
        """
        Obtiene una conexión a la base de datos.
        
        Returns:
            aiosqlite.Connection: Conexión activa a la base de datos.
        """
        if not self.db:
            self.db = await self.initialize_db()
        return self.db

    def create_app(self) -> FastAPI:
        """
        Crea y configura la aplicación FastAPI.
        
        Returns:
            FastAPI: Aplicación configurada con middleware y rutas.
        """
        app = FastAPI(
            title="Real Estate NER API",
            description="API para análisis de texto y búsqueda de propiedades con NER mejorado",
            version="2.0.0",
            lifespan=self.lifespan
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        media_dir = os.getenv('MEDIA_DIR', 'images')
        if os.path.exists(media_dir):
            app.mount("/media", StaticFiles(directory=media_dir), name="media")

        self.register_routes(app)
        return app
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """
        Gestiona el ciclo de vida de la aplicación.
        
        Args:
            app (FastAPI): Instancia de la aplicación.
            
        Yields:
            None: Control temporal durante la vida de la aplicación.
            
        Raises:
            Exception: Si hay errores durante la inicialización.
        """
        self.logger.info("Starting FastAPI application...")
        try:
            self.session = aiohttp.ClientSession()
            self.ner_system = EnhancedNER(
                db_path=os.getenv("DB_PATH"),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                session=self.session,
                max_requests=int(os.getenv("MAX_REQUESTS", "100")),
                time_window=int(os.getenv("TIME_WINDOW", "3600"))
            )
            
            db = await self.initialize_db()
            async with db.execute("SELECT COUNT(*) FROM chat_property") as cursor:
                row = await cursor.fetchone()
                self.logger.info(f"Connected to database. Total properties: {row[0]}")
            
            self.logger.info("Application startup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Startup failed: {e}")
            raise
        
        yield
        
        self.logger.info("Shutting down application...")
        if self.ner_system:
            await self.ner_system.close()
        if self.session:
            await self.session.close()
        if self.db:
            await self.db.close()
            self.db = None

    async def get_db(self):
        """
        Obtiene una conexión del pool de base de datos.

        Returns:
            aiosqlite.Connection: Conexión de la pool.
        """
        if not self.db_pool:
            self.db_pool = await aiosqlite.create_pool(
                os.getenv("DB_PATH"),
                min_size=5,
                max_size=20
            )
        async with self.db_pool.acquire() as conn:
            await conn.set_trace_callback(self.logger.debug)
            return conn

    def register_routes(self, app: FastAPI):
        """
        Registra todas las rutas de la API.
        
        Args:
            app (FastAPI): Instancia de la aplicación.
        """
        
        async def chat_endpoint(
            message: ChatMessage,
            background_tasks: BackgroundTasks
        ):
            """
            Endpoint principal para procesamiento de mensajes de chat.
            
            Args:
                message (ChatMessage): Mensaje del usuario.
                background_tasks (BackgroundTasks): Tareas en segundo plano.
                
            Returns:
                Union[HTMLResponse, AnalysisResponse]: Respuesta procesada.
                
            Raises:
                HTTPException: Si ocurre un error durante el procesamiento.
            """
            start_time = datetime.now()
            try:
                analysis_result = await self.ner_system.analyze_text_async(message.user_input)
                
                properties = []
                if analysis_result["status"] == "success":
                    filter_data = {
                        "location": None,
                        "min_price": None,
                        "max_price": None,
                        "property_type": None,
                        "min_area": None,
                        "max_area": None,
                        "bedrooms": None,
                        "features": None
                    }
                    if message.context:
                        filter_data.update(message.context)
                    
                    property_filter = PropertyFilter(**filter_data)
                    properties = await self._get_properties_from_entities(
                        analysis_result["entities"],
                        property_filter
                    )
                
                background_tasks.add_task(
                    self.logger.info,
                    f"Analysis completed: {json.dumps(analysis_result)}"
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                if message.response_format == "html":
                    html_content = self.build_html_response(
                        analysis_result,
                        properties,
                        processing_time
                    )
                    return HTMLResponse(content=html_content)
                
                return AnalysisResponse(
                    status="success",
                    entities=analysis_result.get("entities"),
                    properties=properties,
                    suggestions=await self.ner_system.get_query_suggestions_async(message.user_input),
                    processing_time=processing_time
                )
                
            except Exception as e:
                self.logger.error(f"Error in chat endpoint: {e}")
                return JSONResponse(
                    status_code=500,
                    content=AnalysisResponse(
                        status="error",
                        error=str(e),
                        processing_time=(datetime.now() - start_time).total_seconds()
                    ).model_dump()
                )
        
        async def _get_properties_from_entities(
            self,
            entities: Dict[str, Any],
            filters: Optional[PropertyFilter] = None
        ) -> List[Dict[str, Any]]:
            """
            Consulta propiedades basadas en entidades identificadas y filtros.
            
            Args:
                entities (Dict[str, Any]): Entidades identificadas por NER.
                filters (Optional[PropertyFilter]): Filtros adicionales.
                
            Returns:
                List[Dict[str, Any]]: Lista de propiedades que coinciden.
                
            Raises:
                HTTPException: Si hay error en la consulta.
            """
            try:
                query_parts = []
                params = []
                
                base_query = """
                    SELECT DISTINCT
                        p.*,
                        cc.name as country_name,
                        cp.name as province_name,
                        cct.name as city_name,
                        (p.price / NULLIF(p.square_meters, 0)) as price_per_m2,
                        GROUP_CONCAT(pf.feature_name) as features
                    FROM chat_property p
                    LEFT JOIN chat_country cc ON p.country_id = cc.id
                    LEFT JOIN chat_province cp ON p.province_id = cp.id
                    LEFT JOIN chat_city cct ON p.city_id = cct.id
                    LEFT JOIN property_features pf ON p.id = pf.property_id
                    WHERE 1=1
                """
                
                # Procesar ubicaciones
                for loc in entities.get("locations", []):
                    if loc.db_match:
                        if loc.db_match["type"] == "country":
                            query_parts.append("p.country_id = ?")
                            params.append(loc.db_match["id"])
                        elif loc.db_match["type"] == "province":
                            query_parts.append("p.province_id = ?")
                            params.append(loc.db_match["id"])
                        elif loc.db_match["type"] == "city":
                            query_parts.append("p.city_id = ?")
                            params.append(loc.db_match["id"])
                
                # Procesar filtros de propiedad
                if filters:
                    if filters.min_price is not None:
                        query_parts.append("p.price >= ?")
                        params.append(filters.min_price)
                    if filters.max_price is not None:
                        query_parts.append("p.price <= ?")
                        params.append(filters.max_price)
                    if filters.property_type:
                        query_parts.append("p.property_type LIKE ?")
                        params.append(f"%{filters.property_type}%")
                    if filters.min_area is not None:
                        query_parts.append("p.square_meters >= ?")
                        params.append(filters.min_area)
                    if filters.max_area is not None:
                        query_parts.append("p.square_meters <= ?")
                        params.append(filters.max_area)
                    if filters.bedrooms is not None:
                        query_parts.append("p.num_bedrooms >= ?")
                        params.append(filters.bedrooms)
                    if filters.features:
                        for feature in filters.features:
                            query_parts.append("""
                                EXISTS (
                                    SELECT 1 FROM property_features pf2 
                                    WHERE pf2.property_id = p.id 
                                    AND pf2.feature_name LIKE ?
                                )
                            """)
                            params.append(f"%{feature}%")
                
                # Construir query final
                query = base_query
                if query_parts:
                    query += " AND " + " AND ".join(query_parts)
                
                query += """
                    GROUP BY p.id
                    ORDER BY p.price_per_m2 ASC, p.created_at DESC
                    LIMIT 50
                """
                
                db = await self.get_db()
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    properties = []
                    for row in rows:
                        prop_dict = dict(row)
                        if prop_dict.get('features'):
                            prop_dict['features'] = prop_dict['features'].split(',')
                        properties.append(prop_dict)
                    
                    return properties
                    
            except Exception as e:
                self.logger.error(f"Error querying properties: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            
        def build_html_response(
            self,
            analysis_result: Dict[str, Any],
            properties: List[Dict[str, Any]],
            processing_time: float
        ) -> str:
            """
            Construye una respuesta HTML rica con los resultados.

            Args:
                analysis_result (Dict[str, Any]): Resultado del análisis NER.
                properties (List[Dict[str, Any]]): Propiedades encontradas.
                processing_time (float): Tiempo de procesamiento en segundos.

            Returns:
                str: HTML formateado con los resultados.
            """
            if analysis_result["status"] != "success":
                return f"""
                <div class="message message-error">
                    <div class="message-content">
                        <p>Lo siento, hubo un problema al procesar tu consulta: 
                        {analysis_result.get('error', 'Error desconocido')}</p>
                    </div>
                </div>
                """

            response_parts = []
            entities = analysis_result["entities"]
            
            # Construir resumen
            summary_parts = []
            if entities.get("locations"):
                locations = [f"{loc.text} ({loc.db_match['type'] if loc.db_match else 'desconocido'})"
                            for loc in entities["locations"]]
                summary_parts.append(f"ubicada en {', '.join(locations)}")
            
            if entities.get("properties"):
                props = [f"{prop.text} ({prop.db_match['type'] if prop.db_match else 'desconocido'})"
                        for prop in entities["properties"]]
                summary_parts.append(f"con características: {', '.join(props)}")

            # Generar HTML para cada sección
            intro = f"""
            <div class="message message-summary">
                <div class="message-content">
                    <p>He encontrado {len(properties)} propiedades {' '.join(summary_parts)}.</p>
                    <small>Tiempo de procesamiento: {processing_time:.2f} segundos</small>
                </div>
            </div>
            """
            response_parts.append(intro)

            if properties:
                grid = self._build_property_grid(properties)
                response_parts.append(grid)
            else:
                response_parts.append(self._build_empty_results_message())

            if analysis_result.get("suggestions"):
                suggestions = self._build_suggestions_section(analysis_result["suggestions"])
                response_parts.append(suggestions)

            return "\n".join(response_parts)

        def _build_property_grid(self, properties: List[Dict[str, Any]]) -> str:
            """
            Construye el grid HTML de propiedades.

            Args:
                properties (List[Dict[str, Any]]): Lista de propiedades.

            Returns:
                str: HTML del grid de propiedades.
            """
            grid = '<div class="property-grid">'
            for prop in properties:
                image_url = self._get_property_image_url(prop)
                features = self._build_property_features(prop)
                location = self._build_property_location(prop)
                
                grid += f"""
                <div class="property-card" data-id="{prop.get('id')}">
                    <div class="property-image-container">
                        <img src="{image_url}" alt="{prop.get('title', 'Propiedad')}" 
                            class="property-image" loading="lazy">
                        <div class="property-price">${prop.get('price', 0):,.2f} USD</div>
                    </div>
                    <div class="property-content">
                        <h3 class="property-title">
                            {prop.get('title', prop.get('property_type', 'Propiedad'))}
                        </h3>
                        {location}
                        {features}
                        <div class="property-description">
                            {prop.get('description', 'Sin descripción disponible')[:200]}...
                        </div>
                        <div class="property-tags">
                            {' '.join(f'<span class="tag">{feature}</span>' 
                                    for feature in prop.get('features', []))}
                        </div>
                        <a href="{prop.get('url', '#')}" class="property-cta" 
                        target="_blank" rel="noopener noreferrer">Ver Detalles</a>
                    </div>
                </div>
                """
            grid += '</div>'
            return grid

        @app.post("/analyze")
        async def analyze_text(
            self,
            text: str,
            background_tasks: BackgroundTasks,
            detailed: bool = False
        ) -> Dict[str, Any]:
            """
            Endpoint para análisis de texto sin búsqueda de propiedades.

            Args:
                text (str): Texto a analizar.
                background_tasks (BackgroundTasks): Tareas en segundo plano.
                detailed (bool): Si se debe retornar análisis detallado.

            Returns:
                Dict[str, Any]: Resultado del análisis.

            Raises:
                HTTPException: Si hay error en el análisis.
            """
            try:
                start_time = datetime.now()
                analysis_result = await self.ner_system.analyze_text_async(text)
                
                background_tasks.add_task(
                    self.logger.info,
                    f"Text analysis completed: {json.dumps(analysis_result)}"
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                if not detailed:
                    return {
                        "status": "success",
                        "entities": {
                            "locations": [loc.text for loc in 
                                        analysis_result["entities"].get("locations", [])],
                            "properties": [prop.text for prop in 
                                        analysis_result["entities"].get("properties", [])]
                        },
                        "processing_time": processing_time
                    }
                
                return {
                    **analysis_result,
                    "processing_time": processing_time
                }
                
            except Exception as e:
                self.logger.error(f"Error in analyze endpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            
        @app.get("/suggestions/{text}")
        async def get_suggestions(
            self,
            text: str,
            limit: int = 5,
            min_confidence: float = 0.5
        ) -> Dict[str, Any]:
            """
            Endpoint para obtener sugerencias de búsqueda.

            Args:
                text (str): Texto base para sugerencias.
                limit (int): Máximo de sugerencias a retornar.
                min_confidence (float): Confianza mínima requerida.

            Returns:
                Dict[str, Any]: Sugerencias generadas.
            """
            try:
                start_time = datetime.now()
                suggestions = await self.ner_system.get_query_suggestions_async(
                    text,
                    limit=limit,
                    min_confidence=min_confidence
                )
                return {
                    "status": "success",
                    "suggestions": suggestions,
                    "query": text,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            except Exception as e:
                self.logger.error(f"Error in suggestions endpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/stats")
        async def get_statistics(self) -> Dict[str, Any]:
            """
            Endpoint para obtener estadísticas del sistema.

            Returns:
                Dict[str, Any]: Estadísticas del sistema incluyendo:
                    - Métricas de propiedades
                    - Top ubicaciones
                    - Distribución de tipos de propiedad
            """
            try:
                async with self.get_db() as conn:
                    stats = {}
                    
                    # Estadísticas generales
                    cursor = await conn.execute("""
                        SELECT 
                            COUNT(*) as total_properties,
                            AVG(price) as avg_price,
                            MIN(price) as min_price,
                            MAX(price) as max_price,
                            AVG(square_meters) as avg_area,
                            COUNT(DISTINCT country_id) as num_countries,
                            COUNT(DISTINCT city_id) as num_cities
                        FROM chat_property
                    """)
                    stats["properties"] = dict(await cursor.fetchone())
                    
                    # Top ubicaciones
                    cursor = await conn.execute("""
                        SELECT 
                            c.name as city,
                            p.name as province,
                            COUNT(*) as count
                        FROM chat_property prop
                        JOIN chat_city c ON prop.city_id = c.id
                        JOIN chat_province p ON c.province_id = p.id
                        GROUP BY city_id
                        ORDER BY count DESC
                        LIMIT 10
                    """)
                    stats["top_locations"] = [dict(row) for row in await cursor.fetchall()]
                    
                    return {
                        "status": "success",
                        "statistics": stats,
                        "timestamp": datetime.now().isoformat(),
                    }
                    
            except Exception as e:
                self.logger.error(f"Error getting statistics: {e}")
                raise HTTPException(status_code=500, detail=str(e))

def create_app() -> FastAPI:
    """
    Crea y configura la aplicación FastAPI.

    Returns:
        FastAPI: Aplicación configurada con middleware y rutas.
    """
    api = RealEstateAPI()
    app = api.app
    
    @app.middleware("http")
    async def add_performance_headers(request: Request, call_next):
        """
        Middleware para añadir headers de rendimiento.
        """
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """
        Middleware para logging de requests.
        """
        request_id = str(uuid.uuid4())
        api.logger.info(f"Request {request_id}: {request.method} {request.url}")
        try:
            response = await call_next(request)
            api.logger.info(f"Response {request_id}: {response.status_code}")
            return response
        except Exception as e:
            api.logger.error(f"Error {request_id}: {str(e)}")
            raise
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    import multiprocessing
    
    workers = min(multiprocessing.cpu_count() + 1, 4)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8800)),
        reload=os.getenv("ENVIRONMENT", "development") == "development",
        workers=workers,
        loop="uvloop",
        http="httptools",
        log_level="info",
        access_log=True,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )