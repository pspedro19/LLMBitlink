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

# Modelos Pydantic mejorados
class ChatMessage(BaseModel):
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
    status: str
    entities: Optional[Dict[str, Any]] = None
    properties: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    suggestions: Optional[List[str]] = None
    processing_time: Optional[float] = None

# Configuración y utilidades
class APIConfig:
    def __init__(self):
        # Cargar variables de entorno
        load_dotenv()
        
        # Validar configuración
        self.validate_config()
        
        # Crear directorios necesarios
        self.setup_directories()
        
        # Configurar logging
        self.setup_logging()
    
    def validate_config(self):
        required_vars = ['OPENAI_API_KEY', 'DB_PATH']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    def setup_directories(self):
        os.makedirs('logs', exist_ok=True)
        os.makedirs(os.getenv('MEDIA_DIR', 'images'), exist_ok=True)
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(f'logs/api_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger("FastAPI-NER")

# Clase principal de la aplicación
class RealEstateAPI:
    def __init__(self):
        self.config = APIConfig()
        self.logger = self.config.setup_logging()
        self.ner_system = None
        self.db = None
        self.session = None
        self.app = self.create_app()
        
    async def initialize_session(self):
        """Inicializa la sesión aiohttp."""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
    async def initialize_db(self):
        """Inicializa la conexión a la base de datos."""
        if not self.db:
            self.db = await aiosqlite.connect(os.getenv("DB_PATH"))
            self.db.row_factory = aiosqlite.Row
        return self.db
    
    async def get_db(self):
        """
        Obtiene una conexión a la base de datos.
        """
        if not self.db:
            self.db = await self.initialize_db()
        return self.db

    def create_app(self) -> FastAPI:
        app = FastAPI(
            title="Real Estate NER API",
            description="API para análisis de texto y búsqueda de propiedades con NER mejorado",
            version="2.0.0",
            lifespan=self.lifespan
        )

        # Configurar CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Montar archivos estáticos
        media_dir = os.getenv('MEDIA_DIR', 'images')
        if os.path.exists(media_dir):
            app.mount("/media", StaticFiles(directory=media_dir), name="media")

        # Registrar rutas
        self.register_routes(app)

        return app

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Gestiona el ciclo de vida de la aplicación."""
        self.logger.info("Starting FastAPI application...")
        try:
            # Inicializar sesión HTTP
            self.session = aiohttp.ClientSession()
            
            # Inicializar NER
            self.ner_system = EnhancedNER(
                db_path=os.getenv("DB_PATH"),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                session=self.session,  # Pasar la sesión al NER
                max_requests=int(os.getenv("MAX_REQUESTS", "100")),  # Configurable por env
                time_window=int(os.getenv("TIME_WINDOW", "3600")) 
            )
            
            # Verificar conexión a DB
            db = await self.initialize_db()
            async with db.execute("SELECT COUNT(*) FROM chat_property") as cursor:
                row = await cursor.fetchone()
                self.logger.info(f"Connected to database. Total properties: {row[0]}")
            
            self.logger.info("Application startup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Startup failed: {e}")
            raise
        
        yield
        
        # Cleanup
        self.logger.info("Shutting down application...")
        if self.ner_system:
            await self.ner_system.close()
        if self.session:
            await self.session.close()
        if self.db:
            await self.db.close()
            self.db = None

    async def get_db(self):
        """Obtiene una conexión de la pool de base de datos."""
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
        """Registra todas las rutas de la API."""
        
        async def chat_endpoint(
            message: ChatMessage,
            background_tasks: BackgroundTasks
        ):
            start_time = datetime.now()
            try:
                # Analizar texto con NER
                analysis_result = await self.ner_system.analyze_text_async(message.user_input)
                
                # Obtener propiedades
                properties = []
                if analysis_result["status"] == "success":
                    # Creamos un diccionario con valores por defecto
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
                    # Actualizamos con el contexto proporcionado
                    if message.context:
                        filter_data.update(message.context)
                    
                    # Creamos el PropertyFilter con los datos
                    property_filter = PropertyFilter(**filter_data)
                    
                    properties = await self._get_properties_from_entities(
                        analysis_result["entities"],
                        property_filter
                    )
                
                # Log en background
                background_tasks.add_task(
                    self.logger.info,
                    f"Analysis completed: {json.dumps(analysis_result)}"
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Construir respuesta
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
        app.post("/chat/", response_model=AnalysisResponse)(chat_endpoint)
        
        @app.get("/health")
        async def health_check():
            """Endpoint de verificación de salud del servicio."""
            try:
                async with self.get_db() as conn:
                    cursor = await conn.execute("SELECT COUNT(*) FROM chat_property")
                    count = await cursor.fetchone()
                
                test_analysis = await self.ner_system.analyze_text_async("test")
                
                return {
                    "status": "healthy",
                    "database": {
                        "connected": True,
                        "property_count": count[0]
                    },
                    "ner_system": {
                        "status": "operational",
                        "last_test": test_analysis["status"]
                    },
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                )
        
        async def _get_properties_from_entities(
            self,
            entities: Dict[str, Any],
            filters: Optional[PropertyFilter] = None
        ) -> List[Dict[str, Any]]:
            """
            Consulta propiedades basadas en entidades identificadas y filtros.
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
                
                # Procesar entidades de ubicación
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
                
                # Procesar entidades de propiedades
                for prop in entities.get("properties", []):
                    if prop.db_match:
                        if prop.db_match["type"] == "property_type":
                            query_parts.append("p.property_type = ?")
                            params.append(prop.text)
                        elif prop.db_match["type"] == "price_range":
                            min_price, max_price = prop.db_match["matches"][0]["price_range"]
                            query_parts.append("p.price BETWEEN ? AND ?")
                            params.extend([min_price, max_price])
                
                # Aplicar filtros adicionales
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
                
                # Agregar group by y ordenamiento
                query += """
                    GROUP BY p.id
                    ORDER BY p.price_per_m2 ASC, p.created_at DESC
                    LIMIT 50
                """
                
                # Ejecutar query
                db = await self.get_db()
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    
                    # Convertir a diccionarios y procesar features
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
            
            # Construir resumen de búsqueda
            summary_parts = []
            
            if entities.get("locations"):
                locations = [
                    f"{loc.text} ({loc.db_match['type'] if loc.db_match else 'desconocido'})"
                    for loc in entities["locations"]
                ]
                summary_parts.append(f"ubicada en {', '.join(locations)}")
            
            if entities.get("properties"):
                props = [
                    f"{prop.text} ({prop.db_match['type'] if prop.db_match else 'desconocido'})"
                    for prop in entities["properties"]
                ]
                summary_parts.append(f"con características: {', '.join(props)}")

            # Mensaje introductorio
            intro = f"""
            <div class="message message-summary">
                <div class="message-content">
                    <p>He encontrado {len(properties)} propiedades {' '.join(summary_parts)}.</p>
                    <small>Tiempo de procesamiento: {processing_time:.2f} segundos</small>
                </div>
            </div>
            """
            response_parts.append(intro)

            # Grid de propiedades
            if properties:
                grid = '<div class="property-grid">'
                
                for prop in properties:
                    # Procesar imagen
                    image_url = prop.get('image_url', '/media/default.jpg')
                    if not image_url.startswith(('http://', 'https://', '/')):
                        image_url = f"/media/{image_url}"
                    
                    # Construir features
                    features = []
                    if prop.get('num_bedrooms'):
                        features.append(f'<span><i class="fas fa-bed"></i> {prop["num_bedrooms"]} hab</span>')
                    if prop.get('num_bathrooms'):
                        features.append(f'<span><i class="fas fa-bath"></i> {prop["num_bathrooms"]} baños</span>')
                    if prop.get('square_meters'):
                        features.append(f'<span><i class="fas fa-ruler-combined"></i> {prop["square_meters"]} m²</span>')
                    if prop.get('price_per_m2'):
                        features.append(f'<span><i class="fas fa-calculator"></i> ${prop["price_per_m2"]:.2f}/m²</span>')
                    
                    # Construir ubicación completa
                    location_parts = []
                    if prop.get('city_name'): location_parts.append(prop['city_name'])
                    if prop.get('province_name'): location_parts.append(prop['province_name'])
                    if prop.get('country_name'): location_parts.append(prop['country_name'])
                    
                    grid += f"""
                    <div class="property-card" data-id="{prop.get('id')}">
                        <div class="property-image-container">
                            <img src="{image_url}" 
                                alt="{prop.get('title', 'Propiedad')}" 
                                class="property-image"
                                loading="lazy">
                            <div class="property-price">
                                ${prop.get('price', 0):,.2f} USD
                            </div>
                        </div>
                        <div class="property-content">
                            <h3 class="property-title">{prop.get('title', prop.get('property_type', 'Propiedad'))}</h3>
                            <div class="property-location">
                                <i class="fas fa-map-marker-alt"></i> 
                                {', '.join(location_parts)}
                            </div>
                            <div class="property-features">
                                {' '.join(features)}
                            </div>
                            <div class="property-description">
                                {prop.get('description', 'Sin descripción disponible')[:200]}...
                            </div>
                            <div class="property-tags">
                                {' '.join(f'<span class="tag">{feature}</span>' for feature in prop.get('features', []))}
                            </div>
                            <a href="{prop.get('url', '#')}" 
                            class="property-cta" 
                            target="_blank" 
                            rel="noopener noreferrer">
                                Ver Detalles
                            </a>
                        </div>
                    </div>
                    """
                
                grid += '</div>'
                response_parts.append(grid)
                
            else:
                response_parts.append("""
                <div class="message message-empty">
                    <div class="message-content">
                        <p>No encontré propiedades que coincidan exactamente con tus criterios.</p>
                        <p>Prueba ajustando algunos filtros o usando términos más generales.</p>
                    </div>
                </div>
                """)

            # Agregar sugerencias
            if analysis_result.get("suggestions"):
                suggestions = """
                <div class="message message-suggestions">
                    <div class="message-content">
                        <p>También podrías estar interesado en:</p>
                        <ul class="suggestions-list">
                        """
                for suggestion in analysis_result["suggestions"]:
                    suggestions += f'<li><a href="#" class="suggestion-link">{suggestion}</a></li>'
                suggestions += "</ul></div></div>"
                response_parts.append(suggestions)

            return "\n".join(response_parts)
    
        @app.post("/analyze")
        async def analyze_text(
            self,
            text: str,
            background_tasks: BackgroundTasks,
            detailed: bool = False
        ) -> Dict[str, Any]:
            """
            Endpoint para análisis de texto sin búsqueda de propiedades.
            """
            try:
                start_time = datetime.now()
                
                # Análisis asíncrono
                analysis_result = await self.ner_system.analyze_text_async(text)
                
                # Log en background
                background_tasks.add_task(
                    self.logger.info,
                    f"Text analysis completed: {json.dumps(analysis_result)}"
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                if not detailed:
                    # Versión simplificada
                    return {
                        "status": "success",
                        "entities": {
                            "locations": [
                                loc.text for loc in analysis_result["entities"].get("locations", [])
                            ],
                            "properties": [
                                prop.text for prop in analysis_result["entities"].get("properties", [])
                            ]
                        },
                        "processing_time": processing_time
                    }
                
                # Versión detallada
                return {
                    **analysis_result,
                    "processing_time": processing_time
                }
                
            except Exception as e:
                self.logger.error(f"Error in analyze endpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/search", response_model=None)
        async def search_properties(
            self,
            filters: PropertyFilter,
            background_tasks: BackgroundTasks,
            response_format: str = "json",
            request: Request = None
        ):
            """
            Endpoint para búsqueda directa de propiedades con filtros.
            """
            try:
                start_time = datetime.now()
                
                # Convertir filtros en análisis de entidades
                entities = {
                    "locations": [],
                    "properties": []
                }
                
                if filters.location:
                    analysis = await self.ner_system.analyze_text_async(filters.location)
                    if analysis["status"] == "success":
                        entities["locations"].extend(
                            analysis["entities"].get("locations", [])
                        )
                
                # Obtener propiedades
                properties = await _get_properties_from_entities(entities, filters)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Log en background
                background_tasks.add_task(
                    self.logger.info,
                    f"Property search completed with filters: {json.dumps(filters.model_dump())}"
                )
                
                # Preparar respuesta según el formato solicitado
                if response_format == "html":
                    html_content = self.build_html_response(
                        {"status": "success", "entities": entities},
                        properties,
                        processing_time
                    )
                    return HTMLResponse(
                        content=html_content,
                        status_code=200
                    )
                
                # Respuesta JSON por defecto
                return {
                    "status": "success",
                    "properties": properties,
                    "total": len(properties),
                    "filters_applied": filters.model_dump(exclude_unset=True),
                    "processing_time": processing_time
                }
                
            except Exception as e:
                self.logger.error(f"Error in search endpoint: {e}")
                if response_format == "html":
                    error_html = f"""
                    <div class="error-message">
                        <p>Error: {str(e)}</p>
                    </div>
                    """
                    return HTMLResponse(
                        content=error_html,
                        status_code=500
                    )
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
            """
            try:
                start_time = datetime.now()
                
                # Obtener sugerencias de forma asíncrona
                suggestions = await self.ner_system.get_query_suggestions_async(
                    text,
                    limit=limit,
                    min_confidence=min_confidence
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    "status": "success",
                    "suggestions": suggestions,
                    "query": text,
                    "processing_time": processing_time
                }
                
            except Exception as e:
                self.logger.error(f"Error in suggestions endpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/stats")
        async def get_statistics(self) -> Dict[str, Any]:
            """
            Endpoint para obtener estadísticas del sistema.
            """
            try:
                async with self.get_db() as conn:
                    stats = {}
                    
                    # Estadísticas de propiedades
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
                    
                    # Distribución de tipos de propiedad
                    cursor = await conn.execute("""
                        SELECT 
                            property_type,
                            COUNT(*) as count,
                            AVG(price) as avg_price
                        FROM chat_property
                        GROUP BY property_type
                        ORDER BY count DESC
                    """)
                    stats["property_types"] = [dict(row) for row in await cursor.fetchall()]
                    
                    return {
                        "status": "success",
                        "statistics": stats,
                        "timestamp": datetime.now().isoformat(),
                    }
                    
            except Exception as e:
                self.logger.error(f"Error getting statistics: {e}")
                raise HTTPException(status_code=500, detail=str(e))

# Inicialización y configuración final
def create_app() -> FastAPI:
    """
    Crea y configura la aplicación FastAPI.
    """
    api = RealEstateAPI()
    app = api.app
    
    # Middleware adicional para medición de rendimiento
    @app.middleware("http")
    async def add_performance_headers(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Middleware para logging de requests
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
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
    
    # Calcular número óptimo de workers
    workers = min(multiprocessing.cpu_count() + 1, 4)
    
    # Configuración de uvicorn con valores optimizados
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
                    
