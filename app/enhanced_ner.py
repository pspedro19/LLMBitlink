import openai
from typing import Dict, List, Any, Optional, Tuple, TypedDict, Literal
from dataclasses import dataclass
import sqlite3
import logging
import json
from datetime import datetime, timedelta
import spacy
from collections import defaultdict, deque
from functools import lru_cache
import asyncio
import aiohttp
import time

# Definición de tipos
class LocationEntity(TypedDict):
    type: Literal['country', 'province', 'city', 'zone']
    text: str
    confidence: float

class PropertyEntity(TypedDict):
    type: Literal['type', 'price', 'area', 'rooms', 'feature']
    text: str
    confidence: float

@dataclass
class EntityMatch:
    text: str
    confidence: float
    db_match: Optional[Dict[str, Any]] = None
    alternatives: List[str] = None

class RateLimiter:
    def __init__(self, max_requests: int, time_window: int):
        """
        Inicializa el rate limiter.
        
        Args:
            max_requests: Número máximo de solicitudes permitidas en la ventana de tiempo
            time_window: Ventana de tiempo en segundos
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()

    def can_make_request(self) -> bool:
        """
        Verifica si se puede realizar una nueva solicitud.
        
        Returns:
            bool: True si se puede realizar la solicitud, False en caso contrario
        """
        now = datetime.now()
        
        # Limpiar solicitudes antiguas
        while self.requests and now - self.requests[0] > timedelta(seconds=self.time_window):
            self.requests.popleft()
        
        # Verificar si podemos hacer una nueva solicitud
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False

class Cache:
    def __init__(self, ttl_seconds: int = 3600):
        """
        Inicializa el sistema de cache.
        
        Args:
            ttl_seconds: Tiempo de vida en segundos para las entradas del cache
        """
        self.cache = {}
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """
        Obtiene un valor del cache.
        
        Args:
            key: Clave a buscar
            
        Returns:
            El valor almacenado o None si no existe o expiró
        """
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Any):
        """
        Almacena un valor en el cache.
        
        Args:
            key: Clave para almacenar
            value: Valor a almacenar
        """
        self.cache[key] = (value, datetime.now())
        
class EnhancedNER:
    def __init__(
        self, 
        db_path: str, 
        openai_api_key: str, 
        session: Optional[aiohttp.ClientSession] = None,
        max_requests: int = 100,  # Valor por defecto para rate limiting
        time_window: int = 3600   # Ventana de tiempo por defecto (1 hora)
    ):
        """
        Inicializa el sistema NER mejorado.
        
        Args:
            db_path: Ruta a la base de datos SQLite
            openai_api_key: API key de OpenAI
            max_requests: Número máximo de solicitudes a GPT-4 por ventana de tiempo
            time_window: Ventana de tiempo en segundos para rate limiting
        """
        self.db_path = db_path
        self.openai_api_key = openai_api_key
        self.session = session
        self.logger = logging.getLogger("EnhancedNER")
        
        # Inicializar sistemas auxiliares
        self.rate_limiter = RateLimiter(max_requests, time_window)
        self.cache = Cache(ttl_seconds=3600)
        
        # Inicializar spaCy
        self.nlp = spacy.load("es_core_news_md")
        
        # Configurar logging
        self.setup_logging()
        
        # Cargar datos de la DB para validación rápida
        self._load_db_data()
        
        # Inicializar cliente aiohttp para requests asíncronos
        self.session = None

    async def __aenter__(self):
        """Setup para uso con context manager asíncrono."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup para uso con context manager asíncrono."""
        if self.session:
            await self.session.close()

    def setup_logging(self):
        """Configura el sistema de logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(f'ner_logs_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("EnhancedNER")

    def _load_db_data(self):
        """Carga datos de la DB para validación en memoria."""
        try:
            # Intentar obtener datos del cache
            cached_data = self.cache.get('db_data')
            if cached_data:
                self.countries, self.provinces, self.cities, self.properties = cached_data
                return

            with sqlite3.connect(self.db_path) as conn:
                # Cargar países
                self.countries = {
                    row[0]: row[1] for row in conn.execute(
                        "SELECT id, name FROM chat_country"
                    )
                }
                
                # Cargar provincias con sus países
                self.provinces = {
                    row[0]: {"name": row[1], "country_id": row[2]} 
                    for row in conn.execute(
                        "SELECT id, name, country_id FROM chat_province"
                    )
                }
                
                # Cargar ciudades con sus provincias
                self.cities = {
                    row[0]: {"name": row[1], "province_id": row[2]}
                    for row in conn.execute(
                        "SELECT id, name, province_id FROM chat_city"
                    )
                }
                
                # Cargar propiedades básicas para validación
                self.properties = {
                    row[0]: {
                        "type": row[1],
                        "location": row[2],
                        "price_range": (row[3], row[4])
                    }
                    for row in conn.execute("""
                        SELECT id, property_type, location, 
                               MIN(price) as min_price, 
                               MAX(price) as max_price 
                        FROM chat_property 
                        GROUP BY property_type, location
                    """)
                }
                
                # Guardar en cache
                self.cache.set('db_data', (
                    self.countries,
                    self.provinces,
                    self.cities,
                    self.properties
                ))
                
        except sqlite3.Error as e:
            self.logger.error(f"Error cargando datos de DB: {e}")
            raise

    def _sanitize_input(self, text: str) -> str:
        """
        Sanitiza el input para prevenir inyección SQL y otros ataques.
        
        Args:
            text: Texto a sanitizar
            
        Returns:
            str: Texto sanitizado
        """
        # Eliminar caracteres potencialmente peligrosos
        text = ''.join(char for char in text if char.isprintable())
        
        # Escapar caracteres especiales SQL
        text = text.replace("'", "''")
        
        return text

    def _get_gpt_prompt(self, text: str) -> str:
        """
        Genera el prompt para GPT-4 con ejemplos few-shot.
        
        Args:
            text: Texto a analizar
            
        Returns:
            str: Prompt formateado
        """
        return f"""Actúa como un sistema experto de reconocimiento de entidades nombradas (NER).
        Analiza el siguiente texto e identifica todas las entidades relacionadas con propiedades inmobiliarias.
        
        Debes identificar y clasificar:
        1. Ubicaciones (LOC):
           - Países
           - Provincias/Estados
           - Ciudades
           - Barrios/Zonas
        
        2. Propiedades (PROP):
           - Tipo de propiedad
           - Precio
           - Metros cuadrados
           - Número de habitaciones
           - Características especiales
        
        Devuelve la respuesta en formato JSON con la siguiente estructura:
        {{
            "locations": [
                {{"type": "country|province|city|zone", "text": "nombre", "confidence": 0.0-1.0}}
            ],
            "properties": [
                {{"type": "type|price|area|rooms|feature", "text": "valor", "confidence": 0.0-1.0}}
            ]
        }}
        
        Texto a analizar: {self._sanitize_input(text)}
        """

    @lru_cache(maxsize=1000)
    async def _query_gpt_async(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Consulta a GPT-4 de forma asíncrona para extraer entidades.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Dict con las entidades extraídas
        """
        if not self.rate_limiter.can_make_request():
            raise Exception("Rate limit exceeded for GPT-4 requests")

        try:
            # Verificar cache
            cache_key = f"gpt_query_{hash(text)}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result

            async with self.session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai.api_key}"},
                json={
                    "model": "gpt-4",
                    "messages": [
                        {"role": "system", "content": self._get_gpt_prompt(text)}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500
                }
            ) as response:
                result = await response.json()
                entities = json.loads(result["choices"][0]["message"]["content"])
                
                # Guardar en cache
                self.cache.set(cache_key, entities)
                
                return entities
                
        except Exception as e:
            self.logger.error(f"Error consultando GPT: {e}")
            return {"locations": [], "properties": []}

    async def _validate_with_spacy_async(
        self, 
        text: str, 
        gpt_entities: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Valida y complementa las entidades de GPT con spaCy de forma asíncrona.
        
        Args:
            text: Texto a analizar
            gpt_entities: Entidades extraídas por GPT
            
        Returns:
            Dict con entidades combinadas y validadas
        """
        # Procesar con spaCy (esto es CPU-bound, usar run_in_executor)
        doc = await asyncio.get_event_loop().run_in_executor(
            None, self.nlp, text
        )
        
        # Extraer entidades de spaCy
        spacy_entities = defaultdict(list)
        for ent in doc.ents:
            if ent.label_ in ["LOC", "GPE"]:
                spacy_entities["locations"].append({
                    "type": "location",
                    "text": ent.text,
                    "confidence": ent._.confidence if hasattr(ent._, 'confidence') else 0.8
                })
            
        # Combinar y validar entidades
        combined_entities = {
            "locations": self._merge_entities(
                gpt_entities.get("locations", []),
                spacy_entities.get("locations", [])
            ),
            "properties": gpt_entities.get("properties", [])
        }
        
        return combined_entities

    def _merge_entities(
        self, 
        gpt_ents: List[Dict], 
        spacy_ents: List[Dict]
    ) -> List[Dict]:
        """
        Combina y deduplica entidades de GPT y spaCy.
        
        Args:
            gpt_ents: Entidades de GPT
            spacy_ents: Entidades de spaCy
            
        Returns:
            Lista de entidades combinadas y deduplicadas
        """
        merged = []
        seen = set()
        
        for ent in gpt_ents + spacy_ents:
            text = ent["text"].lower()
            if text not in seen:
                seen.add(text)
                merged.append(ent)
                
        return merged
    
    async def _validate_with_db_async(
        self, 
        entities: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[EntityMatch]]:
        """
        Valida las entidades contra la base de datos de forma asíncrona.
        
        Args:
            entities: Diccionario con entidades a validar
            
        Returns:
            Dict con entidades validadas y coincidencias en DB
        """
        validated = {
            "locations": [],
            "properties": []
        }
        
        # Validar ubicaciones
        location_tasks = [
            self._find_best_location_match_async(loc["text"], loc["confidence"])
            for loc in entities["locations"]
        ]
        location_results = await asyncio.gather(*location_tasks)
        
        for loc, match in zip(entities["locations"], location_results):
            if match:
                validated["locations"].append(EntityMatch(
                    text=loc["text"],
                    confidence=loc["confidence"],
                    db_match=match
                ))
        
        # Validar propiedades
        property_tasks = [
            self._find_property_match_async(prop)
            for prop in entities["properties"]
        ]
        property_results = await asyncio.gather(*property_tasks)
        
        for prop, match in zip(entities["properties"], property_results):
            if match:
                validated["properties"].append(EntityMatch(
                    text=prop["text"],
                    confidence=prop["confidence"],
                    db_match=match
                ))
                
        return validated

    async def _find_best_location_match_async(
        self, 
        location: str, 
        confidence: float
    ) -> Optional[Dict[str, Any]]:
        """
        Encuentra la mejor coincidencia para una ubicación en la DB de forma asíncrona.
        
        Args:
            location: Texto de la ubicación
            confidence: Nivel de confianza de la entidad
            
        Returns:
            Dict con la información de la coincidencia o None
        """
        # Normalizar texto
        location = location.lower().strip()
        
        # Verificar cache
        cache_key = f"location_match_{location}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Buscar en países
        for id_, name in self.countries.items():
            if location == name.lower():
                result = {
                    "type": "country",
                    "id": id_,
                    "name": name,
                    "confidence": confidence
                }
                self.cache.set(cache_key, result)
                return result
        
        # Buscar en provincias
        for id_, data in self.provinces.items():
            if location == data["name"].lower():
                result = {
                    "type": "province",
                    "id": id_,
                    "name": data["name"],
                    "country_id": data["country_id"],
                    "confidence": confidence
                }
                self.cache.set(cache_key, result)
                return result
        
        # Buscar en ciudades
        for id_, data in self.cities.items():
            if location == data["name"].lower():
                result = {
                    "type": "city",
                    "id": id_,
                    "name": data["name"],
                    "province_id": data["province_id"],
                    "confidence": confidence
                }
                self.cache.set(cache_key, result)
                return result
        
        return None

    async def _find_property_match_async(
        self, 
        property_entity: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Encuentra coincidencias de propiedades en la DB de forma asíncrona.
        
        Args:
            property_entity: Entidad de propiedad a validar
            
        Returns:
            Dict con coincidencias encontradas o None
        """
        prop_type = property_entity["type"]
        text = property_entity["text"]
        confidence = property_entity["confidence"]
        
        # Verificar cache
        cache_key = f"property_match_{hash(f'{prop_type}_{text}')}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        if prop_type == "type":
            # Buscar coincidencias de tipo de propiedad
            matches = [
                prop for prop in self.properties.values()
                if text.lower() in prop["type"].lower()
            ]
            if matches:
                result = {
                    "type": "property_type",
                    "matches": matches,
                    "confidence": confidence
                }
                self.cache.set(cache_key, result)
                return result
        
        elif prop_type == "price":
            # Validar rango de precios
            try:
                price = float(text.replace("$", "").replace(",", ""))
                matches = [
                    prop for prop in self.properties.values()
                    if prop["price_range"][0] <= price <= prop["price_range"][1]
                ]
                if matches:
                    result = {
                        "type": "price_range",
                        "matches": matches,
                        "confidence": confidence
                    }
                    self.cache.set(cache_key, result)
                    return result
            except ValueError:
                pass
        
        return None

    def _process_gpt_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
            """
            Procesa la respuesta de GPT y la convierte en el formato esperado.
            
            Args:
                response: Respuesta raw de la API de GPT
                
            Returns:
                Dict con el resultado procesado
            """
            try:
                if "choices" not in response or not response["choices"]:
                    return {
                        "status": "error",
                        "error": "No response from GPT"
                    }
                    
                content = response["choices"][0]["message"]["content"]
                parsed_content = json.loads(content)
                
                return {
                    "status": "success",
                    "entities": {
                        "locations": [
                            {
                                "type": loc["type"],
                                "text": loc["text"],
                                "confidence": loc["confidence"]
                            }
                            for loc in parsed_content.get("locations", [])
                        ],
                        "properties": [
                            {
                                "type": prop["type"],
                                "text": prop["text"],
                                "confidence": prop["confidence"]
                            }
                            for prop in parsed_content.get("properties", [])
                        ]
                    },
                    "query_text": self._sanitize_input(content),
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                self.logger.error(f"Error processing GPT response: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
    # async def analyze_text_async(self, text: str) -> Dict[str, Any]:
    #     """
    #     Analiza un texto para extraer y validar entidades de forma asíncrona.
        
    #     Args:
    #         text: Texto a analizar
            
    #     Returns:
    #         Dict con entidades validadas y coincidencias en DB
    #     """
    #     try:
    #         if not self.session:
    #             self.session = aiohttp.ClientSession()
                
    #         # 1. Obtener entidades de GPT
    #         gpt_entities = await self._query_gpt_async(text)
            
    #         # 2. Validar con spaCy
    #         combined_entities = await self._validate_with_spacy_async(text, gpt_entities)
            
    #         # 3. Validar contra DB
    #         validated_entities = await self._validate_with_db_async(combined_entities)
            
    #         # 4. Generar resultado final
    #         result = {
    #             "status": "success",
    #             "entities": validated_entities,
    #             "query_text": text,
    #             "timestamp": datetime.now().isoformat()
    #         }
            
    #         # Logging
    #         self.logger.info(f"Analysis completed successfully for text: {text[:100]}...")
            
    #         return result
            
    #     except Exception as e:
    #         self.logger.error(f"Error analyzing text: {e}")
    #         return {
    #             "status": "error",
    #             "error": str(e),
    #             "query_text": text,
    #             "timestamp": datetime.now().isoformat()
    #         }

    async def analyze_text_async(self, text: str) -> Dict[str, Any]:
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            # Usar la sesión para la llamada a la API
            async with self.session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",
                    "messages": [
                        {"role": "system", "content": self._get_gpt_prompt(text)}
                    ]
                }
            ) as response:
                result = await response.json()
                return self._process_gpt_response(result)
                
        except Exception as e:
            self.logger.error(f"Error consultando GPT: {e}")
            return {"status": "error", "error": str(e)}

    async def close(self):
        """Cierra la sesión HTTP si fue creada internamente"""
        if self.session:
            await self.session.close()
    
    async def get_query_suggestions_async(self, text: str) -> List[str]:
        """
        Genera sugerencias de búsqueda basadas en el análisis de entidades de forma asíncrona.
        
        Args:
            text: Texto base para generar sugerencias
            
        Returns:
            Lista de sugerencias de búsqueda
        """
        try:
            # Verificar cache
            cache_key = f"suggestions_{hash(text)}"
            cached_suggestions = self.cache.get(cache_key)
            if cached_suggestions:
                return cached_suggestions
            
            analysis = await self.analyze_text_async(text)
            if analysis["status"] != "success":
                return []
                
            suggestions = []
            entities = analysis["entities"]
            
            # Sugerir basado en ubicaciones
            for loc in entities["locations"]:
                if loc.db_match:
                    if loc.db_match["type"] == "city":
                        suggestions.append(f"Propiedades en {loc.text}")
                        suggestions.append(f"Departamentos en {loc.text}")
                        suggestions.append(f"Casas en {loc.text}")
            
            # Sugerir basado en propiedades
            for prop in entities["properties"]:
                if prop.db_match and prop.db_match["type"] == "property_type":
                    for loc in entities["locations"]:
                        if loc.db_match:
                            suggestions.append(
                                f"{prop.text} en {loc.text}"
                            )
            
            # Almacenar en cache
            suggestions = suggestions[:5]  # Limitar a 5 sugerencias
            self.cache.set(cache_key, suggestions)
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating suggestions: {e}")
            return []