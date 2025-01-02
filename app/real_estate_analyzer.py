import sqlite3
import json
import pandas as pd
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Set, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import anthropic
from anthropic import Anthropic
import spacy
from spacy.tokens import Doc
from spacy.language import Language
import pycountry
import Levenshtein
import unicodedata
import re
from anthropic import Anthropic
from collections import defaultdict

# Definir el modelo de Claude a usar
CLAUDE_MODEL = "claude-3-sonnet-20240229"
    
@dataclass
class EntityContext:
    """Contexto para entidades geográficas y de propiedades"""
    preceding: str
    following: str
    related_entities: List[Dict[str, str]]

@dataclass
class NumberContext:
    """Contexto para valores numéricos (precios, áreas, etc.)"""
    type: Optional[str]
    unit: Optional[str]

@dataclass
class Entity:
    """Entidad extraída del texto con su contexto"""
    text: str
    start: int
    end: int
    context: Union[EntityContext, NumberContext]

@dataclass
class GeoEntity:
    """Entidad geográfica con metadata y relaciones"""
    name: str
    type: str
    confidence: float
    normalized_name: str
    alternatives: List[str]
    parent_entity: Optional['GeoEntity'] = None
    metadata: Dict[str, Any] = None

class EntityExtractorInterface(ABC):
    """Interfaz abstracta para extractores de entidades"""
    @abstractmethod
    def extract_entities(self, text: str) -> Dict[str, List[Entity]]:
        pass

class QueryBuilderInterface(ABC):
    """Interfaz abstracta para constructores de consultas"""
    @abstractmethod
    def build_query(self, entities: Dict[str, List[Entity]]) -> Tuple[str, List[Any]]:
        pass

class GeoEntityManager:
    """Gestiona las entidades geográficas y su normalización"""
    
    def __init__(self):
        self.country_names: Dict[str, Set[str]] = defaultdict(set)
        self.city_names: Dict[str, Set[str]] = defaultdict(set)
        self.region_names: Dict[str, Set[str]] = defaultdict(set)
        self.location_cache: Dict[str, List[GeoEntity]] = {}
        self._init_geo_data()

    # def _init_geo_data(self):
    #     """Inicializa los datos geográficos desde pycountry"""
    #     for country in pycountry.countries:
    #         self._add_country_name(country.name, country.alpha_2)
    #         if hasattr(country, 'common_name'):
    #             self._add_country_name(country.common_name, country.alpha_2)
    #         if hasattr(country, 'official_name'):
    #             self._add_country_name(country.official_name, country.alpha_2)

    def _init_geo_data(self):
        """Inicializa los datos geográficos desde pycountry y la base de datos"""
        # Inicializar países desde pycountry
        self._init_countries()
        # Inicializar ciudades y regiones desde la base de datos
        self._init_cities_and_regions()

    def _init_cities_and_regions(self):
        """Inicializa datos de ciudades y regiones desde la base de datos"""
        try:
            conn = sqlite3.connect(os.getenv('DB_PATH'))
            cursor = conn.cursor()

            # Cargar provincias/regiones
            cursor.execute("SELECT name, country_id FROM chat_province")
            for name, country_id in cursor.fetchall():
                self.region_names[country_id].add(self._normalize_text(name))

            # Cargar ciudades
            cursor.execute("SELECT name, province_id FROM chat_city")
            for name, province_id in cursor.fetchall():
                self.city_names[province_id].add(self._normalize_text(name))

        except sqlite3.Error as e:
            logging.error(f"Error cargando datos geográficos: {e}")
        finally:
            if conn:
                conn.close()
                
    def _init_countries(self):
        """Inicializa datos de países"""
        for country in pycountry.countries:
            self._add_country_name(country.name, country.alpha_2)
            if hasattr(country, 'common_name'):
                self._add_country_name(country.common_name, country.alpha_2)
            if hasattr(country, 'official_name'):
                self._add_country_name(country.official_name, country.alpha_2)
                
    def _normalize_text(self, text: str) -> str:
        """Normaliza el texto para comparaciones
        
        Args:
            text: Texto a normalizar
            
        Returns:
            Texto normalizado sin acentos, en minúsculas y sin caracteres especiales
        """
        text = text.lower()
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()

    def _add_country_name(self, name: str, country_code: str):
        """Añade un nombre de país y sus variaciones al diccionario
        
        Args:
            name: Nombre del país
            country_code: Código ISO del país
        """
        normalized = self._normalize_text(name)
        self.country_names[country_code].add(normalized)
        variations = self._generate_name_variations(normalized)
        self.country_names[country_code].update(variations)

    @staticmethod
    def _generate_name_variations(name: str) -> Set[str]:
        """Genera variaciones de nombres para mejor coincidencia
        
        Args:
            name: Nombre base para generar variaciones
            
        Returns:
            Conjunto de variaciones del nombre
        """
        variations = {name}
        variations.add(name.replace(' ', ''))
        variations.add(name.replace(' ', '-'))
        return variations

    def extract_geo_entities(self, text: str) -> List[Entity]:
        """Extrae entidades geográficas del texto
        
        Args:
            text: Texto donde buscar entidades geográficas
            
        Returns:
            Lista de entidades geográficas encontradas
        """
        entities = []
        normalized_text = self._normalize_text(text)
        
        # Buscar coincidencias de países
        for country_code, names in self.country_names.items():
            for name in names:
                if name in normalized_text:
                    start = text.lower().find(name)
                    if start != -1:
                        entities.append(Entity(
                            text=name,
                            start=start,
                            end=start + len(name),
                            context=EntityContext(
                                preceding=text[max(0, start-20):start],
                                following=text[start+len(name):min(len(text), start+len(name)+20)],
                                related_entities=[]
                            )
                        ))
        
        # Buscar coincidencias de ciudades
        for country_code, names in self.city_names.items():
            for name in names:
                if name in normalized_text:
                    start = text.lower().find(name)
                    if start != -1:
                        entities.append(Entity(
                            text=name,
                            start=start,
                            end=start + len(name),
                            context=EntityContext(
                                preceding=text[max(0, start-20):start],
                                following=text[start+len(name):min(len(text), start+len(name)+20)],
                                related_entities=[]
                            )
                        ))
                        
        # Buscar coincidencias por regiones o provincias
        for country_code, names in self.region_names.items():
            for name in names:
                if name in normalized_text:
                    start = text.lower().find(name)
                    if start != -1:
                        entities.append(Entity(
                            text=name,
                            start=start,
                            end=start + len(name),
                            context=EntityContext(
                                preceding=text[max(0, start-20):start],
                                following=text[start+len(name):min(len(text), start+len(name)+20)],
                                related_entities=[]
                            )
                        ))
        
        return entities
    
class EnhancedEntityExtractor(EntityExtractorInterface):
    """Implementación mejorada del extractor de entidades"""

    def __init__(self, nlp_models: List[Language], geo_manager: GeoEntityManager):
        self.nlp_models = nlp_models
        self.geo_manager = geo_manager
        self._init_entity_patterns()

    def _init_entity_patterns(self):
        """Inicializa los patrones de entidades para el reconocimiento"""
        patterns = {
            "PROP_TYPE": [
                "casa", "apartamento", "piso", "chalet", "villa", "estudio", 
                "duplex", "ático", "local", "oficina", "departamento",
                "casa de campo", "penthouse", "loft"
            ],
            "FEATURE": [
                "dormitorios", "habitaciones", "baños", "metros cuadrados",
                "m2", "garage", "jardín", "piscina", "terraza", "balcón",
                "cochera", "estacionamiento", "ascensor", "seguridad"
            ],
            "PRICE_RANGE": [
                "económico", "lujoso", "accesible", "premium", "gama alta",
                "bajo presupuesto", "alto standing", "exclusivo"
            ],
            "CONDITION": [
                "nuevo", "usado", "a estrenar", "reformado", "buen estado",
                "para reformar", "recién construido", "en construcción"
            ],
            "AMENITIES": [
                "aire acondicionado", "calefacción", "amueblado", "cocina equipada",
                "vestidor", "trastero", "bodega", "gimnasio", "seguridad 24h"
            ],
            "LOCATION_TYPE": [
                "céntrico", "urbano", "suburbano", "rural", "playa", "montaña",
                "residencial", "comercial", "industrial"
            ]
        }
        
        # Agregar los patrones a cada modelo de spaCy
        for nlp in self.nlp_models:
            ruler = nlp.get_pipe("entity_ruler") if "entity_ruler" in nlp.pipe_names else nlp.add_pipe("entity_ruler")
            for label, terms in patterns.items():
                ruler.add_patterns([{"label": label, "pattern": term} for term in terms])

    def extract_entities(self, text: str) -> Dict[str, List[Entity]]:
        """Extrae todas las entidades del texto
        
        Args:
            text: Texto para analizar
            
        Returns:
            Diccionario con las entidades encontradas agrupadas por tipo
        """
        entities = {
            "LOC": [], "PRICE": [], "NUM": [], "PROP_TYPE": [],
            "FEATURE": [], "CONDITION": [], "PRICE_RANGE": [],
            "AMENITIES": [], "LOCATION_TYPE": []
        }

        # Extraer entidades geográficas
        geo_entities = self.geo_manager.extract_geo_entities(text)
        entities["LOC"].extend(geo_entities)

        # Procesar el texto con cada modelo de spaCy
        for nlp in self.nlp_models:
            doc = nlp(text.lower())
            self._process_named_entities(doc, entities)
            self._process_numbers(doc, entities)

        return entities

    def _process_named_entities(self, doc: Doc, entities: Dict[str, List[Entity]]):
        """Procesa las entidades nombradas encontradas por spaCy"""
        for ent in doc.ents:
            if ent.label_ in entities:
                context = self._get_entity_context(doc, ent)
                entities[ent.label_].append(
                    Entity(
                        text=ent.text,
                        start=ent.start_char,
                        end=ent.end_char,
                        context=context
                    )
                )

    def _process_numbers(self, doc: Doc, entities: Dict[str, List[Entity]]):
        """Procesa números y valores numéricos en el texto"""
        for token in doc:
            if token.like_num:
                context = self._get_number_context(doc, token)
                if context:
                    entities["NUM"].append(
                        Entity(
                            text=token.text,
                            start=token.idx,
                            end=token.idx + len(token.text),
                            context=context
                        )
                    )

    def _get_entity_context(self, doc: Doc, ent: spacy.tokens.Span) -> EntityContext:
        """Obtiene el contexto de una entidad"""
        preceding = doc[max(0, ent.start-3):ent.start].text
        following = doc[ent.end:min(len(doc), ent.end+3)].text
        
        related_entities = []
        for other_ent in doc.ents:
            if other_ent != ent and abs(other_ent.start - ent.end) < 5:
                related_entities.append({
                    "text": other_ent.text,
                    "label": other_ent.label_,
                    "distance": abs(other_ent.start - ent.end)
                })

        return EntityContext(
            preceding=preceding,
            following=following,
            related_entities=related_entities
        )

    def _get_number_context(self, doc: Doc, token: spacy.tokens.Token) -> Optional[NumberContext]:
        """Determina el contexto de un número"""
        next_token = doc[token.i + 1] if token.i + 1 < len(doc) else None
        prev_token = doc[token.i - 1] if token.i > 0 else None

        context_rules = [
            # Precio
            (lambda: next_token and next_token.text in ["euros", "€", "usd", "$", "dólares"],
             "price", lambda: next_token.text),
            # Área
            (lambda: next_token and next_token.text in ["m2", "metros", "metros cuadrados"],
             "area", lambda: "m2"),
            # Habitaciones
            (lambda: (next_token and next_token.text in ["habitaciones", "dormitorios", "baños"]) or
                    (prev_token and prev_token.text in ["habitación", "dormitorio", "baño"]),
             "rooms", lambda: next_token.text if next_token else prev_token.text)
        ]

        for condition, type_, unit_func in context_rules:
            if condition():
                return NumberContext(type=type_, unit=unit_func())

        return None


class EnhancedQueryBuilder(QueryBuilderInterface):
    """Constructor mejorado de consultas SQL basadas en entidades"""

    BASE_QUERY = """
        SELECT DISTINCT
            cp.id,
            cp.location,
            cp.property_type,
            cp.price,
            cp.square_meters,
            cp.num_bedrooms as avg_bedrooms,
            cp.num_rooms as avg_rooms,
            cp.description,
            cp.image,
            cp.url,
            ROUND(cp.price/cp.square_meters, 2) as price_per_m2,
            cc.name as country_name,
            cd.name as province_name,
            cct.name as city_name,
            cp.project_category,
            cp.project_type,
            cp.residence_type
        FROM chat_property cp
        LEFT JOIN chat_country cc ON cp.country_id = cc.id
        LEFT JOIN chat_province cd ON cp.province_id = cd.id
        LEFT JOIN chat_city cct ON cp.city_id = cct.id
        WHERE 1=1
    """

    def build_query(self, entities: Dict[str, List[Entity]]) -> Tuple[str, List[Any]]:
        """Construye la consulta SQL basada en las entidades extraídas
        
        Args:
            entities: Diccionario de entidades extraídas por tipo
            
        Returns:
            Tupla con la consulta SQL y sus parámetros
        """
        conditions = []
        params = []
        
        self._add_location_conditions(entities.get("LOC", []), conditions, params)
        self._add_price_conditions(entities.get("PRICE", []), conditions, params)
        self._add_numeric_conditions(entities.get("NUM", []), conditions, params)
        self._add_property_type_conditions(entities.get("PROP_TYPE", []), conditions, params)
        self._add_feature_conditions(entities.get("FEATURE", []), conditions, params)
        self._add_condition_conditions(entities.get("CONDITION", []), conditions, params)
        self._add_amenities_conditions(entities.get("AMENITIES", []), conditions, params)
        self._add_location_type_conditions(entities.get("LOCATION_TYPE", []), conditions, params)
        
        # Construir la consulta final
        query = self.BASE_QUERY
        if conditions:
            query += " AND " + " AND ".join(conditions)
        
        # Agregar ordenamiento por relevancia
        query += self._add_ordering(entities)
        
        return query, params

    def _add_location_conditions(self, locations: List[Entity], conditions: List[str], params: List[Any]):
        """Añade condiciones de ubicación"""
        if locations:
            location_conditions = []
            for loc in locations:
                location_conditions.extend([
                    "LOWER(cp.location) LIKE LOWER(?)",
                    "LOWER(cc.name) LIKE LOWER(?)",
                    "LOWER(cd.name) LIKE LOWER(?)",
                    "LOWER(cct.name) LIKE LOWER(?)"
                ])
                params.extend([f"%{loc.text}%"] * 4)
            conditions.append(f"({' OR '.join(location_conditions)})")

    def _add_price_conditions(self, prices: List[Entity], conditions: List[str], params: List[Any]):
        """Añade condiciones de precio"""
        for price in prices:
            if isinstance(price.context, EntityContext):
                amount = self._extract_number(price.text)
                if not amount:
                    continue

                if "menos" in price.context.preceding or "máximo" in price.context.preceding:
                    conditions.append("cp.price <= ?")
                    params.append(amount)
                elif "más" in price.context.preceding or "mínimo" in price.context.preceding:
                    conditions.append("cp.price >= ?")
                    params.append(amount)
                else:
                    # Rango de ±10% si no hay indicador específico
                    conditions.append("cp.price BETWEEN ? AND ?")
                    margin = amount * 0.1
                    params.extend([amount - margin, amount + margin])

    def _add_numeric_conditions(self, numbers: List[Entity], conditions: List[str], params: List[Any]):
        """Añade condiciones basadas en números"""
        for num in numbers:
            if isinstance(num.context, NumberContext):
                if num.context.type == "rooms":
                    conditions.append("cp.num_bedrooms = ?")
                    params.append(int(num.text))
                elif num.context.type == "area":
                    area = float(num.text)
                    conditions.append("cp.square_meters BETWEEN ? AND ?")
                    params.extend([area * 0.9, area * 1.1])  # ±10% de tolerancia

    def _add_property_type_conditions(self, prop_types: List[Entity], conditions: List[str], params: List[Any]):
        """Añade condiciones de tipo de propiedad"""
        if prop_types:
            type_conditions = []
            for prop_type in prop_types:
                type_conditions.append("LOWER(cp.property_type) LIKE LOWER(?)")
                params.append(f"%{prop_type.text}%")
            conditions.append(f"({' OR '.join(type_conditions)})")

    def _add_feature_conditions(self, features: List[Entity], conditions: List[str], params: List[Any]):
        """Añade condiciones de características"""
        for feature in features:
            conditions.append("""
                (LOWER(cp.description) LIKE LOWER(?) 
                OR LOWER(cp.project_category) LIKE LOWER(?)
                OR LOWER(cp.residence_type) LIKE LOWER(?))
            """)
            feature_param = f"%{feature.text}%"
            params.extend([feature_param] * 3)

    def _add_condition_conditions(self, conditions_list: List[Entity], conditions: List[str], params: List[Any]):
        """Añade condiciones de estado de la propiedad"""
        for condition in conditions_list:
            conditions.append("LOWER(cp.description) LIKE LOWER(?)")
            params.append(f"%{condition.text}%")

    def _add_amenities_conditions(self, amenities: List[Entity], conditions: List[str], params: List[Any]):
        """Añade condiciones de amenidades"""
        for amenity in amenities:
            conditions.append("LOWER(cp.description) LIKE LOWER(?)")
            params.append(f"%{amenity.text}%")

    def _add_location_type_conditions(self, location_types: List[Entity], conditions: List[str], params: List[Any]):
        """Añade condiciones de tipo de ubicación"""
        for loc_type in location_types:
            conditions.append("LOWER(cp.description) LIKE LOWER(?)")
            params.append(f"%{loc_type.text}%")

    def _add_ordering(self, entities: Dict[str, List[Entity]]) -> str:
        """Determina el orden óptimo basado en las entidades"""
        if entities.get("PRICE") or any(num.context.type == "price" 
                                      for num in entities.get("NUM", []) 
                                      if isinstance(num.context, NumberContext)):
            return " ORDER BY cp.price ASC"
        
        if any(num.context.type == "area" 
               for num in entities.get("NUM", []) 
               if isinstance(num.context, NumberContext)):
            return " ORDER BY price_per_m2 ASC"
        
        return " ORDER BY cp.price ASC"

    @staticmethod
    def _extract_number(text: str) -> Optional[float]:
        """Extrae un número de un texto"""
        text = text.replace("€", "").replace("$", "").replace(",", "").strip()
        try:
            return float(text)
        except ValueError:
            return None

class RealEstateAnalyzer:
    """Analizador principal de propiedades inmobiliarias con integración de Claude"""

    def __init__(self, db_path: str, nlp_models: List[Language], log_path: str = "logs"):
        """Inicializa el analizador
        
        Args:
            db_path: Ruta a la base de datos SQLite
            nlp_models: Lista de modelos spaCy cargados
            log_path: Directorio para los logs
        """
        # Configuración básica
        self.db_path = db_path
        self._setup_logging(log_path)
        self._validate_and_connect_db()
        
        # Inicializar componentes NER y Claude
        try:
            self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.geo_manager = GeoEntityManager()
            self.entity_extractor = EnhancedEntityExtractor(nlp_models, self.geo_manager)
            self.query_builder = EnhancedQueryBuilder()
            
            # Cache para consultas frecuentes
            self._query_cache = {}
            
        except Exception as e:
            self.logger.error(f"Error inicializando componentes: {str(e)}")
            raise

    def _setup_logging(self, log_path: str):
        """Configura el sistema de logs"""
        os.makedirs(log_path, exist_ok=True)
        log_file = os.path.join(log_path, f'real_estate_{datetime.now().strftime("%Y%m%d")}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("RealEstateAnalyzer")

    def _validate_and_connect_db(self):
        """Valida y establece la conexión a la base de datos"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Base de datos no encontrada en: {self.db_path}")
        
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("PRAGMA foreign_keys = ON")
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_property';")
            
            if not cursor.fetchone():
                raise Exception("La tabla chat_property no existe en la base de datos")
                
            self.logger.info(f"Conectado a la base de datos en {self.db_path}")
            
        except sqlite3.Error as e:
            self.logger.error(f"Error al conectar a la base de datos: {str(e)}")
            raise

    def process_user_query(self, query: str) -> Dict[str, Any]:
        """Procesa una consulta de usuario
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Diccionario con los resultados del análisis
        """
        try:
            # Extraer entidades y construir consulta SQL
            entities = self.entity_extractor.extract_entities(query)
            sql_query, params = self.query_builder.build_query(entities)
            
            # Ejecutar consulta
            df = pd.read_sql_query(sql_query, self.conn, params=params)
            df = self._process_media_urls(df)
            
            # Análisis geográfico
            geo_analysis = self._analyze_geo_distribution(df)
            
            # Generar respuesta con Claude
            claude_response = self._generate_claude_response(query, {
                "results": df.to_dict('records'),
                "geo_analysis": geo_analysis,
                "entities": entities
            })
            
            return {
                "status": "success",
                "entities": entities,
                "results": df.to_dict('records'),
                "geo_analysis": geo_analysis,
                "claude_response": claude_response,
                "query": query
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "query": query
            }

    def _process_media_urls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Procesa y normaliza URLs de imágenes y enlaces"""
        if 'image' in df.columns:
            df['image'] = df['image'].apply(self._normalize_image_url)
        
        if 'url' in df.columns:
            df['url'] = df['url'].apply(self._clean_url)
        
        return df

    @staticmethod
    def _normalize_image_url(image: Optional[str]) -> str:
        """Normaliza las URLs de imágenes"""
        if not image:
            return '/media/default.jpg'
        return f"/media/{image.replace('property_images/', '')}"

    @staticmethod
    def _clean_url(url: Optional[str]) -> str:
        """Limpia y normaliza URLs"""
        if not url:
            return '#'
        url = re.sub(r'\s*(class|target|rel)=[""][^""]*[""]', '', url)
        return url.strip('" ')

    def _analyze_geo_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza la distribución geográfica de los resultados"""
        geo_analysis = {
            "locations": defaultdict(int),
            "countries": defaultdict(int),
            "provinces": defaultdict(int),
            "cities": defaultdict(int),
            "price_ranges": defaultdict(list),
            "property_types": defaultdict(int)
        }

        for _, row in df.iterrows():
            # Análisis geográfico
            if row.get('location'):
                geo_analysis["locations"][row['location']] += 1
            if row.get('country_name'):
                geo_analysis["countries"][row['country_name']] += 1
            if row.get('province_name'):
                geo_analysis["provinces"][row['province_name']] += 1
            if row.get('city_name'):
                geo_analysis["cities"][row['city_name']] += 1

            # Análisis de precios por ubicación
            if row.get('price') and row.get('location'):
                geo_analysis["price_ranges"][row['location']].append(row['price'])

                # Análisis de tipos de propiedad
                if row.get('property_type'):
                    geo_analysis["property_types"][row['property_type']] += 1

        # Calcular estadísticas de precios por ubicación
        price_stats = {}
        for location, prices in geo_analysis["price_ranges"].items():
            price_stats[location] = {
                "min": min(prices),
                "max": max(prices),
                "avg": sum(prices) / len(prices),
                "count": len(prices)
            }
        geo_analysis["price_ranges"] = price_stats

        return dict(geo_analysis)

    def _generate_claude_response(self, query: str, context: Dict[str, Any]) -> str:
        """Genera una respuesta usando Claude
        
        Args:
            query: Consulta del usuario
            context: Contexto con resultados y análisis
            
        Returns:
            Respuesta generada por Claude
        """
        try:
            system_prompt = """
            Eres un experto asistente inmobiliario que ayuda a usuarios a encontrar propiedades.
            Tus respuestas deben:
            1. Ser claras y concisas
            2. Destacar los aspectos más relevantes de las propiedades
            3. Incluir detalles sobre ubicación, precio y características
            4. Hacer recomendaciones personalizadas
            5. Responder en español de manera natural y profesional
            
            Asegúrate de:
            - Mencionar rangos de precios cuando sea relevante
            - Destacar características únicas
            - Sugerir alternativas si es apropiado
            - Ser honesto sobre limitaciones o falta de opciones
            """

            # Preparar contexto para Claude
            properties = context.get("results", [])
            geo_analysis = context.get("geo_analysis", {})
            entities = context.get("entities", {})

            user_prompt = f"""
            Consulta del usuario: {query}

            Propiedades encontradas: {len(properties)}
            
            Detalles de las propiedades:
            {json.dumps(properties, ensure_ascii=False, indent=2)}
            
            Análisis geográfico:
            {json.dumps(geo_analysis, ensure_ascii=False, indent=2)}
            
            Criterios identificados:
            - Ubicaciones: {[e.text for e in entities.get('LOC', [])]}
            - Tipos de propiedad: {[e.text for e in entities.get('PROP_TYPE', [])]}
            - Características: {[e.text for e in entities.get('FEATURE', [])]}
            - Rangos de precio: {[e.text for e in entities.get('PRICE_RANGE', [])]}
            
            Por favor, genera una respuesta natural y útil que responda a la consulta del usuario.
            """

            # response = self.anthropic.messages.create(
            #     model="claude-3-sonnet-20240229",
            #     max_tokens=2000,
            #     temperature=0.7,
            #     system=system_prompt,
            #     messages=[{
            #         "role": "user",
            #         "content": user_prompt
            #     }]
            # )
            response = self.anthropic.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2000,
                temperature=0.7,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": user_prompt
                }]
            )
            
            if not response or not response.content:
                raise ValueError("Respuesta vacía de Claude")

            return response.content[0].text
        except (anthropic.APIError, anthropic.APIConnectionError) as e:
            self.logger.error(f"Error de API de Claude: {str(e)}")
            return self._generate_fallback_response(context)
        except Exception as e:
            self.logger.error(f"Error inesperado generando respuesta: {str(e)}")
            return self._generate_fallback_response(context)

    def _generate_fallback_response(self, context: Dict[str, Any]) -> str:
        """Genera una respuesta de respaldo cuando Claude no está disponible
        
        Args:
            context: Contexto con resultados y análisis
            
        Returns:
            Respuesta generada como fallback
        """
        properties = context.get("results", [])
        if not properties:
            return "Lo siento, no encontré propiedades que coincidan con tus criterios."

        # Construir respuesta básica
        response = f"He encontrado {len(properties)} propiedades que podrían interesarte.\n\n"

        # Agregar rango de precios
        prices = [p['price'] for p in properties if 'price' in p]
        if prices:
            response += f"Rango de precios: USD {min(prices):,.2f} - {max(prices):,.2f}\n"

        # Agregar ubicaciones
        locations = set(p['location'] for p in properties if 'location' in p)
        if locations:
            response += f"Ubicaciones: {', '.join(locations)}\n"

        # Agregar tipos de propiedades
        prop_types = set(p['property_type'] for p in properties if 'property_type' in p)
        if prop_types:
            response += f"Tipos de propiedades disponibles: {', '.join(prop_types)}"

        return response

    def __del__(self):
        """Limpieza al destruir el objeto"""
        if hasattr(self, 'conn') and self.conn:
            try:
                self.conn.close()
                self.logger.info("Conexión a la base de datos cerrada correctamente")
            except Exception as e:
                self.logger.error(f"Error al cerrar la conexión de la base de datos: {str(e)}")
                
def format_currency(amount: float, currency: str = "USD") -> str:
    """Formatea valores monetarios
    
    Args:
        amount: Cantidad a formatear
        currency: Código de moneda
        
    Returns:
        Cadena formateada con el valor monetario
    """
    return f"{currency} {amount:,.2f}"

def calculate_price_per_sqm(price: float, area: float) -> float:
    """Calcula el precio por metro cuadrado
    
    Args:
        price: Precio total
        area: Área en metros cuadrados
        
    Returns:
        Precio por metro cuadrado
    """
    if area <= 0:
        raise ValueError("El área debe ser mayor que 0")
    return price / area

def normalize_location_name(name: str) -> str:
    """Normaliza nombres de ubicaciones
    
    Args:
        name: Nombre de ubicación a normalizar
        
    Returns:
        Nombre normalizado
    """
    name = name.lower().strip()
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    name = re.sub(r'[^a-z0-9\s]', '', name)
    return name.strip()

def get_similar_locations(target: str, locations: List[str], threshold: float = 0.85) -> List[str]:
    """Encuentra ubicaciones similares usando distancia de Levenshtein
    
    Args:
        target: Ubicación objetivo
        locations: Lista de ubicaciones para comparar
        threshold: Umbral de similitud (0-1)
        
    Returns:
        Lista de ubicaciones similares
    """
    normalized_target = normalize_location_name(target)
    similar = []
    
    for loc in locations:
        normalized_loc = normalize_location_name(loc)
        similarity = Levenshtein.ratio(normalized_target, normalized_loc)
        if similarity >= threshold:
            similar.append(loc)
            
    return similar

def parse_area_string(area_str: str) -> Optional[float]:
    """Parsea strings de área a valores numéricos
    
    Args:
        area_str: String con el área (ej: "100 m²", "100m2")
        
    Returns:
        Valor numérico del área o None si no se puede parsear
    """
    match = re.search(r'(\d+(?:\.\d+)?)\s*(?:m²|m2)', area_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def validate_property_data(data: Dict[str, Any]) -> List[str]:
    """Valida datos de una propiedad
    
    Args:
        data: Diccionario con datos de la propiedad
        
    Returns:
        Lista de errores encontrados
    """
    errors = []
    
    # Campos requeridos
    required_fields = ['location', 'price', 'square_meters', 'property_type']
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f"Campo requerido faltante: {field}")
    
    # Validaciones de tipo y rango
    if 'price' in data and data['price'] is not None:
        if not isinstance(data['price'], (int, float)) or data['price'] <= 0:
            errors.append("El precio debe ser un número positivo")
            
    if 'square_meters' in data and data['square_meters'] is not None:
        if not isinstance(data['square_meters'], (int, float)) or data['square_meters'] <= 0:
            errors.append("El área debe ser un número positivo")
    
    return errors

class PropertyDataFormatter:
    """Clase utilitaria para formatear datos de propiedades"""
    
    @staticmethod
    def format_description(desc: str, max_length: int = 200) -> str:
        """Formatea y trunca descripciones largas"""
        if not desc:
            return "Sin descripción disponible"
        
        desc = desc.strip()
        if len(desc) <= max_length:
            return desc
            
        return desc[:max_length].rsplit(' ', 1)[0] + "..."
    
    @staticmethod
    def format_features(features: List[str]) -> str:
        """Formatea lista de características"""
        if not features:
            return "Sin características especificadas"
            
        return ", ".join(features)
    
    @staticmethod
    def format_location_hierarchy(
        city: Optional[str], 
        province: Optional[str], 
        country: Optional[str]
    ) -> str:
        """Formatea jerarquía de ubicación"""
        parts = []
        if city:
            parts.append(city)
        if province:
            parts.append(province)
        if country:
            parts.append(country)
            
        return ", ".join(parts) if parts else "Ubicación no especificada"