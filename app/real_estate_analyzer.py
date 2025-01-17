import sqlite3
import json
import pandas as pd
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Set, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import openai
import spacy
from spacy.tokens import Doc
from spacy.language import Language
import pycountry
import Levenshtein
import unicodedata
import re
from collections import defaultdict

@dataclass
class EntityContext:
    preceding: str
    following: str
    related_entities: List[Dict[str, str]]

@dataclass
class NumberContext:
    type: Optional[str]
    unit: Optional[str]

@dataclass
class Entity:
    text: str
    start: int
    end: int
    context: Union[EntityContext, NumberContext]

@dataclass
class GeoEntity:
    name: str
    type: str
    confidence: float
    normalized_name: str
    alternatives: List[str]
    parent_entity: Optional['GeoEntity'] = None
    metadata: Dict[str, Any] = None

class EntityExtractorInterface(ABC):
    @abstractmethod
    def extract_entities(self, text: str) -> Dict[str, List[Entity]]:
        pass

class QueryBuilderInterface(ABC):
    @abstractmethod
    def build_query(self, entities: Dict[str, List[Entity]]) -> Tuple[str, List[Any]]:
        pass

class GeoEntityManager:
    def __init__(self):
        self.country_names: Dict[str, Set[str]] = defaultdict(set)
        self.city_names: Dict[str, Set[str]] = defaultdict(set)
        self.region_names: Dict[str, Set[str]] = defaultdict(set)
        self._init_geo_data()

    def _init_geo_data(self):
        """Inicializa los datos geográficos desde pycountry"""
        for country in pycountry.countries:
            self._add_country_name(country.name, country.alpha_2)
            if hasattr(country, 'common_name'):
                self._add_country_name(country.common_name, country.alpha_2)
            if hasattr(country, 'official_name'):
                self._add_country_name(country.official_name, country.alpha_2)

    def _normalize_text(self, text: str) -> str:
        """Normaliza el texto para comparaciones"""
        text = text.lower()
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()

    def _add_country_name(self, name: str, country_code: str):
        """Añade un nombre de país y sus variaciones al diccionario"""
        normalized = self._normalize_text(name)
        self.country_names[country_code].add(normalized)
        # Añade variaciones comunes
        variations = self._generate_name_variations(normalized)
        self.country_names[country_code].update(variations)

    @staticmethod
    def _generate_name_variations(name: str) -> Set[str]:
        """Genera variaciones de nombres para mejor coincidencia"""
        variations = {name}
        # Añade versiones sin espacios y con guiones
        variations.add(name.replace(' ', ''))
        variations.add(name.replace(' ', '-'))
        return variations

    def extract_geo_entities(self, text: str) -> List[Entity]:
        """Extrae entidades geográficas del texto"""
        entities = []
        normalized_text = self._normalize_text(text)
        
        # Buscar coincidencias de países
        for country_code, names in self.country_names.items():
            for name in names:
                if name in normalized_text:
                    # Calcular la posición en el texto original
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
    def __init__(self, nlp_models: List[Language], geo_manager: GeoEntityManager):
        self.nlp_models = nlp_models
        self.geo_manager = geo_manager
        self._init_entity_patterns()

    # def _init_entity_patterns(self):
    #     """Inicializa los patrones de entidades para el reconocimiento"""
    #     patterns = {
    #         "PROP_TYPE": [
    #             "casa", "apartamento", "piso", "chalet", "villa", "estudio", 
    #             "duplex", "ático", "local", "oficina", "departamento",
    #             "casa de campo", "penthouse", "loft"
    #         ],
    #         "FEATURE": [
    #             "dormitorios", "habitaciones", "baños", "metros cuadrados",
    #             "m2", "garage", "jardín", "piscina", "terraza", "balcón",
    #             "cochera", "estacionamiento", "ascensor", "seguridad"
    #         ],
    #         "PRICE_RANGE": [
    #             "económico", "lujoso", "accesible", "premium", "gama alta",
    #             "bajo presupuesto", "alto standing", "exclusivo"
    #         ],
    #         "CONDITION": [
    #             "nuevo", "usado", "a estrenar", "reformado", "buen estado",
    #             "para reformar", "recién construido", "en construcción"
    #         ]
    #     }
        
    #     # Agregar los patrones a cada modelo de spaCy
    #     for nlp in self.nlp_models:
    #         ruler = nlp.get_pipe("entity_ruler") if "entity_ruler" in nlp.pipe_names else nlp.add_pipe("entity_ruler")
    #         for label, terms in patterns.items():
    #             patterns = [{"label": label, "pattern": term} for term in terms]
    #             ruler.add_patterns(patterns)

    def _init_entity_patterns(self):
        """Inicializa los patrones de entidades para el reconocimiento"""
        patterns = {  # Cambiado de lista a diccionario
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
            ]
        }
        
        # Agregar los patrones a cada modelo de spaCy
        for nlp in self.nlp_models:
            ruler = nlp.get_pipe("entity_ruler") if "entity_ruler" in nlp.pipe_names else nlp.add_pipe("entity_ruler")
            for label, terms in patterns.items():
                ruler.add_patterns([{"label": label, "pattern": term} for term in terms])
                
    def extract_entities(self, text: str) -> Dict[str, List[Entity]]:
        """Extrae todas las entidades del texto"""
        entities = {
            "LOC": [], "PRICE": [], "NUM": [], "PROP_TYPE": [],
            "FEATURE": [], "CONDITION": [], "PRICE_RANGE": []
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
        
        # Buscar entidades relacionadas en el contexto cercano
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

        # Reglas para detectar el tipo de número
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

        # Aplicar reglas
        for condition, type_, unit_func in context_rules:
            if condition():
                return NumberContext(type=type_, unit=unit_func())

        return None

class EnhancedQueryBuilder(QueryBuilderInterface):
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
        """Construye la consulta SQL basada en las entidades extraídas"""
        conditions = []
        params = []
        
        self._add_location_conditions(entities.get("LOC", []), conditions, params)
        self._add_price_conditions(entities.get("PRICE", []), conditions, params)
        self._add_numeric_conditions(entities.get("NUM", []), conditions, params)
        self._add_property_type_conditions(entities.get("PROP_TYPE", []), conditions, params)
        self._add_feature_conditions(entities.get("FEATURE", []), conditions, params)
        self._add_condition_conditions(entities.get("CONDITION", []), conditions, params)
        
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

    def _add_ordering(self, entities: Dict[str, List[Entity]]) -> str:
        """Determina el orden óptimo basado en las entidades"""
        # Priorizar ordenamiento por precio si hay entidades de precio
        if entities.get("PRICE") or any(num.context.type == "price" 
                                      for num in entities.get("NUM", []) 
                                      if isinstance(num.context, NumberContext)):
            return " ORDER BY cp.price ASC"
        
        # Ordenar por precio por m² si hay menciones de área
        if any(num.context.type == "area" 
               for num in entities.get("NUM", []) 
               if isinstance(num.context, NumberContext)):
            return " ORDER BY price_per_m2 ASC"
        
        # Orden por relevancia como fallback
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
    def __init__(self, db_path: str, nlp_models: List[Language], log_path: str = "logs"):
        # Configuración básica
        self.db_path = db_path
        self._setup_logging(log_path)
        self._validate_and_connect_db()
        
        # Inicializar componentes NER
        try:
            self.geo_manager = GeoEntityManager()
            self.entity_extractor = EnhancedEntityExtractor(nlp_models, self.geo_manager)
            self.query_builder = EnhancedQueryBuilder()
            
            # Cache para consultas frecuentes
            self._query_cache = {}
            
            # Inicializar categorías y consultas
            self._init_query_mappings()
            
        except Exception as e:
            self.logger.error(f"Error inicializando componentes NER: {str(e)}")
            raise

    def _setup_logging(self, log_path: str):
        """Configura el sistema de logs."""
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
        """Valida y establece una conexión a la base de datos."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Base de datos no encontrada en: {self.db_path}")
        
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_property';")
            
            if not cursor.fetchone():
                raise Exception("La tabla chat_property no existe en la base de datos")
                
            self.logger.info(f"Conectado a la base de datos en {self.db_path}")
            
        except sqlite3.Error as e:
            self.logger.error(f"Error al conectar a la base de datos: {str(e)}")
            raise

    def process_user_query(self, query: str) -> Dict[str, Any]:
        """
        Procesa una consulta de usuario con análisis de entidades
        """
        try:
            # Extraer entidades
            entities = self.entity_extractor.extract_entities(query)
            self.logger.info(f"Extracted entities: {entities}")
            
            # Construir y ejecutar consulta SQL
            sql_query, params = self.query_builder.build_query(entities)
            self.logger.debug(f"Generated SQL: {sql_query}")
            self.logger.debug(f"Query parameters: {params}")
            
            # Ejecutar consulta y procesar resultados
            df = pd.read_sql_query(sql_query, self.conn, params=params)
            df = self._process_media_urls(df)
            
            # Análisis geográfico adicional
            geo_analysis = self._analyze_geo_distribution(df)
            
            return {
                "status": "success",
                "entities": entities,
                "results": df.to_dict('records'),
                "geo_analysis": geo_analysis,
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

        # Calcular estadísticas de precios
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

    def _init_query_mappings(self):
        """Inicializa los mapeos de consultas"""
        # Categorías de consultas
        self.categorias = {
            "identificacion_localizacion": {
                "info_basica": """
                    SELECT 
                        cp.id,
                        cp.location,
                        cp.property_type,
                        cp.url,
                        cp.square_meters,
                        cp.description,
                        cp.image,
                        cp.num_bedrooms as avg_bedrooms,
                        cp.num_rooms as avg_rooms,
                        ROUND(cp.price, 2) as price,
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
                    WHERE {condition}
                    ORDER BY cp.price/cp.square_meters ASC
                """,
            },
            "analisis_precio": {
                "comparativa_precios": """
                    SELECT 
                        cp.property_type,
                        cc.name as country_name,
                        cd.name as province_name,
                        COUNT(*) as total_properties,
                        ROUND(AVG(cp.price), 2) as avg_price,
                        ROUND(MIN(cp.price), 2) as min_price,
                        ROUND(MAX(cp.price), 2) as max_price,
                        ROUND(AVG(cp.price/cp.square_meters), 2) as avg_price_per_m2,
                        ROUND(AVG(cp.square_meters), 2) as avg_size
                    FROM chat_property cp
                    LEFT JOIN chat_country cc ON cp.country_id = cc.id
                    LEFT JOIN chat_province cd ON cp.province_id = cd.id
                    GROUP BY cp.property_type, cc.name, cd.name
                    HAVING total_properties > 0
                    ORDER BY avg_price_per_m2 ASC
                """
            },
            "detalles_propiedad": {
                "estadisticas_tipos": """
                    SELECT 
                        cp.property_type,
                        cp.project_category,
                        cp.project_type,
                        cp.residence_type,
                        COUNT(*) as total,
                        ROUND(AVG(cp.square_meters), 2) as avg_size,
                        ROUND(AVG(cp.num_bedrooms), 1) as avg_bedrooms,
                        ROUND(AVG(cp.num_rooms), 1) as avg_rooms,
                        ROUND(AVG(cp.price), 2) as avg_price,
                        ROUND(AVG(cp.price/cp.square_meters), 2) as avg_price_per_m2
                    FROM chat_property cp
                    GROUP BY 
                        cp.property_type,
                        cp.project_category,
                        cp.project_type,
                        cp.residence_type
                    HAVING total > 0
                    ORDER BY avg_price_per_m2 ASC
                """
            }
        }
    
    def generate_gpt_response(self, query: str, analysis_results: Dict[str, Any]) -> str:
        """Genera una respuesta natural usando GPT basada en el análisis"""
        try:
            prompt = self._build_gpt_prompt(query, analysis_results)
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": """Eres un experto en análisis inmobiliario. 
                        Proporciona análisis detallados y relevantes basados en los datos proporcionados.
                        Incluye información sobre ubicaciones, precios, características y tendencias cuando sea relevante."""
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message["content"]
            
        except Exception as e:
            self.logger.error(f"Error generating GPT response: {str(e)}")
            return self._generate_fallback_response(analysis_results)

    def _build_gpt_prompt(self, query: str, analysis_results: Dict[str, Any]) -> str:
        """Construye un prompt detallado para GPT"""
        entities = analysis_results.get("entities", {})
        results = analysis_results.get("results", [])
        geo_analysis = analysis_results.get("geo_analysis", {})
        
        prompt = f"""
        Consulta del usuario: {query}
        
        Entidades detectadas:
        - Ubicaciones: {[e.text for e in entities.get('LOC', [])]}
        - Características: {[e.text for e in entities.get('FEATURE', [])]}
        - Tipo de propiedad: {[e.text for e in entities.get('PROP_TYPE', [])]}
        - Números/Precios: {[e.text for e in entities.get('NUM', [])]}
        
        Resultados encontrados: {len(results)} propiedades
        
        Distribución geográfica:
        {json.dumps(geo_analysis, ensure_ascii=False, indent=2)}
        
        Por favor, proporciona un análisis detallado que incluya:
        1. Resumen de los requisitos del usuario
        2. Análisis de las propiedades encontradas
        3. Información sobre precios y ubicaciones
        4. Recomendaciones relevantes
        """
        
        return prompt

    def _generate_fallback_response(self, analysis_results: Dict[str, Any]) -> str:
        """Genera una respuesta de fallback cuando GPT no está disponible"""
        results = analysis_results.get("results", [])
        if not results:
            return "Lo siento, no encontré propiedades que coincidan con tus criterios."
        
        return f"""
        He encontrado {len(results)} propiedades que podrían interesarte.
        
        Rango de precios: 
        - Mínimo: {min(r['price'] for r in results if 'price' in r)}
        - Máximo: {max(r['price'] for r in results if 'price' in r)}
        
        Ubicaciones principales: {', '.join(set(r['location'] for r in results if 'location' in r))}
        """

    def __del__(self):
        """Limpieza al destruir el objeto"""
        if hasattr(self, 'conn') and self.conn:
            try:
                self.conn.close()
            except Exception as e:
                self.logger.error(f"Error al cerrar la conexión de la base de datos: {str(e)}")
