import openai
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import sqlite3
import logging
import json
from datetime import datetime
import spacy
from collections import defaultdict

@dataclass
class EntityMatch:
    text: str
    confidence: float
    db_match: Optional[Dict[str, Any]] = None
    alternatives: List[str] = None

class EnhancedNER:
    def __init__(self, db_path: str, openai_api_key: str):
        """
        Inicializa el sistema NER mejorado.
        
        Args:
            db_path: Ruta a la base de datos SQLite
            openai_api_key: API key de OpenAI
        """
        self.db_path = db_path
        openai.api_key = openai_api_key
        
        # Inicializar spaCy
        self.nlp = spacy.load("es_core_news_md")
        
        # Configurar logging
        self.setup_logging()
        
        # Inicializar cache
        self._cache = {}
        
        # Cargar datos de la DB para validación rápida
        self._load_db_data()

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
                
        except sqlite3.Error as e:
            self.logger.error(f"Error cargando datos de DB: {e}")
            raise

    def _get_gpt_prompt(self, text: str) -> str:
        """
        Genera el prompt para GPT-4 con ejemplos few-shot.
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
        
        Texto a analizar: {text}
        """

    def _query_gpt(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Consulta a GPT-4 para extraer entidades.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self._get_gpt_prompt(text)}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            self.logger.error(f"Error consultando GPT: {e}")
            return {"locations": [], "properties": []}

    def _validate_with_spacy(self, text: str, gpt_entities: Dict[str, List[Dict[str, Any]]]):
        """
        Valida y complementa las entidades de GPT con spaCy.
        """
        doc = self.nlp(text)
        
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

    def _merge_entities(self, gpt_ents: List[Dict], spacy_ents: List[Dict]) -> List[Dict]:
        """
        Combina y deduplica entidades de GPT y spaCy.
        """
        merged = []
        seen = set()
        
        for ent in gpt_ents + spacy_ents:
            text = ent["text"].lower()
            if text not in seen:
                seen.add(text)
                merged.append(ent)
                
        return merged

    def _validate_with_db(self, entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[EntityMatch]]:
        """
        Valida las entidades contra la base de datos.
        """
        validated = {
            "locations": [],
            "properties": []
        }
        
        # Validar ubicaciones
        for loc in entities["locations"]:
            match = self._find_best_location_match(loc["text"])
            if match:
                validated["locations"].append(EntityMatch(
                    text=loc["text"],
                    confidence=loc["confidence"],
                    db_match=match
                ))
        
        # Validar propiedades
        for prop in entities["properties"]:
            match = self._find_property_match(prop)
            if match:
                validated["properties"].append(EntityMatch(
                    text=prop["text"],
                    confidence=prop["confidence"],
                    db_match=match
                ))
                
        return validated

    def _find_best_location_match(self, location: str) -> Optional[Dict[str, Any]]:
        """
        Encuentra la mejor coincidencia para una ubicación en la DB.
        """
        # Normalizar texto
        location = location.lower().strip()
        
        # Buscar en países
        for id_, name in self.countries.items():
            if location == name.lower():
                return {"type": "country", "id": id_, "name": name}
        
        # Buscar en provincias
        for id_, data in self.provinces.items():
            if location == data["name"].lower():
                return {
                    "type": "province",
                    "id": id_,
                    "name": data["name"],
                    "country_id": data["country_id"]
                }
        
        # Buscar en ciudades
        for id_, data in self.cities.items():
            if location == data["name"].lower():
                return {
                    "type": "city",
                    "id": id_,
                    "name": data["name"],
                    "province_id": data["province_id"]
                }
        
        return None

    def _find_property_match(self, property_entity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Encuentra coincidencias de propiedades en la DB.
        """
        prop_type = property_entity["type"]
        text = property_entity["text"]
        
        if prop_type == "type":
            # Buscar coincidencias de tipo de propiedad
            matches = [
                prop for prop in self.properties.values()
                if text.lower() in prop["type"].lower()
            ]
            if matches:
                return {
                    "type": "property_type",
                    "matches": matches
                }
        
        elif prop_type == "price":
            # Validar rango de precios
            try:
                price = float(text.replace("$", "").replace(",", ""))
                matches = [
                    prop for prop in self.properties.values()
                    if prop["price_range"][0] <= price <= prop["price_range"][1]
                ]
                if matches:
                    return {
                        "type": "price_range",
                        "matches": matches
                    }
            except ValueError:
                pass
        
        return None

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analiza un texto para extraer y validar entidades.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Dict con entidades validadas y coincidencias en DB
        """
        try:
            # 1. Obtener entidades de GPT
            gpt_entities = self._query_gpt(text)
            
            # 2. Validar con spaCy
            combined_entities = self._validate_with_spacy(text, gpt_entities)
            
            # 3. Validar contra DB
            validated_entities = self._validate_with_db(combined_entities)
            
            # 4. Generar resultado final
            result = {
                "status": "success",
                "entities": validated_entities,
                "query_text": text,
                "timestamp": datetime.now().isoformat()
            }
            
            # Logging
            self.logger.info(f"Analysis completed successfully for text: {text[:100]}...")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing text: {e}")
            return {
                "status": "error",
                "error": str(e),
                "query_text": text,
                "timestamp": datetime.now().isoformat()
            }

    def get_query_suggestions(self, text: str) -> List[str]:
        """
        Genera sugerencias de búsqueda basadas en el análisis de entidades.
        """
        try:
            analysis = self.analyze_text(text)
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
            
            return suggestions[:5]  # Limitar a 5 sugerencias
            
        except Exception as e:
            self.logger.error(f"Error generating suggestions: {e}")
            return []