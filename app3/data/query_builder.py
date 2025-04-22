# app/core/data/query_builder.py
from typing import Dict, List, Any, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

class QueryBuilder:
    """
    Construye consultas estructuradas para la base de datos a partir
    de preferencias del usuario.
    """
    def __init__(self):
        """Inicializa el constructor de consultas"""
        # Mapeo de intereses a tablas relevantes
        self.interest_to_tables = {
            'cultural': ['tourist_spots', 'activities'],
            'naturaleza': ['tourist_spots', 'activities'],
            'gastronomía': ['restaurants'],
            'aventura': ['activities'],
            'actividades_acuáticas': ['activities'],
            'vida_nocturna': ['nightclubs']
        }
        
        # Mapeo de intereses a tipos para filtrado
        self.interest_to_types = {
            'cultural': ['museo', 'histórico', 'cultural', 'monumento', 'arts', 'history', 'architecture'],
            'naturaleza': ['natural', 'parque', 'playa', 'nature', 'beach', 'park'],
            'gastronomía': ['comida', 'gastronómico', 'restaurante', 'food', 'restaurant', 'cuisine'],
            'aventura': ['aventura', 'senderismo', 'trekking', 'adventure', 'hiking', 'outdoor'],
            'actividades_acuáticas': ['buceo', 'snorkel', 'acuático', 'diving', 'snorkel', 'swim'],
            'vida_nocturna': ['club', 'bar', 'discoteca', 'nightlife', 'music', 'party']
        }
    
    def build_query(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construye una consulta estructurada a partir de preferencias del usuario
        
        Args:
            preferences (Dict[str, Any]): Preferencias del usuario
            
        Returns:
            Dict[str, Any]: Consulta estructurada para la base de datos
        """
        try:
            query = {
                "tables": [],
                "filters": {},
                "sort": {
                    "key": "rating",
                    "ascending": False
                },
                "limit": 15  # Valor predeterminado
            }
            
            # Determinar tablas a consultar según intereses
            if preferences.get('interests'):
                tables = self._get_tables_from_interests(preferences['interests'])
                query["tables"] = tables
            else:
                # Si no hay intereses, consultar todas las tablas principales
                query["tables"] = ['tourist_spots', 'activities', 'restaurants']
            
            # Añadir filtro de ubicaciones
            if preferences.get('locations'):
                query["filters"]["locations"] = preferences['locations']
            
            # Añadir filtro de presupuesto
            if preferences.get('budget'):
                query["filters"]["budget"] = preferences['budget']
            
            # Añadir filtro de intereses
            if preferences.get('interests'):
                query["filters"]["interests"] = preferences['interests']
            
            # Añadir filtro de calificación mínima (predeterminado a 3.5)
            query["filters"]["min_rating"] = 3.5
            
            # Ajustar límite basado en duración
            if preferences.get('duration'):
                # Aproximadamente 5 recomendaciones por día
                duration = int(preferences['duration'])
                query["limit"] = min(30, duration * 5)
                
            # Considerar si hay niños
            if preferences.get('with_children'):
                query["filters"]["family_friendly"] = True
                
            logger.info(f"Consulta construida: {query}")
            return query
            
        except Exception as e:
            logger.error(f"Error construyendo consulta: {e}")
            # Devolver consulta predeterminada si hay error
            return {
                "tables": ['tourist_spots', 'activities', 'restaurants'],
                "filters": {
                    "min_rating": 3.5
                },
                "sort": {
                    "key": "rating",
                    "ascending": False
                },
                "limit": 15
            }
    
    def _get_tables_from_interests(self, interests: List[str]) -> List[str]:
        """
        Determina qué tablas consultar basándose en los intereses
        
        Args:
            interests (List[str]): Intereses del usuario
            
        Returns:
            List[str]: Lista de tablas a consultar
        """
        tables = set()
        
        for interest in interests:
            if interest in self.interest_to_tables:
                for table in self.interest_to_tables[interest]:
                    tables.add(table)
        
        # Si no hay tablas específicas, incluir las más generales
        if not tables:
            tables = {'tourist_spots', 'activities'}
            
        return list(tables)
    
    def build_nl_to_sql_query(self, nl_query: str) -> str:
        """
        Convierte una consulta en lenguaje natural a SQL para acceso directo a la base de datos.
        Esto es un ejemplo simplificado para el paradigma RAG.
        
        Args:
            nl_query (str): Consulta en lenguaje natural
            
        Returns:
            str: Consulta SQL generada
        """
        # Este es un ejemplo simplificado. En un sistema real, esto sería generado
        # por un modelo de lenguaje con ejemplos few-shot o fine-tuning específico.
        
        # Consultas predefinidas basadas en patrones comunes
        nl_query = nl_query.lower()
        
        # Ejemplo: consulta de restaurantes
        if "restaurantes" in nl_query or "comida" in nl_query or "restaurants" in nl_query:
            sql = """
            SELECT name, cuisine_type, location, average_person_expense, rating
            FROM restaurants
            WHERE rating >= 4.0
            ORDER BY rating DESC
            LIMIT 10
            """
            
        # Ejemplo: consulta de playas
        elif "playa" in nl_query or "playas" in nl_query or "beach" in nl_query:
            sql = """
            SELECT name, location, description, rating
            FROM tourist_spots
            WHERE type LIKE '%playa%' OR type LIKE '%beach%'
            ORDER BY rating DESC
            LIMIT 10
            """
            
        # Ejemplo: consulta de museos/cultura
        elif "museo" in nl_query or "cultural" in nl_query or "museum" in nl_query:
            sql = """
            SELECT name, location, entry_fee, rating, description
            FROM tourist_spots
            WHERE type LIKE '%museo%' OR type LIKE '%museum%' OR type LIKE '%cultural%'
            ORDER BY rating DESC
            LIMIT 10
            """
            
        # Ejemplo: consulta de actividades acuáticas
        elif "buceo" in nl_query or "snorkel" in nl_query or "diving" in nl_query:
            sql = """
            SELECT name, location, cost, duration_hours, rating
            FROM activities
            WHERE type LIKE '%buceo%' OR type LIKE '%diving%' OR type LIKE '%snorkel%'
            ORDER BY rating DESC
            LIMIT 10
            """
            
        # Consulta predeterminada para turismo
        else:
            sql = """
            SELECT name, type, location, rating, description
            FROM tourist_spots
            WHERE rating >= 4.0
            ORDER BY rating DESC
            LIMIT 10
            """
            
        return sql