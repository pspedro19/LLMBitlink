"""
Agente especializado en actividades y experiencias turísticas.
"""
from typing import List, Dict, Any, Optional
from langchain.graphs import Neo4jGraph
from pydantic import BaseModel
from datetime import datetime, timedelta
import langgraph.graph as lg
from langgraph.prebuilt import ToolExecutor
from utils.logger import get_logger

logger = get_logger(__name__)

class ActivityRecommendation(BaseModel):
    """Modelo para recomendaciones de actividades"""
    id: str
    name: str
    type: str
    description: str
    duration: int  # en minutos
    cost: float
    difficulty: Optional[str]
    requirements: List[str] = []
    best_time: Optional[str]
    location: str
    rating: float
    categories: List[str] = []

class ActivityAgent:
    def __init__(
        self,
        graph: Neo4jGraph,
        config: Optional[Dict[str, Any]] = None
    ):
        self.graph = graph
        self.config = config or {}
        self.tool_executor = ToolExecutor()

    async def get_activities(
        self,
        preferences: Dict[str, Any],
        location: Optional[str] = None,
        max_duration: Optional[int] = None
    ) -> List[ActivityRecommendation]:
        """
        Obtiene actividades basadas en preferencias del usuario
        
        Args:
            preferences: Preferencias del usuario
            location: Ubicación específica
            max_duration: Duración máxima en minutos
        """
        try:
            # Construir query dinámica
            conditions = []
            params = {}

            if location:
                conditions.append("a.location = $location")
                params["location"] = location

            if max_duration:
                conditions.append("a.duration <= $max_duration")
                params["max_duration"] = max_duration

            if preferences.get("difficulty"):
                conditions.append("a.difficulty = $difficulty")
                params["difficulty"] = preferences["difficulty"]

            if preferences.get("categories"):
                conditions.append("ANY(cat IN a.categories WHERE cat IN $categories)")
                params["categories"] = preferences["categories"]

            if preferences.get("budget_per_day"):
                conditions.append("a.cost <= $max_cost")
                params["max_cost"] = preferences["budget_per_day"]

            # Construir query base
            cypher_query = """
            MATCH (a:Activity)
            WHERE {conditions}
            RETURN a
            ORDER BY a.rating DESC
            LIMIT 10
            """.format(conditions=" AND ".join(conditions) if conditions else "true")

            # Ejecutar consulta
            results = await self.graph.aquery(cypher_query, params)
            
            return [
                ActivityRecommendation(**self._process_activity(record["a"]))
                for record in results
            ]

        except Exception as e:
            logger.error(f"Error getting activities: {str(e)}")
            return []

    async def get_similar_activities(
        self,
        activity_id: str,
        limit: int = 3
    ) -> List[ActivityRecommendation]:
        """Encuentra actividades similares"""
        cypher_query = """
        MATCH (a:Activity {id: $activity_id})-[:SIMILAR_TO]->(s:Activity)
        RETURN s
        LIMIT $limit
        """

        try:
            results = await self.graph.aquery(
                cypher_query,
                {"activity_id": activity_id, "limit": limit}
            )
            
            return [
                ActivityRecommendation(**self._process_activity(record["s"]))
                for record in results
            ]

        except Exception as e:
            logger.error(f"Error getting similar activities: {str(e)}")
            return []

    async def get_activity_details(
        self,
        activity_id: str
    ) -> Optional[Dict[str, Any]]:
        """Obtiene detalles completos de una actividad"""
        cypher_query = """
        MATCH (a:Activity {id: $activity_id})
        OPTIONAL MATCH (a)-[:HAS_REQUIREMENT]->(r:Requirement)
        OPTIONAL MATCH (a)-[:BEST_TIME]->(t:TimeRecommendation)
        OPTIONAL MATCH (a)-[:LOCATED_AT]->(l:Location)
        RETURN a, 
               collect(DISTINCT r) as requirements,
               collect(DISTINCT t) as time_recommendations,
               l
        """

        try:
            results = await self.graph.aquery(
                cypher_query,
                {"activity_id": activity_id}
            )
            
            if not results:
                return None

            record = results[0]
            activity = self._process_activity(record["a"])
            
            # Añadir información adicional
            activity.update({
                "requirements": [dict(r) for r in record["requirements"]],
                "time_recommendations": [dict(t) for t in record["time_recommendations"]],
                "location_details": dict(record["l"]) if record["l"] else None
            })

            return activity

        except Exception as e:
            logger.error(f"Error getting activity details: {str(e)}")
            return None

    async def check_availability(
        self,
        activity_id: str,
        date: datetime
    ) -> Dict[str, Any]:
        """Verifica disponibilidad para una fecha"""
        cypher_query = """
        MATCH (a:Activity {id: $activity_id})
        OPTIONAL MATCH (a)-[:HAS_SCHEDULE]->(s:Schedule)
        WHERE s.date = date($date)
        RETURN a.max_capacity as capacity,
               s.booked as booked,
               s.available_slots as available_slots
        """

        try:
            results = await self.graph.aquery(
                cypher_query,
                {
                    "activity_id": activity_id,
                    "date": date.strftime("%Y-%m-%d")
                }
            )
            
            if not results:
                return {"available": False, "reason": "No schedule found"}

            record = results[0]
            capacity = record["capacity"]
            booked = record["booked"] or 0
            available_slots = record["available_slots"] or (capacity - booked)

            return {
                "available": available_slots > 0,
                "total_capacity": capacity,
                "booked": booked,
                "available_slots": available_slots
            }

        except Exception as e:
            logger.error(f"Error checking availability: {str(e)}")
            return {"available": False, "reason": str(e)}

    def _process_activity(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa datos de actividad del grafo"""
        # Convertir datos del grafo a diccionario
        activity = dict(activity_data)
        
        # Asegurar campos requeridos
        activity.setdefault("id", str(activity.get("_id", "unknown")))
        activity.setdefault("name", "Unnamed Activity")
        activity.setdefault("type", "general")
        activity.setdefault("description", "")
        activity.setdefault("duration", 60)
        activity.setdefault("cost", 0.0)
        activity.setdefault("rating", 0.0)
        activity.setdefault("categories", [])
        
        # Limpiar datos
        if isinstance(activity.get("categories"), str):
            activity["categories"] = activity["categories"].split(",")
            
        return activity