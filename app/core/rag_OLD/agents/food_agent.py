"""
Agente especializado en recomendaciones gastronómicas y experiencias culinarias.
"""
from typing import List, Dict, Any, Optional
from langchain.graphs import Neo4jGraph
from pydantic import BaseModel
from datetime import datetime, time
import langgraph.graph as lg
from langgraph.prebuilt import ToolExecutor
from utils.logger import get_logger

logger = get_logger(__name__)

class DiningRecommendation(BaseModel):
    """Modelo para recomendaciones gastronómicas"""
    id: str
    name: str
    type: str = "restaurant"  # restaurant, cafe, bar, etc.
    cuisine: List[str]
    price_range: str
    location: str
    rating: float
    specialties: List[str] = []
    opening_hours: Dict[str, Any]
    dress_code: Optional[str]
    reservations_required: bool = False

class FoodAgent:
    def __init__(
        self,
        graph: Neo4jGraph,
        config: Optional[Dict[str, Any]] = None
    ):
        self.graph = graph
        self.config = config or {}
        self.tool_executor = ToolExecutor()

    async def get_restaurants(
        self,
        preferences: Dict[str, Any],
        location: Optional[str] = None,
        meal_time: Optional[str] = None
    ) -> List[DiningRecommendation]:
        """
        Obtiene recomendaciones de restaurantes
        
        Args:
            preferences: Preferencias del usuario
            location: Ubicación específica
            meal_time: Momento del día (breakfast, lunch, dinner)
        """
        try:
            conditions = []
            params = {
                "price_ranges": self._get_price_ranges(
                    preferences.get("budget_per_day", 0)
                )
            }

            if location:
                conditions.append("r.location = $location")
                params["location"] = location

            if preferences.get("cuisine_preferences"):
                conditions.append(
                    "ANY(cuisine IN r.cuisine WHERE cuisine IN $cuisines)"
                )
                params["cuisines"] = preferences["cuisine_preferences"]

            if meal_time:
                conditions.append("$meal_time IN r.meal_times")
                params["meal_time"] = meal_time

            cypher_query = f"""
            MATCH (r:Restaurant)
            WHERE r.price_range IN $price_ranges
            {' AND ' + ' AND '.join(conditions) if conditions else ''}
            RETURN r
            ORDER BY r.rating DESC
            LIMIT 10
            """

            results = await self.graph.aquery(cypher_query, params)
            
            return [
                DiningRecommendation(**self._process_restaurant(record["r"]))
                for record in results
            ]

        except Exception as e:
            logger.error(f"Error getting nearby restaurants: {str(e)}")
            return []

    def _get_price_ranges(self, budget_per_day: float) -> List[str]:
        """Determina rangos de precio basados en presupuesto"""
        if budget_per_day <= 30:
            return ["$"]
        elif budget_per_day <= 60:
            return ["$", "$"]
        elif budget_per_day <= 100:
            return ["$", "$", "$$"]
        else:
            return ["$", "$", "$$", "$$"]

    async def _get_restaurant_schedule(self, restaurant_id: str) -> Dict[str, Any]:
        """Obtiene horario de un restaurante"""
        cypher_query = """
        MATCH (r:Restaurant {id: $restaurant_id})-[:HAS_SCHEDULE]->(s:Schedule)
        RETURN s
        """

        try:
            results = await self.graph.aquery(
                cypher_query,
                {"restaurant_id": restaurant_id}
            )
            
            if not results:
                return {}
                
            return dict(results[0]["s"])

        except Exception as e:
            logger.error(f"Error getting restaurant schedule: {str(e)}")
            return {}

    def _is_restaurant_open(
        self,
        schedule: Dict[str, Any],
        datetime_obj: datetime
    ) -> bool:
        """Verifica si el restaurante está abierto"""
        if not schedule:
            return False

        weekday = datetime_obj.strftime("%A").lower()
        if weekday not in schedule:
            return False

        current_time = datetime_obj.time()
        day_schedule = schedule[weekday]

        for period in day_schedule:
            open_time = datetime.strptime(period["open"], "%H:%M").time()
            close_time = datetime.strptime(period["close"], "%H:%M").time()
            
            if open_time <= current_time <= close_time:
                return True

        return False

    async def _find_next_availability(
        self,
        restaurant_id: str,
        datetime_obj: datetime,
        party_size: int
    ) -> Optional[datetime]:
        """Encuentra próxima disponibilidad"""
        # Buscar en los próximos 7 días
        for i in range(1, 8):
            next_date = datetime_obj + timedelta(days=i)
            
            # Verificar cada hora de operación
            schedule = await self._get_restaurant_schedule(restaurant_id)
            weekday = next_date.strftime("%A").lower()
            
            if weekday in schedule:
                for period in schedule[weekday]:
                    open_time = datetime.strptime(period["open"], "%H:%M").time()
                    check_datetime = datetime.combine(next_date.date(), open_time)
                    
                    availability = await self.check_restaurant_availability(
                        restaurant_id,
                        check_datetime,
                        party_size
                    )
                    
                    if availability.get("available"):
                        return check_datetime

        return None

    def _process_restaurant(self, restaurant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa datos de restaurante del grafo"""
        restaurant = dict(restaurant_data)
        
        # Asegurar campos requeridos
        restaurant.setdefault("id", str(restaurant.get("_id", "unknown")))
        restaurant.setdefault("name", "Unnamed Restaurant")
        restaurant.setdefault("type", "restaurant")
        restaurant.setdefault("cuisine", [])
        restaurant.setdefault("price_range", "$")
        restaurant.setdefault("rating", 0.0)
        restaurant.setdefault("specialties", [])
        
        # Procesar horarios
        restaurant["opening_hours"] = self._process_hours(
            restaurant.get("hours", {})
        )
        
        return restaurant

    def _process_hours(self, hours_data: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa horarios de operación"""
        processed_hours = {}
        
        for day, periods in hours_data.items():
            if isinstance(periods, list):
                processed_hours[day] = periods
            elif isinstance(periods, dict):
                processed_hours[day] = [periods]
            else:
                processed_hours[day] = []
                
        return processed_hours

            ]

        except Exception as e:
            logger.error(f"Error getting restaurants: {str(e)}")
            return []

    async def get_local_specialties(
        self,
        location: str
    ) -> List[Dict[str, Any]]:
        """Obtiene especialidades locales de una ubicación"""
        cypher_query = """
        MATCH (l:Location {name: $location})<-[:ORIGINATED_IN]-(d:Dish)
        WHERE d.is_specialty = true
        RETURN d
        ORDER BY d.popularity DESC
        """

        try:
            results = await self.graph.aquery(
                cypher_query,
                {"location": location}
            )
            
            return [dict(record["d"]) for record in results]

        except Exception as e:
            logger.error(f"Error getting local specialties: {str(e)}")
            return []

    async def get_food_experiences(
        self,
        preferences: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Obtiene experiencias culinarias (clases, tours, etc.)"""
        cypher_query = """
        MATCH (e:Experience)
        WHERE e.type IN ['cooking_class', 'food_tour', 'tasting']
        AND e.cost <= $max_cost
        RETURN e
        ORDER BY e.rating DESC
        LIMIT 5
        """

        try:
            results = await self.graph.aquery(
                cypher_query,
                {"max_cost": preferences.get("budget_per_day", 1000)}
            )
            
            return [dict(record["e"]) for record in results]

        except Exception as e:
            logger.error(f"Error getting food experiences: {str(e)}")
            return []

    async def check_restaurant_availability(
        self,
        restaurant_id: str,
        datetime_obj: datetime,
        party_size: int
    ) -> Dict[str, Any]:
        """Verifica disponibilidad en un restaurante"""
        try:
            # Primero verificar horario de apertura
            schedule = await self._get_restaurant_schedule(restaurant_id)
            if not self._is_restaurant_open(schedule, datetime_obj):
                return {
                    "available": False,
                    "reason": "Restaurant is closed at this time"
                }

            # Luego verificar reservaciones
            cypher_query = """
            MATCH (r:Restaurant {id: $restaurant_id})
            OPTIONAL MATCH (r)-[:HAS_RESERVATION]->(b:Booking)
            WHERE b.date = date($date)
            AND b.time = time($time)
            RETURN r.capacity as capacity,
                   sum(b.party_size) as booked
            """

            results = await self.graph.aquery(
                cypher_query,
                {
                    "restaurant_id": restaurant_id,
                    "date": datetime_obj.strftime("%Y-%m-%d"),
                    "time": datetime_obj.strftime("%H:%M")
                }
            )

            if not results:
                return {"available": False, "reason": "Restaurant not found"}

            record = results[0]
            capacity = record["capacity"]
            booked = record["booked"] or 0
            available = capacity - booked

            return {
                "available": available >= party_size,
                "party_size": party_size,
                "available_seats": available,
                "next_available": await self._find_next_availability(
                    restaurant_id,
                    datetime_obj,
                    party_size
                ) if available < party_size else None
            }

        except Exception as e:
            logger.error(f"Error checking restaurant availability: {str(e)}")
            return {"available": False, "reason": str(e)}

    async def get_nearby_restaurants(
        self,
        location: str,
        max_distance: float = 1.0  # km
    ) -> List[Dict[str, Any]]:
        """Encuentra restaurantes cercanos a una ubicación"""
        cypher_query = """
        MATCH (l:Location {name: $location})<-[:LOCATED_IN]-(r:Restaurant)
        WHERE r.distance <= $max_distance
        RETURN r
        ORDER BY r.distance
        LIMIT 10
        """

        try:
            results = await self.graph.aquery(
                cypher_query,
                {
                    "location": location,
                    "max_distance": max_distance
                }
            )
            
            return self._process_restaurant(record["r"])