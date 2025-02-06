"""
Agente especializado en planificación de tours y recomendaciones personalizadas
utilizando LangGraph para gestionar el flujo de conversación.
"""
from typing import List, Dict, Any, Optional
from langchain.graphs import Neo4jGraph
from pydantic import BaseModel, Field
import langgraph.graph as lg
from langgraph.prebuilt import ToolExecutor
from langgraph.graph import END, StateGraph
from core.rag.retriever import HybridRetriever
from utils.logger import get_logger
import asyncio
import json
from datetime import datetime, timedelta

logger = get_logger(__name__)

class TourPlan(BaseModel):
    """Modelo para plan de tour"""
    days: List[Dict[str, Any]] = Field(default_factory=list)
    overview: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TourAgent:
    def __init__(
        self,
        retriever: HybridRetriever,
        graph: Neo4jGraph,
        config: Optional[Dict[str, Any]] = None
    ):
        self.retriever = retriever
        self.graph = graph
        self.config = config or {}
        self.tool_executor = ToolExecutor()
        self._init_graph()

    def _init_graph(self):
        """Inicializa el grafo de estados para el agente"""
        # Definir estados del agente
        self.state_graph = StateGraph(nodes=[
            "start",
            "analyze_preferences",
            "search_attractions",
            "plan_itinerary",
            "add_recommendations",
            "enrich_content",
            "finalize"
        ])

        # Definir transiciones
        self.state_graph.add_edge("start", "analyze_preferences")
        self.state_graph.add_edge("analyze_preferences", "search_attractions")
        self.state_graph.add_edge("search_attractions", "plan_itinerary")
        self.state_graph.add_edge("plan_itinerary", "add_recommendations")
        self.state_graph.add_edge("add_recommendations", "enrich_content")
        self.state_graph.add_edge("enrich_content", "finalize")
        self.state_graph.add_edge("finalize", END)

        # Configurar funciones de estado
        self.state_graph.add_node_function(
            "analyze_preferences",
            self._analyze_preferences
        )
        self.state_graph.add_node_function(
            "search_attractions",
            self._search_attractions
        )
        self.state_graph.add_node_function(
            "plan_itinerary",
            self._plan_itinerary
        )
        self.state_graph.add_node_function(
            "add_recommendations",
            self._add_recommendations
        )
        self.state_graph.add_node_function(
            "enrich_content",
            self._enrich_content
        )
        self.state_graph.add_node_function(
            "finalize",
            self._finalize_plan
        )

    async def plan_tour(
        self,
        preferences: Dict[str, Any]
    ) -> TourPlan:
        """
        Planifica un tour completo basado en preferencias
        
        Args:
            preferences: Preferencias del usuario
            
        Returns:
            TourPlan con el itinerario completo
        """
        try:
            # Iniciar estado
            state = {
                "preferences": preferences,
                "attractions": [],
                "itinerary": [],
                "recommendations": [],
                "metadata": {
                    "start_time": datetime.now(),
                    "status": "planning"
                }
            }

            # Ejecutar grafo de estados
            final_state = await self.state_graph.arun(state)

            # Construir y retornar plan
            return TourPlan(
                days=final_state["itinerary"],
                overview={
                    "total_attractions": len(final_state["attractions"]),
                    "duration": preferences.get("trip_duration", 1),
                    "locations": list(set(loc["location"] for loc in final_state["attractions"])),
                    "total_cost": sum(attr.get("cost", 0) for attr in final_state["attractions"])
                },
                metadata=final_state["metadata"]
            )

        except Exception as e:
            logger.error(f"Error planning tour: {str(e)}")
            raise

    async def _analyze_preferences(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza y enriquece las preferencias del usuario"""
        try:
            preferences = state["preferences"]
            
            # Validar y normalizar ubicaciones
            if "locations" in preferences:
                cypher_query = """
                MATCH (l:Location)
                WHERE l.name IN $locations
                RETURN l.name as name, l.type as type, l.coordinates as coords
                """
                results = await self.graph.aquery(
                    cypher_query,
                    {"locations": preferences["locations"]}
                )
                
                validated_locations = [
                    {
                        "name": record["name"],
                        "type": record["type"],
                        "coordinates": record["coords"]
                    }
                    for record in results
                ]
                
                preferences["validated_locations"] = validated_locations

            # Enriquecer intereses con categorías relacionadas
            if "interests" in preferences:
                cypher_query = """
                MATCH (i:Interest)-[:RELATED_TO]->(r:Interest)
                WHERE i.name IN $interests
                RETURN DISTINCT r.name as related_interest
                """
                results = await self.graph.aquery(
                    cypher_query,
                    {"interests": preferences["interests"]}
                )
                
                related_interests = [
                    record["related_interest"]
                    for record in results
                ]
                
                preferences["expanded_interests"] = list(set(
                    preferences["interests"] + related_interests
                ))

            # Calcular restricciones temporales
            if "trip_duration" in preferences:
                duration = preferences["trip_duration"]
                preferences["daily_schedule"] = {
                    "activities_per_day": min(5, duration * 2),
                    "max_travel_time": 60 if duration > 2 else 30  # minutos
                }

            state["preferences"] = preferences
            state["metadata"]["preference_analysis_time"] = datetime.now()
            
            return state

        except Exception as e:
            logger.error(f"Error analyzing preferences: {str(e)}")
            raise

    async def _search_attractions(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Busca atracciones basadas en preferencias"""
        try:
            preferences = state["preferences"]
            
            # Construir query para búsqueda
            search_query = {
                "interests": preferences.get("expanded_interests", []),
                "locations": [loc["name"] for loc in preferences.get("validated_locations", [])],
                "budget_per_day": preferences.get("budget_per_day"),
                "max_distance": preferences.get("daily_schedule", {}).get("max_travel_time")
            }

            # Realizar búsqueda híbrida
            results = await self.retriever.hybrid_search(
                query=json.dumps(search_query),
                preferences=preferences,
                k=preferences.get("daily_schedule", {}).get("activities_per_day", 5) * 2
            )

            # Agrupar por tipo de atracción
            attractions_by_type = {}
            for result in results:
                attr_type = result.type
                if attr_type not in attractions_by_type:
                    attractions_by_type[attr_type] = []
                attractions_by_type[attr_type].append(result.dict())

            state["attractions"] = results
            state["attractions_by_type"] = attractions_by_type
            state["metadata"]["search_completion_time"] = datetime.now()

            return state

        except Exception as e:
            logger.error(f"Error searching attractions: {str(e)}")
            raise

    async def _plan_itinerary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Organiza las atracciones en un itinerario diario"""
        try:
            preferences = state["preferences"]
            attractions = state["attractions"]
            duration = preferences.get("trip_duration", 1)

            # Planificar días
            itinerary = []
            remaining_attractions = attractions.copy()
            
            for day in range(duration):
                daily_plan = {
                    "day": day + 1,
                    "activities": [],
                    "total_cost": 0,
                    "total_duration": 0
                }

                # Obtener ubicación inicial para el día
                start_location = None
                if day == 0 and preferences.get("validated_locations"):
                    start_location = preferences["validated_locations"][0]

                # Seleccionar actividades para el día
                current_location = start_location
                while len(daily_plan["activities"]) < preferences.get("daily_schedule", {}).get("activities_per_day", 5):
                    next_activity = self._select_next_activity(
                        remaining_attractions,
                        current_location,
                        daily_plan["total_duration"],
                        preferences
                    )
                    
                    if not next_activity:
                        break

                    daily_plan["activities"].append(next_activity)
                    daily_plan["total_cost"] += next_activity.get("cost", 0)
                    daily_plan["total_duration"] += next_activity.get("duration", 60)
                    current_location = {
                        "name": next_activity["location"],
                        "coordinates": next_activity.get("coordinates")
                    }
                    remaining_attractions.remove(next_activity)

                # Añadir información de transporte entre actividades
                if len(daily_plan["activities"]) > 1:
                    await self._add_transportation_info(daily_plan["activities"])

                itinerary.append(daily_plan)

            state["itinerary"] = itinerary
            state["metadata"]["itinerary_planning_time"] = datetime.now()

            return state

        except Exception as e:
            logger.error(f"Error planning itinerary: {str(e)}")
            raise

    def _select_next_activity(
        self,
        attractions: List[Dict[str, Any]],
        current_location: Optional[Dict[str, Any]],
        current_duration: int,
        preferences: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Selecciona la siguiente mejor actividad basada en ubicación y preferencias"""
        if not attractions:
            return None

        max_daily_duration = preferences.get("daily_schedule", {}).get("max_duration", 480)  # 8 horas por defecto
        if current_duration >= max_daily_duration:
            return None

        # Calcular scores para cada atracción disponible
        scored_attractions = []
        for attraction in attractions:
            score = 0

            # Factor de distancia si hay ubicación actual
            if current_location and "coordinates" in attraction:
                distance = self._calculate_distance(
                    current_location["coordinates"],
                    attraction["coordinates"]
                )
                score -= distance * 0.1  # Penalizar distancias largas

            # Factor de rating
            score += attraction.get("rating", 0) * 0.3

            # Factor de coincidencia con intereses
            if "categories" in attraction and "interests" in preferences:
                matches = len(
                    set(attraction["categories"]) & 
                    set(preferences["interests"])
                )
                score += matches * 0.2

            # Factor de presupuesto
            if "budget_per_day" in preferences:
                if attraction.get("cost", 0) <= preferences["budget_per_day"]:
                    score += 0.2

            scored_attractions.append((attraction, score))

        # Seleccionar la atracción con mejor score
        if scored_attractions:
            scored_attractions.sort(key=lambda x: x[1], reverse=True)
            return scored_attractions[0][0]

        return None

    async def _add_transportation_info(
        self,
        activities: List[Dict[str, Any]]
    ):
        """Añade información de transporte entre actividades"""
        for i in range(len(activities) - 1):
            current = activities[i]
            next_act = activities[i + 1]

            # Buscar rutas en Neo4j
            cypher_query = """
            MATCH (a:Location {name: $from}),
                  (b:Location {name: $to}),
                  path = shortestPath((a)-[r:ROUTE_TO*]-(b))
            RETURN [r IN relationships(path) | {
                type: r.type,
                duration: r.duration,
                cost: r.cost
            }] as route
            """

            try:
                results = await self.graph.aquery(
                    cypher_query,
                    {
                        "from": current["location"],
                        "to": next_act["location"]
                    }
                )

                if results and results[0]["route"]:
                    current["transportation_to_next"] = {
                        "route": results[0]["route"],
                        "total_duration": sum(r["duration"] for r in results[0]["route"]),
                        "total_cost": sum(r["cost"] for r in results[0]["route"])
                    }

            except Exception as e:
                logger.warning(f"Error getting transportation info: {str(e)}")
                current["transportation_to_next"] = {
                    "type": "walking",
                    "estimated_duration": 30,
                    "estimated_cost": 0
                }

    async def _add_recommendations(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Añade recomendaciones personalizadas al itinerario"""
        try:
            itinerary = state["itinerary"]
            preferences = state["preferences"]

            for day in itinerary:
                # Obtener recomendaciones de restaurantes cercanos
                for activity in day["activities"]:
                    nearby_services = await self._find_nearby_services(
                        activity["location"],
                        preferences
                    )
                    activity["nearby_recommendations"] = nearby_services

                # Añadir sugerencias de actividades alternativas
                alternatives = await self._find_alternative_activities(
                    [act["id"] for act in day["activities"]],
                    preferences
                )
                day["alternative_activities"] = alternatives

            state["metadata"]["recommendations_added_time"] = datetime.now()
            return state

        except Exception as e:
            logger.error(f"Error adding recommendations: {str(e)}")
            raise

    async def _find_nearby_services(
        self,
        location: str,
        preferences: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Encuentra servicios cercanos a una ubicación"""
        cypher_query = """
        MATCH (l:Location {name: $location})<-[:LOCATED_IN]-(s:Service)
        WHERE s.type IN $service_types
        RETURN s
        LIMIT 5
        """
        
        service_types = ["restaurant", "cafe", "transport", "shopping"]
        
        try:
            results = await self.graph.aquery(
                cypher_query,
                {
                    "location": location,
                    "service_types": service_types
                }
            )
            
            return [dict(record["s"]) for record in results]
            
        except Exception as e:
            logger.warning(f"Error finding nearby services: {str(e)}")
            return []

    async def _find_alternative_activities(
        self,
        activity_ids: List[str],
        preferences: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Encuentra actividades alternativas"""
        cypher_query = """
        MATCH (a:Activity)
        WHERE a.id IN $activity_ids
        MATCH (a)-[:SIMILAR_TO]->(s:Activity)
        WHERE s.id NOT IN $activity_ids
        RETURN DISTINCT s
        LIMIT 3
        """
        
        try:
            results = await self.graph.aquery(
                cypher_query,
                {"activity_ids": activity_ids}
            )
            
            return [dict(record["s"]) for record in results]
            
        except Exception as e:
            logger.warning(f"Error finding alternative activities: {str(e)}")
            return []

    async def _enrich_content(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enriquece el contenido del itinerario con información adicional"""
        try:
            itinerary = state["itinerary"]

            for day in itinerary:
                for activity in day["activities"]:
                    # Añadir información histórica
                    historical_info = await self._get_historical_info(activity["id"])
                    if historical_info:
                        activity["historical_context"] = historical_info

                    # Añadir tips de otros viajeros
                    traveler_tips = await self._get_traveler_tips(activity["id"])
                    if traveler_tips:
                        activity["traveler_tips"] = traveler_tips

                    # Añadir información de eventos especiales
                    events = await self._get_special_events(
                        activity["location"],
                        state["preferences"].get("travel_dates", {})
                    )
                    if events:
                        activity["special_events"] = events

            state["metadata"]["enrichment_time"] = datetime.now()
            return state

        except Exception as e:
            logger.error(f"Error enriching content: {str(e)}")
            raise

    async def _get_historical_info(self, activity_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información histórica de una actividad"""
        cypher_query = """
        MATCH (a {id: $activity_id})-[:HAS_HISTORY]->(h:History)
        RETURN h
        LIMIT 1
        """
        
        try:
            results = await self.graph.aquery(
                cypher_query,
                {"activity_id": activity_id}
            )
            
            if results:
                return dict(results[0]["h"])
            return None
            
        except Exception as e:
            logger.warning(f"Error getting historical info: {str(e)}")
            return None

    async def _get_traveler_tips(self, activity_id: str) -> List[Dict[str, Any]]:
        """Obtiene tips de otros viajeros"""
        cypher_query = """
        MATCH (a {id: $activity_id})<-[:ABOUT]-(t:Tip)
        WHERE t.rating >= 4
        RETURN t
        ORDER BY t.rating DESC
        LIMIT 3
        """
        
        try:
            results = await self.graph.aquery(
                cypher_query,
                {"activity_id": activity_id}
            )
            
            return [dict(record["t"]) for record in results]
            
        except Exception as e:
            logger.warning(f"Error getting traveler tips: {str(e)}")
            return []

    async def _get_special_events(
        self,
        location: str,
        travel_dates: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Obtiene eventos especiales para una ubicación y fechas"""
        cypher_query = """
        MATCH (l:Location {name: $location})<-[:TAKES_PLACE_IN]-(e:Event)
        WHERE e.start_date <= $end_date AND e.end_date >= $start_date
        RETURN e
        ORDER BY e.start_date
        LIMIT 5
        """
        
        try:
            # Determinar fechas a usar
            if travel_dates:
                start_date = travel_dates.get("start")
                end_date = travel_dates.get("end")
            else:
                # Si no hay fechas, usar próximos 30 días
                start_date = datetime.now().strftime("%Y-%m-%d")
                end_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

            results = await self.graph.aquery(
                cypher_query,
                {
                    "location": location,
                    "start_date": start_date,
                    "end_date": end_date
                }
            )
            
            return [dict(record["e"]) for record in results]
            
        except Exception as e:
            logger.warning(f"Error getting special events: {str(e)}")
            return []

    async def _finalize_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Finaliza el plan y agrega metadatos finales"""
        try:
            # Calcular métricas finales
            state["metadata"].update({
                "end_time": datetime.now(),
                "total_activities": sum(len(day["activities"]) for day in state["itinerary"]),
                "total_cost": sum(day["total_cost"] for day in state["itinerary"]),
                "coverage": len(state["attractions"]) / len(state.get("preferences", {}).get("interests", [])),
                "status": "completed"
            })

            # Calcular duración total del procesamiento
            start_time = state["metadata"]["start_time"]
            end_time = state["metadata"]["end_time"]
            state["metadata"]["processing_duration"] = (end_time - start_time).total_seconds()

            return state

        except Exception as e:
            logger.error(f"Error finalizing plan: {str(e)}")
            state["metadata"]["status"] = "error"
            state["metadata"]["error"] = str(e)
            return state

    def _calculate_distance(
        self,
        coord1: Dict[str, float],
        coord2: Dict[str, float]
    ) -> float:
        """Calcula la distancia entre dos coordenadas"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Radio de la Tierra en km

        lat1 = radians(coord1["lat"])
        lon1 = radians(coord1["lon"])
        lat2 = radians(coord2["lat"])
        lon2 = radians(coord2["lon"])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = R * c

        return distance