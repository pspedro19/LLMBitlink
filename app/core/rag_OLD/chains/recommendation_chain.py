"""
Cadena de recomendación que coordina múltiples agentes para generar recomendaciones
personalizadas.
"""
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import langgraph.graph as lg
from langgraph.prebuilt import ToolExecutor
from core.rag.agents import TourAgent, FoodAgent, ActivityAgent
from utils.logger import get_logger
from datetime import datetime

logger = get_logger(__name__)

class RecommendationRequest(BaseModel):
    """Modelo para solicitud de recomendación"""
    query: str
    preferences: Dict[str, Any]
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)

class RecommendationResult(BaseModel):
    """Modelo para resultado de recomendación"""
    recommendations: List[Dict[str, Any]]
    conversation_flow: Dict[str, Any]
    detected_intents: List[str]
    detected_locations: List[str]
    metrics: Dict[str, Any]

class RecommendationChain:
    def __init__(
        self,
        tour_agent: TourAgent,
        food_agent: FoodAgent,
        activity_agent: ActivityAgent,
        config: Optional[Dict[str, Any]] = None
    ):
        self.tour_agent = tour_agent
        self.food_agent = food_agent
        self.activity_agent = activity_agent
        self.config = config or {}
        self.tool_executor = ToolExecutor()
        self._init_graph()

    def _init_graph(self):
        """Inicializa el grafo de procesamiento de recomendaciones"""
        self.nodes = [
            "start",
            "analyze_query",
            "extract_preferences",
            "get_tour_recommendations",
            "get_food_recommendations",
            "get_activity_recommendations",
            "merge_recommendations",
            "rank_recommendations",
            "enrich_content",
            "finalize"
        ]

        self.graph = lg.Graph()
        self._add_nodes()
        self._add_edges()

    def _add_nodes(self):
        """Añade nodos al grafo con sus funciones correspondientes"""
        for node in self.nodes:
            self.graph.add_node(
                node,
                getattr(self, f"_process_{node}")
            )

    def _add_edges(self):
        """Configura las conexiones entre nodos del grafo"""
        # Flujo principal
        self.graph.add_edge("start", "analyze_query")
        self.graph.add_edge("analyze_query", "extract_preferences")

        # Procesamiento paralelo de recomendaciones
        self.graph.add_parallel_edges(
            "extract_preferences",
            [
                "get_tour_recommendations",
                "get_food_recommendations",
                "get_activity_recommendations"
            ]
        )

        # Convergencia y finalización
        for node in [
            "get_tour_recommendations",
            "get_food_recommendations",
            "get_activity_recommendations"
        ]:
            self.graph.add_edge(node, "merge_recommendations")

        self.graph.add_edge("merge_recommendations", "rank_recommendations")
        self.graph.add_edge("rank_recommendations", "enrich_content")
        self.graph.add_edge("enrich_content", "finalize")

    async def run(
        self,
        query: str,
        preferences: Dict[str, Any],
        conversation_flow: Optional[Dict[str, Any]] = None
    ) -> RecommendationResult:
        """
        Ejecuta la cadena de recomendación
        
        Args:
            query: Consulta del usuario
            preferences: Preferencias del usuario
            conversation_flow: Flujo de conversación opcional
            
        Returns:
            Resultado con recomendaciones y metadatos
        """
        try:
            # Estado inicial
            state = {
                "query": query,
                "preferences": preferences,
                "conversation_flow": conversation_flow or {},
                "start_time": datetime.now(),
                "recommendations": [],
                "metadata": {}
            }

            # Ejecutar grafo
            final_state = await self.graph.arun(state)

            # Construir resultado
            return RecommendationResult(
                recommendations=final_state["recommendations"],
                conversation_flow=final_state["conversation_flow"],
                detected_intents=final_state["metadata"].get("detected_intents", []),
                detected_locations=final_state["metadata"].get("detected_locations", []),
                metrics=self._calculate_metrics(final_state)
            )

        except Exception as e:
            logger.error(f"Error running recommendation chain: {str(e)}")
            raise

    async def _process_start(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Inicializa el procesamiento"""
        state["metadata"]["processing_stages"] = []
        state["metadata"]["processing_stages"].append({
            "stage": "start",
            "timestamp": datetime.now()
        })
        return state

    async def _process_analyze_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza la consulta del usuario"""
        try:
            query = state["query"]

            # Análisis de intenciones
            intents = await self._analyze_intents(query)
            state["metadata"]["detected_intents"] = intents

            # Extracción de ubicaciones
            locations = await self._extract_locations(query)
            state["metadata"]["detected_locations"] = locations

            # Actualizar estado
            state["metadata"]["processing_stages"].append({
                "stage": "analyze_query",
                "timestamp": datetime.now(),
                "results": {
                    "intents": intents,
                    "locations": locations
                }
            })

            return state

        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            state["metadata"]["errors"] = state.get("metadata", {}).get("errors", [])
            state["metadata"]["errors"].append(str(e))
            return state

    async def _process_extract_preferences(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extrae y enriquece preferencias"""
        try:
            preferences = state["preferences"]
            query = state["query"]

            # Combinar preferencias explícitas con las extraídas del query
            extracted_prefs = await self._extract_additional_preferences(query)
            enriched_prefs = self._merge_preferences(preferences, extracted_prefs)

            # Validar y normalizar preferencias
            validated_prefs = self._validate_preferences(enriched_prefs)

            state["preferences"] = validated_prefs
            state["metadata"]["processing_stages"].append({
                "stage": "extract_preferences",
                "timestamp": datetime.now()
            })

            return state

        except Exception as e:
            logger.error(f"Error extracting preferences: {str(e)}")
            state["metadata"]["errors"] = state.get("metadata", {}).get("errors", [])
            state["metadata"]["errors"].append(str(e))
            return state

    async def _process_get_tour_recommendations(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Obtiene recomendaciones de tours"""
        try:
            if "tour" in state["metadata"]["detected_intents"]:
                tour_recs = await self.tour_agent.plan_tour(state["preferences"])
                state["tour_recommendations"] = tour_recs.dict()

            state["metadata"]["processing_stages"].append({
                "stage": "get_tour_recommendations",
                "timestamp": datetime.now()
            })

            return state

        except Exception as e:
            logger.error(f"Error getting tour recommendations: {str(e)}")
            state["metadata"]["errors"] = state.get("metadata", {}).get("errors", [])
            state["metadata"]["errors"].append(str(e))
            return state

    async def _process_get_food_recommendations(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Obtiene recomendaciones de restaurantes"""
        try:
            if "food" in state["metadata"]["detected_intents"]:
                food_recs = await self.food_agent.get_restaurants(
                    preferences=state["preferences"],
                    location=state["preferences"].get("location")
                )
                state["food_recommendations"] = [rec.dict() for rec in food_recs]

            state["metadata"]["processing_stages"].append({
                "stage": "get_food_recommendations",
                "timestamp": datetime.now()
            })

            return state

        except Exception as e:
            logger.error(f"Error getting food recommendations: {str(e)}")
            state["metadata"]["errors"] = state.get("metadata", {}).get("errors", [])
            state["metadata"]["errors"].append(str(e))
            return state

    async def _process_get_activity_recommendations(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Obtiene recomendaciones de actividades"""
        try:
            if "activity" in state["metadata"]["detected_intents"]:
                activity_recs = await self.activity_agent.get_activities(
                    preferences=state["preferences"],
                    location=state["preferences"].get("location")
                )
                state["activity_recommendations"] = [rec.dict() for rec in activity_recs]

            state["metadata"]["processing_stages"].append({
                "stage": "get_activity_recommendations",
                "timestamp": datetime.now()
            })

            return state

        except Exception as e:
            logger.error(f"Error getting activity recommendations: {str(e)}")
            state["metadata"]["errors"] = state.get("metadata", {}).get("errors", [])
            state["metadata"]["errors"].append(str(e))
            return state

    async def _process_merge_recommendations(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combina recomendaciones de diferentes fuentes"""
        try:
            all_recommendations = []

            # Añadir recomendaciones de tours
            if "tour_recommendations" in state:
                all_recommendations.extend(
                    self._process_tour_recommendations(state["tour_recommendations"])
                )

            # Añadir recomendaciones de comida
            if "food_recommendations" in state:
                all_recommendations.extend(
                    self._process_food_recommendations(state["food_recommendations"])
                )

            # Añadir recomendaciones de actividades
            if "activity_recommendations" in state:
                all_recommendations.extend(
                    self._process_activity_recommendations(state["activity_recommendations"])
                )

            state["merged_recommendations"] = all_recommendations
            state["metadata"]["processing_stages"].append({
                "stage": "merge_recommendations",
                "timestamp": datetime.now()
            })

            return state

        except Exception as e:
            logger.error(f"Error merging recommendations: {str(e)}")
            state["metadata"]["errors"] = state.get("metadata", {}).get("errors", [])
            state["metadata"]["errors"].append(str(e))
            return state

    async def _process_rank_recommendations(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Rankea y filtra las recomendaciones"""
        try:
            recommendations = state["merged_recommendations"]

            # Calcular scores
            scored_recommendations = []
            for rec in recommendations:
                score = self._calculate_recommendation_score(
                    rec,
                    state["preferences"]
                )
                scored_recommendations.append((rec, score))

            # Ordenar por score
            scored_recommendations.sort(key=lambda x: x[1], reverse=True)

            # Filtrar mejores recomendaciones
            top_recommendations = [
                rec for rec, score in scored_recommendations[:10]  # Top 10
            ]

            state["recommendations"] = top_recommendations
            state["metadata"]["processing_stages"].append({
                "stage": "rank_recommendations",
                "timestamp": datetime.now()
            })

            return state

        except Exception as e:
            logger.error(f"Error ranking recommendations: {str(e)}")
            state["metadata"]["errors"] = state.get("metadata", {}).get("errors", [])
            state["metadata"]["errors"].append(str(e))
            return state

    async def _process_enrich_content(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enriquece las recomendaciones con información adicional"""
        try:
            recommendations = state["recommendations"]
            enriched_recommendations = []

            for rec in recommendations:
                enriched_rec = rec.copy()

                # Añadir información contextual
                if rec.get("location"):
                    location_info = await self._get_location_details(rec["location"])
                    if location_info:
                        enriched_rec["location_details"] = location_info

                # Añadir tips y reseñas
                if rec.get("id"):
                    reviews = await self._get_top_reviews(rec["id"])
                    if reviews:
                        enriched_rec["top_reviews"] = reviews

                # Añadir recomendaciones relacionadas
                related_recs = await self._get_related_recommendations(rec)
                if related_recs:
                    enriched_rec["related_recommendations"] = related_recs

                enriched_recommendations.append(enriched_rec)

            state["recommendations"] = enriched_recommendations
            state["metadata"]["processing_stages"].append({
                "stage": "enrich_content",
                "timestamp": datetime.now()
            })

            return state

        except Exception as e:
            logger.error(f"Error enriching content: {str(e)}")
            state["metadata"]["errors"] = state.get("metadata", {}).get("errors", [])
            state["metadata"]["errors"].append(str(e))
            return state

    async def _process_finalize(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Finaliza el procesamiento y prepara el resultado"""
        try:
            # Calcular métricas finales
            state["metadata"]["end_time"] = datetime.now()
            state["metadata"]["total_recommendations"] = len(state["recommendations"])
            state["metadata"]["processing_time"] = (
                state["metadata"]["end_time"] - state["metadata"]["start_time"]
            ).total_seconds()

            # Limpiar datos temporales
            state.pop("merged_recommendations", None)
            state.pop("tour_recommendations", None)
            state.pop("food_recommendations", None)
            state.pop("activity_recommendations", None)

            return state

        except Exception as e:
            logger.error(f"Error finalizing process: {str(e)}")
            state["metadata"]["errors"] = state.get("metadata", {}).get("errors", [])
            state["metadata"]["errors"].append(str(e))
            return state

    def _calculate_metrics(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Calcula métricas de calidad de las recomendaciones"""
        try:
            preferences = state["preferences"]
            recommendations = state["recommendations"]

            metrics = {
                "preference_match_score": self._calculate_preference_match(
                    recommendations,
                    preferences
                ),
                "diversity_score": self._calculate_diversity_score(recommendations),
                "coverage_score": self._calculate_coverage_score(
                    recommendations,
                    preferences
                ),
                "processing_time": state["metadata"]["processing_time"]
            }

            if "errors" in state["metadata"]:
                metrics["error_rate"] = len(state["metadata"]["errors"]) / len(
                    state["metadata"]["processing_stages"]
                )

            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}

    def _calculate_preference_match(
        self,
        recommendations: List[Dict[str, Any]],
        preferences: Dict[str, Any]
    ) -> float:
        """Calcula qué tan bien coinciden las recomendaciones con las preferencias"""
        if not recommendations:
            return 0.0

        total_score = 0.0
        for rec in recommendations:
            # Verificar coincidencia de categorías
            if preferences.get("interests"):
                categories = set(rec.get("categories", []))
                interests = set(preferences["interests"])
                category_match = len(categories & interests) / len(interests)
                total_score += category_match * 0.4

            # Verificar coincidencia de presupuesto
            if preferences.get("budget_per_day") and rec.get("cost"):
                if rec["cost"] <= preferences["budget_per_day"]:
                    total_score += 0.3

            # Verificar coincidencia de ubicación
            if preferences.get("location") and rec.get("location"):
                if preferences["location"].lower() == rec["location"].lower():
                    total_score += 0.3

        return total_score / len(recommendations)

    def _calculate_diversity_score(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> float:
        """Calcula la diversidad de las recomendaciones"""
        if not recommendations:
            return 0.0

        # Contar tipos únicos
        unique_types = len(set(rec.get("type") for rec in recommendations))
        type_diversity = unique_types / len(recommendations)

        # Contar categorías únicas
        unique_categories = len(set(
            cat for rec in recommendations 
            for cat in rec.get("categories", [])
        ))
        category_diversity = unique_categories / (len(recommendations) * 2)  # Factor 2 es arbitrario

        # Contar ubicaciones únicas
        unique_locations = len(set(rec.get("location") for rec in recommendations))
        location_diversity = unique_locations / len(recommendations)

        return (type_diversity + category_diversity + location_diversity) / 3