"""
Pruebas unitarias para los agentes especializados del sistema.
"""
import pytest
from typing import Dict, Any, List
from datetime import datetime, timedelta
from core.rag.agents.tour_agent import TourAgent
from core.rag.agents.food_agent import FoodAgent
from core.rag.agents.activity_agent import ActivityAgent
from langchain.graphs import Neo4jGraph

# Fixtures
@pytest.fixture
def neo4j_graph():
    """Fixture para conexión a Neo4j"""
    return Neo4jGraph(
        url="neo4j://localhost:7687",
        username="neo4j",
        password="testpassword"
    )

@pytest.fixture
def tour_agent(neo4j_graph):
    """Fixture para TourAgent"""
    return TourAgent(
        graph=neo4j_graph,
        config={"use_cache": True}
    )

@pytest.fixture
def food_agent(neo4j_graph):
    """Fixture para FoodAgent"""
    return FoodAgent(
        graph=neo4j_graph,
        config={"use_cache": True}
    )

@pytest.fixture
def activity_agent(neo4j_graph):
    """Fixture para ActivityAgent"""
    return ActivityAgent(
        graph=neo4j_graph,
        config={"use_cache": True}
    )

@pytest.fixture
def sample_preferences():
    """Fixture para preferencias de prueba"""
    return {
        "interests": ["history", "culture", "food"],
        "locations": ["Willemstad"],
        "budget_per_day": 150.0,
        "trip_duration": 3,
        "group_size": 2,
        "activity_types": ["walking_tour", "sightseeing"],
        "cuisine_preferences": ["local", "seafood"]
    }

# Tests para TourAgent
async def test_tour_agent_plan_tour(tour_agent, sample_preferences):
    """Prueba planificación de tour"""
    result = await tour_agent.plan_tour(sample_preferences)
    
    assert result is not None
    assert "days" in result.dict()
    assert "overview" in result.dict()
    
    # Verificar estructura del plan
    assert len(result.days) == sample_preferences["trip_duration"]
    for day in result.days:
        assert "activities" in day
        assert "total_cost" in day
        assert day["total_cost"] <= sample_preferences["budget_per_day"]

# Tests para FoodAgent
async def test_food_agent_get_restaurants(food_agent, sample_preferences):
    """Prueba recomendaciones de restaurantes"""
    restaurants = await food_agent.get_restaurants(
        preferences=sample_preferences,
        location="Willemstad"
    )
    
    assert isinstance(restaurants, list)
    assert len(restaurants) > 0
    
    for restaurant in restaurants:
        assert "name" in restaurant
        assert "cuisine" in restaurant
        assert "price_range" in restaurant
        assert "rating" in restaurant
        
        # Verificar que coincida con preferencias
        assert any(cuisine in sample_preferences["cuisine_preferences"] 
                  for cuisine in restaurant.cuisine)

async def test_food_agent_check_availability(food_agent):
    """Prueba verificación de disponibilidad"""
    restaurant_id = "test_restaurant"
    datetime_obj = datetime.now() + timedelta(days=1)
    party_size = 2
    
    result = await food_agent.check_restaurant_availability(
        restaurant_id,
        datetime_obj,
        party_size
    )
    
    assert "available" in result
    if result["available"]:
        assert result["party_size"] == party_size
        assert "available_seats" in result
    else:
        assert "reason" in result

# Tests para ActivityAgent
async def test_activity_agent_get_activities(activity_agent, sample_preferences):
    """Prueba recomendaciones de actividades"""
    activities = await activity_agent.get_activities(
        preferences=sample_preferences
    )
    
    assert isinstance(activities, list)
    assert len(activities) > 0
    
    for activity in activities:
        assert "name" in activity
        assert "type" in activity
        assert "duration" in activity
        assert "cost" in activity
        assert "rating" in activity
        
        # Verificar restricciones de presupuesto
        assert activity.cost <= sample_preferences["budget_per_day"]

async def test_activity_agent_get_similar_activities(activity_agent):
    """Prueba búsqueda de actividades similares"""
    activity_id = "test_activity"
    
    similar_activities = await activity_agent.get_similar_activities(
        activity_id,
        limit=3
    )
    
    assert isinstance(similar_activities, list)
    if similar_activities:
        for activity in similar_activities:
            assert "name" in activity
            assert "type" in activity
            assert activity.id != activity_id

# Tests de integración
async def test_agent_coordination(
    tour_agent,
    food_agent,
    activity_agent,
    sample_preferences
):
    """Prueba coordinación entre agentes"""
    # Obtener plan de tour
    tour_plan = await tour_agent.plan_tour(sample_preferences)
    
    # Para cada día del tour
    for day in tour_plan.days:
        # Verificar restaurantes cercanos a actividades
        for activity in day["activities"]:
            restaurants = await food_agent.get_nearby_restaurants(
                activity["location"]
            )
            assert isinstance(restaurants, list)
            
            # Verificar actividades alternativas
            alternatives = await activity_agent.get_similar_activities(
                activity["id"]
            )
            assert isinstance(alternatives, list)

# Tests de manejo de errores
async def test_tour_agent_error_handling(tour_agent):
    """Prueba manejo de errores en TourAgent"""
    with pytest.raises(ValueError):
        await tour_agent.plan_tour({})  # Preferencias vacías

    invalid_preferences = {
        "budget_per_day": -100,
        "trip_duration": 0,
        "group_size": -1
    }
    with pytest.raises(ValueError):
        await tour_agent.plan_tour(invalid_preferences)

async def test_food_agent_error_handling(food_agent):
    """Prueba manejo de errores en FoodAgent"""
    # Probar con ubicación inválida
    restaurants = await food_agent.get_restaurants(
        preferences={},
        location="NonexistentLocation"
    )
    assert len(restaurants) == 0

    # Probar check_availability con ID inválido
    result = await food_agent.check_restaurant_availability(
        "invalid_id",
        datetime.now(),
        2
    )
    assert not result["available"]
    assert "reason" in result

async def test_activity_agent_error_handling(activity_agent):
    """Prueba manejo de errores en ActivityAgent"""
    # Probar con preferencias inválidas
    activities = await activity_agent.get_activities({
        "difficulty": "impossible",
        "budget_per_day": -50
    })
    assert len(activities) == 0

    # Probar con ID de actividad inválido
    details = await activity_agent.get_activity_details("invalid_id")
    assert details is None

# Tests de rendimiento
@pytest.mark.timeout(5)  # Timeout de 5 segundos
async def test_tour_agent_performance(tour_agent, sample_preferences):
    """Prueba rendimiento de TourAgent"""
    start_time = datetime.now()
    
    result = await tour_agent.plan_tour(sample_preferences)
    
    duration = (datetime.now() - start_time).total_seconds()
    assert duration < 5  # No debe tomar más de 5 segundos
    assert result is not None

@pytest.mark.timeout(2)  # Timeout de 2 segundos
async def test_food_agent_performance(food_agent, sample_preferences):
    """Prueba rendimiento de FoodAgent"""
    start_time = datetime.now()
    
    restaurants = await food_agent.get_restaurants(sample_preferences)
    
    duration = (datetime.now() - start_time).total_seconds()
    assert duration < 2  # No debe tomar más de 2 segundos
    assert len(restaurants) > 0

@pytest.mark.timeout(2)  # Timeout de 2 segundos
async def test_activity_agent_performance(activity_agent, sample_preferences):
    """Prueba rendimiento de ActivityAgent"""
    start_time = datetime.now()
    
    activities = await activity_agent.get_activities(sample_preferences)
    
    duration = (datetime.now() - start_time).total_seconds()
    assert duration < 2  # No debe tomar más de 2 segundos
    assert len(activities) > 0

# Tests de concurrencia
async def test_concurrent_agent_requests(
    tour_agent,
    food_agent,
    activity_agent,
    sample_preferences
):
    """Prueba solicitudes concurrentes a múltiples agentes"""
    import asyncio
    
    async def make_requests():
        tasks = [
            tour_agent.plan_tour(sample_preferences),
            food_agent.get_restaurants(sample_preferences),
            activity_agent.get_activities(sample_preferences)
        ]
        results = await asyncio.gather(*tasks)
        return results

    results = await make_requests()
    assert len(results) == 3
    assert all(result is not None for result in results)

# Tests de validación de datos
def test_validate_tour_plan(tour_agent, sample_preferences):
    """Prueba validación de plan de tour"""
    async def validate():
        plan = await tour_agent.plan_tour(sample_preferences)
        plan_dict = plan.dict()
        
        assert "days" in plan_dict
        assert "overview" in plan_dict
        assert isinstance(plan_dict["days"], list)
        
        total_cost = sum(
            activity["cost"]
            for day in plan_dict["days"]
            for activity in day["activities"]
        )
        assert total_cost <= sample_preferences["budget_per_day"] * len(plan_dict["days"])

    asyncio.run(validate())

def test_validate_restaurant_data(food_agent, sample_preferences):
    """Prueba validación de datos de restaurantes"""
    async def validate():
        restaurants = await food_agent.get_restaurants(sample_preferences)
        
        for restaurant in restaurants:
            assert isinstance(restaurant.name, str)
            assert isinstance(restaurant.cuisine, list)
            assert isinstance(restaurant.price_range, str)
            assert isinstance(restaurant.rating, float)
            assert 0 <= restaurant.rating <= 5
            assert restaurant.location in sample_preferences["locations"]

    asyncio.run(validate())

def test_validate_activity_data(activity_agent, sample_preferences):
    """Prueba validación de datos de actividades"""
    async def validate():
        activities = await activity_agent.get_activities(sample_preferences)
        
        for activity in activities:
            assert isinstance(activity.name, str)
            assert isinstance(activity.type, str)
            assert isinstance(activity.duration, int)
            assert isinstance(activity.cost, float)
            assert activity.cost >= 0
            assert isinstance(activity.rating, float)
            assert 0 <= activity.rating <= 5
            assert activity.location in sample_preferences["locations"]

    asyncio.run(validate())

# Tests de cache
async def test_tour_agent_cache(tour_agent, sample_preferences):
    """Prueba funcionamiento del cache en TourAgent"""
    # Primera llamada
    start_time = datetime.now()
    result1 = await tour_agent.plan_tour(sample_preferences)
    duration1 = (datetime.now() - start_time).total_seconds()
    
    # Segunda llamada (debería usar cache)
    start_time = datetime.now()
    result2 = await tour_agent.plan_tour(sample_preferences)
    duration2 = (datetime.now() - start_time).total_seconds()
    
    assert duration2 < duration1  # Segunda llamada debe ser más rápida
    assert result1.dict() == result2.dict()  # Resultados deben ser idénticos

# Test de utilidades
def test_tour_agent_utils(tour_agent):
    """Prueba funciones utilitarias del TourAgent"""
    # Prueba cálculo de distancia
    coord1 = {"lat": 12.1096, "lon": -68.9367}  # Willemstad
    coord2 = {"lat": 12.2129, "lon": -69.0820}  # Westpunt
    
    distance = tour_agent._calculate_distance(coord1, coord2)
    assert isinstance(distance, float)
    assert distance > 0

def test_food_agent_utils(food_agent):
    """Prueba funciones utilitarias del FoodAgent"""
    # Prueba procesamiento de horarios
    schedule = {
        "monday": {"open": "09:00", "close": "22:00"},
        "tuesday": {"open": "09:00", "close": "22:00"}
    }
    
    processed = food_agent._process_hours(schedule)
    assert isinstance(processed, dict)
    assert "monday" in processed
    assert "tuesday" in processed

def test_activity_agent_utils(activity_agent):
    """Prueba funciones utilitarias del ActivityAgent"""
    # Prueba procesamiento de datos de actividad
    activity_data = {
        "name": "Test Activity",
        "type": "tour",
        "duration": "120",
        "cost": "50.0"
    }
    
    processed = activity_agent._process_activity(activity_data)
    assert isinstance(processed["duration"], int)
    assert isinstance(processed["cost"], float)

if __name__ == "__main__":
    pytest.main([__file__])