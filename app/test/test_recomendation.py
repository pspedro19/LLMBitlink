import pytest
import os
import json
from app.core.recommender.recommend import recommend
from app.core.analyzer.nlp_processor import ImprovedNLPProcessor

class TestRecommendationSystem:
    """Tests para el sistema completo de recomendaciones turísticas."""
    
    @pytest.fixture
    def nlp_processor(self):
        """Fixture para el procesador NLP."""
        return ImprovedNLPProcessor()
    
    def test_basic_recommendation_flow(self, nlp_processor):
        """Test del flujo básico de recomendaciones con una consulta simple."""
        # Consulta básica en español
        query = """Quiero visitar Willemstad durante 3 días con un presupuesto de $150 por día. 
               Me interesa la arquitectura colonial holandesa y museos."""
        
        # Obtener recomendaciones
        result = recommend(query, {})
        
        # Imprimir resultados para análisis
        print("\n==== RESULTADOS DE RECOMENDACIÓN BÁSICA ====")
        print(f"Estado: {result.get('status')}")
        print(f"Número de recomendaciones: {len(result.get('recommendations', []))}")
        
        if 'query_analysis' in result:
            print("\nAnálisis de consulta:")
            for key, value in result.get('query_analysis', {}).items():
                if key == 'extracted_preferences':
                    print(f"  - Preferencias extraídas: {value}")
                else:
                    print(f"  - {key}: {value}")
                    
        # Verificaciones
        assert result.get('status') in ['success', 'no_results'], "El estado debería ser 'success' o 'no_results'"
        assert 'recommendations' in result, "Debería haber una clave 'recommendations'"
        assert 'query_analysis' in result, "Debería haber una clave 'query_analysis'"
        assert 'metadata' in result, "Debería haber una clave 'metadata'"
    
    def test_recommendation_with_explicit_preferences(self, nlp_processor):
        """Test de recomendaciones con preferencias explícitas."""
        # Consulta breve
        query = "Lugares para visitar en Curaçao"
        
        # Preferencias explícitas
        explicit_preferences = {
            "interests": ["cultural", "history"],
            "locations": ["Willemstad", "Punda"],
            "budget_per_day": 200,
            "trip_duration": 4
        }
        
        # Obtener recomendaciones
        result = recommend(query, explicit_preferences)
        
        # Imprimir resultados para análisis
        print("\n==== RESULTADOS CON PREFERENCIAS EXPLÍCITAS ====")
        print(f"Estado: {result.get('status')}")
        print(f"Número de recomendaciones: {len(result.get('recommendations', []))}")
        
        if result.get('recommendations'):
            print("\nPrimera recomendación:")
            first_rec = result.get('recommendations')[0]
            for key in ['name', 'type', 'location', 'rating']:
                if key in first_rec:
                    print(f"  - {key}: {first_rec[key]}")
        
        # Verificaciones
        assert result.get('status') in ['success', 'no_results'], "El estado debería ser 'success' o 'no_results'"
        
        # Verificar que las recomendaciones sean relevantes (si hay)
        if result.get('recommendations'):
            recs = result.get('recommendations')
            # Verificar estructura de recomendaciones
            assert all('name' in rec for rec in recs), "Todas las recomendaciones deben tener 'name'"
            assert all('location' in rec for rec in recs), "Todas las recomendaciones deben tener 'location'"
            
            # Verificar relevancia (ubicaciones)
            relevant_locations = [loc.lower() for loc in explicit_preferences.get('locations', [])]
            location_match = any(
                any(loc.lower() in rec.get('location', '').lower() for loc in relevant_locations)
                for rec in recs[:5]  # Verificar primeras 5 recomendaciones
            )
            print(f"\nCoincidencia de ubicación en primeras 5 recomendaciones: {location_match}")
    
    def test_advanced_recommendation_scenarios(self):
        """Test de escenarios avanzados de recomendación."""
        test_cases = [
            {
                "name": "Escenario de playa y snorkel",
                "query": "Quiero hacer snorkel en las mejores playas de Curaçao durante 5 días",
                "preferences": {
                    "interests": ["water_activities", "nature"],
                    "budget_per_day": 180
                }
            },
            {
                "name": "Escenario cultural e histórico",
                "query": "Me interesan los sitios históricos y la arquitectura colonial",
                "preferences": {
                    "locations": ["Willemstad"],
                    "trip_duration": 3
                }
            },
            {
                "name": "Escenario gastronómico",
                "query": "Busco probar la comida local y visitar restaurantes tradicionales",
                "preferences": {
                    "interests": ["food"],
                    "budget_per_day": 250
                }
            }
        ]
        
        for i, case in enumerate(test_cases):
            print(f"\n==== TEST CASO {i+1}: {case['name']} ====")
            result = recommend(case["query"], case["preferences"])
            
            print(f"Estado: {result.get('status')}")
            print(f"Número de recomendaciones: {len(result.get('recommendations', []))}")
            
            # Verificaciones básicas
            assert result.get('status') in ['success', 'no_results']
            assert 'recommendations' in result
            
            # Guardar preferencias extraídas
            if 'query_analysis' in result and 'extracted_preferences' in result['query_analysis']:
                extracted = result['query_analysis']['extracted_preferences']
                print(f"Preferencias extraídas: {extracted}")
            
            # Verificar recomendaciones
            if result.get('recommendations'):
                print("\nCategorías de las primeras 5 recomendaciones:")
                categories = {}
                for rec in result.get('recommendations')[:5]:
                    rec_type = rec.get('type', 'unknown')
                    categories[rec_type] = categories.get(rec_type, 0) + 1
                
                for category, count in categories.items():
                    print(f"  - {category}: {count}")
    
    def test_recommendation_with_validation(self, nlp_processor):
        """Test que incluya validación de los resultados."""
        query = """Want to explore the UNESCO World Heritage sites in Punda and
                Otrobanda, focusing on the colorful Dutch architecture and Queen
                Emma Bridge. Interested in the Mikvé Israel-Emanuel Synagogue and
                Maritime Museum. Budget $200/day."""
        
        # Preferencias explícitas mínimas
        explicit_preferences = {
            "trip_duration": 3,
            "group_size": 2
        }
        
        # Obtener recomendaciones
        result = recommend(query, explicit_preferences)
        
        # Imprimir resultados de validación
        print("\n==== RESULTADOS DE VALIDACIÓN ====")
        if 'validation' in result:
            print("\nMétricas de validación:")
            for metric, value in result.get('validation', {}).items():
                print(f"  - {metric}: {value}")
        
        # Verificar que haya resultados
        assert result.get('status') in ['success', 'no_results']
        
        # Si hay métricas de validación, verificarlas
        if 'validation' in result and result.get('validation'):
            validation = result.get('validation')
            print("\nVerificación de métricas:")
            for metric in ['location_match', 'interest_match']:
                if metric in validation:
                    score = validation[metric]
                    print(f"  - {metric}: {score}")
                    assert 0 <= score <= 1, f"La métrica {metric} debe estar entre 0 y 1"
    
    def test_recommendations_with_various_languages(self):
        """Test de recomendaciones con consultas en diferentes idiomas."""
        language_queries = [
            {
                "language": "English",
                "query": "I want to visit beaches and do snorkeling for 4 days"
            },
            {
                "language": "Spanish",
                "query": "Quiero conocer sitios históricos y museos por 3 días"
            }
        ]
        
        for lang_case in language_queries:
            print(f"\n==== PRUEBA EN {lang_case['language']} ====")
            print(f"Consulta: {lang_case['query']}")
            
            result = recommend(lang_case['query'], {})
            
            print(f"Estado: {result.get('status')}")
            print(f"Número de recomendaciones: {len(result.get('recommendations', []))}")
            
            # Verificar preferencias extraídas
            if 'query_analysis' in result and 'extracted_preferences' in result['query_analysis']:
                extracted = result['query_analysis']['extracted_preferences']
                print(f"Preferencias extraídas en {lang_case['language']}:")
                for key, value in extracted.items():
                    if value and key not in ['timestamp']:
                        print(f"  - {key}: {value}")
            
            # Verificaciones básicas
            assert result.get('status') in ['success', 'no_results']
            assert 'recommendations' in result

    def save_test_results_to_file(self, nlp_processor):
        """Test que guarda los resultados de recomendación para análisis posterior."""
        # Esta función es opcional y puede usarse para generar datos 
        # para análisis detallado fuera de las pruebas
        
        query = "Quiero visitar Willemstad y hacer snorkel en Playa Kalki. Me interesa la arquitectura colonial y probar comida local."
        
        # Obtener recomendaciones
        result = recommend(query, {})
        
        # Convertir a formato guardable (eliminar elementos no serializables)
        savable_result = {
            "status": result.get("status"),
            "query": query,
            "recommendations_count": len(result.get("recommendations", [])),
            "recommendations_sample": result.get("recommendations", [])[:3],  # Primeras 3 recomendaciones
            "extracted_preferences": result.get("query_analysis", {}).get("extracted_preferences", {}),
            "validation": result.get("validation", {})
        }
        
        # Guardar resultados en un archivo JSON
        try:
            os.makedirs("test_outputs", exist_ok=True)
            with open("test_outputs/recommendation_test_result.json", "w", encoding="utf-8") as f:
                json.dump(savable_result, f, indent=2, ensure_ascii=False)
            print("\nResultados guardados en test_outputs/recommendation_test_result.json")
        except Exception as e:
            print(f"Error al guardar resultados: {str(e)}")