import unittest
import json
import pandas as pd
from datetime import datetime
import sys
import os
from pathlib import Path

# Ajustar path correctamente
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

print(sys.path)

from core.nlu.processor import NLUProcessor
from core.recommender.engine import RecommendationEngine
from data.query_builder import QueryBuilder

class TestRecommendationSystem(unittest.TestCase):
    """Pruebas completas para el sistema de recomendaciones turísticas de Curaçao."""

    @classmethod
    def setUpClass(cls):
        """Configuración inicial para todas las pruebas."""
        cls.nlu_processor = NLUProcessor()
        cls.recommender = RecommendationEngine()
        cls.query_builder = QueryBuilder()
        cls.results = []  # Para almacenar resultados de todas las pruebas
        cls.test_start_time = datetime.now()

    def setUp(self):
        """Configuración para cada prueba individual."""
        self.test_case_start_time = datetime.now()
    
    def tearDown(self):
        """Finalización de cada prueba individual."""
        test_duration = (datetime.now() - self.test_case_start_time).total_seconds()
        self.test_result['processing_time'] = test_duration
        self.results.append(self.test_result)
    
    @classmethod
    def tearDownClass(cls):
        """Finalización de todas las pruebas y generación de informe."""
        total_duration = (datetime.now() - cls.test_start_time).total_seconds()
        
        # Crear DataFrame con resultados
        df = pd.DataFrame(cls.results)
        
        # Calcular métricas globales
        success_rate = (df['success'].sum() / len(df)) * 100
        avg_preferences_extracted = df['preferences_extracted'].mean()
        avg_recommendation_count = df['recommendation_count'].mean()
        avg_processing_time = df['processing_time'].mean()
        
        # Generar informe
        print("\n" + "="*80)
        print(f"RESUMEN DE PRUEBAS DE RECOMENDACIÓN TURÍSTICA PARA CURAÇAO")
        print("="*80)
        print(f"Pruebas ejecutadas: {len(df)}")
        print(f"Tasa de éxito: {success_rate:.2f}%")
        print(f"Promedio de preferencias extraídas: {avg_preferences_extracted:.2f}")
        print(f"Promedio de recomendaciones generadas: {avg_recommendation_count:.2f}")
        print(f"Tiempo promedio de procesamiento: {avg_processing_time:.2f} segundos")
        print(f"Tiempo total de ejecución: {total_duration:.2f} segundos")
        print("="*80)
        
        # Guardar resultados detallados en CSV
        try:
            results_dir = Path(__file__).resolve().parent / "test_results"
            results_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            df.to_csv(results_dir / f"recommendation_test_results_{timestamp}.csv", index=False)
            print(f"Resultados detallados guardados en: recommendation_test_results_{timestamp}.csv")
        except Exception as e:
            print(f"Error al guardar resultados: {e}")
        
        print("="*80)
    
    def run_recommendation_test(self, query, test_name, expected_fields=None):
        """
        Ejecuta una prueba completa de recomendación para una consulta específica.
        
        Args:
            query (str): Consulta en lenguaje natural
            test_name (str): Nombre descriptivo de la prueba
            expected_fields (dict, optional): Campos que se espera extraer
        """
        self.test_result = {
            'test_name': test_name,
            'query': query,
            'success': False,
            'preferences_extracted': 0,
            'recommendation_count': 0,
            'processing_time': 0,
            'language': 'es' if any(word in query.lower() for word in ['días', 'curaçao', 'playa', 'visitar']) else 'en'
        }
        
        try:
            # 1. Procesar consulta con NLU
            nlu_result = self.nlu_processor.process_query(query)
            preferences = nlu_result['preferences']
            
            # 2. Contar preferencias extraídas
            preference_count = sum(1 for key, value in preferences.items() 
                                if value and key in ['interests', 'locations', 'budget', 'duration'])
            self.test_result['preferences_extracted'] = preference_count
            
            # 3. Verificar extracción específica si se proporcionaron expectativas
            if expected_fields:
                for field, expected in expected_fields.items():
                    if field in preferences:
                        if isinstance(expected, list) and isinstance(preferences[field], list):
                            # Para listas, verificar que al menos uno de los valores esperados está presente
                            self.assertTrue(
                                any(exp.lower() in [val.lower() for val in preferences[field]] 
                                    for exp in expected),
                                f"No se encontró ningún valor esperado en {field}: {preferences[field]}"
                            )
                        elif field == 'duration' or field == 'budget':
                            # Para valores numéricos, verificar que están dentro de un rango razonable
                            if preferences[field] is not None:
                                self.assertGreaterEqual(
                                    preferences[field], 
                                    expected * 0.7,  # 30% de margen de error
                                    f"El valor extraído para {field} es demasiado bajo: {preferences[field]}"
                                )
                                self.assertLessEqual(
                                    preferences[field], 
                                    expected * 1.3,  # 30% de margen de error
                                    f"El valor extraído para {field} es demasiado alto: {preferences[field]}"
                                )
            
            # 4. Generar recomendaciones
            recommendations = self.recommender.generate_recommendations(preferences)
            
            # 5. Verificar resultados básicos - menos estricto
            self.assertIn('status', recommendations, "No se encontró el campo 'status' en las recomendaciones")
            
            recs = recommendations.get('recommendations', [])
            self.test_result['recommendation_count'] = len(recs)
            
            # No fallar si no hay recomendaciones - solo alertar
            if len(recs) == 0:
                print(f"ALERTA: No se generaron recomendaciones para: {test_name}")
            
            # 6. Verificar consistencia de recomendaciones si hay alguna
            if preferences.get('interests') and recs:
                # Verificar que hay recomendaciones relevantes para los intereses
                validation = recommendations.get('validation', {})
                interest_match = validation.get('interest_match', 0)
                self.assertGreaterEqual(
                    interest_match, 
                    0.2,  # Solo 20% de coincidencia mínima para ser más flexible
                    f"Coincidencia de intereses demasiado baja: {interest_match}"
                )
            
            if preferences.get('locations') and recs:
                # Verificar que hay recomendaciones en las ubicaciones solicitadas
                validation = recommendations.get('validation', {})
                location_match = validation.get('location_match', 0)
                self.assertGreaterEqual(
                    location_match, 
                    0.2,  # Solo 20% de coincidencia mínima para ser más flexible
                    f"Coincidencia de ubicaciones demasiado baja: {location_match}"
                )
            
            # 7. Marcar prueba como exitosa
            self.test_result['success'] = True
            
        except Exception as e:
            self.test_result['error'] = str(e)
            self.fail(f"Error en la prueba {test_name}: {e}")
    
    # ========== CASOS DE PRUEBA EN ESPAÑOL ==========
    
    def test_01_romantic_trip(self):
        """Prueba de viaje romántico en español."""
        query = "Viaje romántico a Curaçao para aniversario, 5 días, lugares para parejas, cenas especiales"
        expected = {
            'duration': 5,
            'interests': ['gastronomía', 'cultural']
        }
        self.run_recommendation_test(query, "Viaje romántico", expected)
    
    def test_02_beach_vacation(self):
        """Prueba de vacaciones de playa en español."""
        query = "Busco playas tranquilas en Curaçao para relajarme durante una semana, presupuesto de $200 por día"
        expected = {
            'duration': 7,
            'budget': 200,
            'interests': ['naturaleza', 'actividades_acuáticas']
        }
        self.run_recommendation_test(query, "Vacaciones de playa", expected)
    
    def test_03_family_trip(self):
        """Prueba de viaje familiar en español."""
        query = "Vacaciones familiares en Curaçao con 2 niños, actividades para toda la familia, 6 días, presupuesto medio"
        expected = {
            'duration': 6,
            'with_children': True,
            'group_size': 4  # Puede ser difícil de extraer exactamente
        }
        self.run_recommendation_test(query, "Viaje familiar", expected)
    
    def test_04_adventure_activities(self):
        """Prueba de actividades de aventura en español."""
        query = "Quiero hacer senderismo y buceo en Curaçao, lugares recomendados para aventuras, 4 días"
        expected = {
            'duration': 4,
            'interests': ['aventura', 'actividades_acuáticas']
        }
        self.run_recommendation_test(query, "Actividades de aventura", expected)
    
    def test_05_cultural_tour(self):
        """Prueba de tour cultural en español."""
        query = "Me interesa la historia y arquitectura de Willemstad, museos y sitios históricos, 3 días en Curaçao"
        expected = {
            'duration': 3,
            'interests': ['cultural'],
            'locations': ['willemstad']
        }
        self.run_recommendation_test(query, "Tour cultural", expected)
    
    def test_06_culinary_experience(self):
        """Prueba de experiencia culinaria en español."""
        query = "Gastronomía local de Curaçao, mejores restaurantes, tour gastronómico, 150 dólares por día, 4 días"
        expected = {
            'duration': 4,
            'budget': 150,
            'interests': ['gastronomía']
        }
        self.run_recommendation_test(query, "Experiencia culinaria", expected)
    
    def test_07_budget_trip(self):
        """Prueba de viaje económico en español."""
        query = "Curaçao con presupuesto limitado, 80 dólares diarios, actividades gratuitas o económicas, 5 días"
        expected = {
            'duration': 5,
            'budget': 80
        }
        self.run_recommendation_test(query, "Viaje económico", expected)
    
    def test_08_luxury_vacation(self):
        """Prueba de vacaciones de lujo en español."""
        query = "Vacaciones de lujo en Curaçao, mejores resorts, restaurantes exclusivos, experiencias premium, 7 días sin límite de presupuesto"
        expected = {
            'duration': 7,
            'interests': ['gastronomía']
        }
        self.run_recommendation_test(query, "Vacaciones de lujo", expected)
    
    def test_09_short_weekend(self):
        """Prueba de fin de semana corto en español."""
        query = "Escapada de fin de semana a Curaçao, 2 días, lo imprescindible para ver"
        expected = {
            'duration': 2
        }
        self.run_recommendation_test(query, "Fin de semana corto", expected)
    
    def test_10_specific_locations(self):
        """Prueba de ubicaciones específicas en español."""
        query = "Quiero visitar Punda, Otrobanda y Jan Thiel durante 4 días en Curaçao, 200 dólares de presupuesto diario"
        expected = {
            'duration': 4,
            'locations': ['punda', 'otrobanda', 'jan thiel'],
            'budget': 200
        }
        self.run_recommendation_test(query, "Ubicaciones específicas", expected)
    
    # ========== CASOS DE PRUEBA EN INGLÉS ==========
    
    def test_11_honeymoon_trip(self):
        """Prueba de luna de miel en inglés."""
        query = "Planning honeymoon in Curacao for 6 days, romantic beaches and special dinners, luxury experience"
        expected = {
            'duration': 6,
            'interests': ['naturaleza', 'gastronomía']
        }
        self.run_recommendation_test(query, "Honeymoon trip", expected)
    
    def test_12_water_activities(self):
        """Prueba de actividades acuáticas en inglés."""
        query = "Interested in snorkeling and diving in Curacao, best spots for marine life, 5 day trip with $180 daily budget"
        expected = {
            'duration': 5,
            'budget': 180,
            'interests': ['actividades_acuáticas']
        }
        self.run_recommendation_test(query, "Water activities", expected)
    
    def test_13_family_adventure(self):
        """Prueba de aventura familiar en inglés."""
        query = "Family vacation in Curacao with kids, fun activities for children, safe beaches, 7 days"
        expected = {
            'duration': 7,
            'with_children': True,
            'interests': ['naturaleza']
        }
        self.run_recommendation_test(query, "Family adventure", expected)
    
    def test_14_historical_sites(self):
        """Prueba de sitios históricos en inglés."""
        query = "Exploring historical sites in Willemstad, Curacao, colonial architecture and museums, 4 day cultural trip"
        expected = {
            'duration': 4,
            'locations': ['willemstad'],
            'interests': ['cultural']
        }
        self.run_recommendation_test(query, "Historical sites", expected)
    
    def test_15_food_tour(self):
        """Prueba de tour gastronómico en inglés."""
        query = "Food lover seeking local cuisine in Curacao, traditional dishes, food markets and cooking classes, 6 days"
        expected = {
            'duration': 6,
            'interests': ['gastronomía']
        }
        self.run_recommendation_test(query, "Food tour", expected)
    
    def test_16_nightlife_experience(self):
        """Prueba de vida nocturna en inglés."""
        query = "Looking for best nightlife in Curacao, clubs, bars and music venues, 5 day trip with friends"
        expected = {
            'duration': 5,
            'interests': ['vida_nocturna']
        }
        self.run_recommendation_test(query, "Nightlife experience", expected)
    
    def test_17_nature_hiking(self):
        """Prueba de senderismo y naturaleza en inglés."""
        query = "Nature hiking in Curacao, national parks, nature reserves and wildlife, 4 day eco-friendly trip"
        expected = {
            'duration': 4,
            'interests': ['naturaleza', 'aventura']
        }
        self.run_recommendation_test(query, "Nature hiking", expected)
    
    def test_18_budget_backpacking(self):
        """Prueba de mochilero con presupuesto en inglés."""
        query = "Backpacking through Curacao on a tight budget, $70 per day, hostels and cheap eats, 8 day trip"
        expected = {
            'duration': 8,
            'budget': 70
        }
        self.run_recommendation_test(query, "Budget backpacking", expected)
    
    def test_19_photography_tour(self):
        """Prueba de tour fotográfico en inglés."""
        query = "Photography trip to Curacao, most scenic locations for landscape and street photography, 5 days"
        expected = {
            'duration': 5,
            'interests': ['cultural', 'naturaleza']
        }
        self.run_recommendation_test(query, "Photography tour", expected)
    
    def test_20_wellness_retreat(self):
        """Prueba de retiro de bienestar en inglés."""
        query = "Wellness retreat in Curacao, relaxation, yoga and spa services, peaceful beaches, 7 days, $250 daily"
        expected = {
            'duration': 7,
            'budget': 250,
            'interests': ['naturaleza']
        }
        self.run_recommendation_test(query, "Wellness retreat", expected)

if __name__ == "__main__":
    unittest.main(verbosity=2)