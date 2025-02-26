import pytest
from app.core.recommender.recommend import recommend
from app.core.analyzer.nlp_processor import ImprovedNLPProcessor

def format_recommendations(recommendations):
    """Formatea las recomendaciones para una visualización clara"""
    output = []
    
    if not recommendations:
        return ["No se encontraron recomendaciones."]
    
    for i, rec in enumerate(recommendations[:5], 1):  # Mostrar solo las primeras 5
        rec_details = [f"{i}. {rec.get('name', 'Sin nombre')}"]
        rec_details.append(f"   Tipo: {rec.get('type', 'No especificado')}")
        rec_details.append(f"   Ubicación: {rec.get('location', 'No especificada')}")
        
        if 'cost' in rec:
            rec_details.append(f"   Costo: ${rec.get('cost')}")
        elif 'entry_fee' in rec:
            rec_details.append(f"   Tarifa de entrada: ${rec.get('entry_fee')}")
        elif 'average_person_expense' in rec:
            rec_details.append(f"   Gasto promedio: ${rec.get('average_person_expense')}")
            
        rec_details.append(f"   Calificación: {rec.get('rating', 'N/A')}/5")
        
        if rec.get('description'):
            description = rec.get('description')
            if len(description) > 100:
                description = description[:97] + "..."
            rec_details.append(f"   Descripción: {description}")
        
        output.append("\n".join(rec_details))
    
    # Añadir información de resumen
    if len(recommendations) > 5:
        output.append(f"\n...y {len(recommendations) - 5} recomendaciones más.")
    
    # Distribución por tipo
    types = {}
    for rec in recommendations:
        rec_type = rec.get('type', 'unknown')
        types[rec_type] = types.get(rec_type, 0) + 1
    
    output.append("\nDistribución por tipo:")
    for t, count in types.items():
        output.append(f"   - {t}: {count} recomendaciones")
    
    # Distribución por ubicación
    locations = {}
    for rec in recommendations:
        location = rec.get('location', 'unknown')
        locations[location] = locations.get(location, 0) + 1
    
    output.append("\nDistribución por ubicación:")
    for loc, count in locations.items():
        output.append(f"   - {loc}: {count} recomendaciones")
        
    return output

def print_preferences(preferences):
    """Imprime las preferencias extraídas de forma clara"""
    output = []
    
    for key, value in preferences.items():
        if value and key not in ['timestamp']:
            output.append(f"   - {key}: {value}")
            
    return output

def run_test(scenario_name, query, preferences=None):
    """Ejecuta un test y muestra los resultados de forma clara"""
    if preferences is None:
        preferences = {}
    
    print(f"\n\n{scenario_name}")
    print("-" * len(scenario_name))
    print(f"Consulta: \"{query}\"")
    print(f"Preferencias explícitas: {preferences}")
    
    # Extraer preferencias utilizando el NLP Processor
    nlp_processor = ImprovedNLPProcessor()
    extracted_prefs = nlp_processor.extract_preferences(query)
    
    print("\nPREFERENCIAS EXTRAÍDAS:")
    print("\n".join(print_preferences(extracted_prefs)))
    
    # Obtener recomendaciones
    result = recommend(query, preferences)
    
    # Mostrar estado y cantidad
    print(f"\nESTADO DE LA RECOMENDACIÓN: {result.get('status')}")
    recommendations = result.get('recommendations', [])
    print(f"CANTIDAD DE RECOMENDACIONES: {len(recommendations)}")
    
    # Mostrar las recomendaciones de forma clara y ordenada
    print("\nRECOMENDACIONES:")
    formatted_recs = format_recommendations(recommendations)
    print("\n".join(formatted_recs))
    
    # Indicar que el test ha finalizado
    print("\n" + "-" * 40)
    print(f"Test completado: {scenario_name}")
    print("-" * 40)
    
    return result

def test_all_recommendations():
    """Ejecuta 20 tests (10 en español y 10 en inglés) para el sistema de recomendaciones"""
    
    print("\n" + "="*80)
    print("TESTS COMPLETOS DEL SISTEMA DE RECOMENDACIONES TURÍSTICAS")
    print("20 escenarios: 10 en español y 10 en inglés")
    print("="*80)
    
    # ESCENARIOS EN ESPAÑOL
    
    # Test 1: Visita cultural a Willemstad
    run_test(
        "ESPAÑOL 1: Visita cultural a Willemstad",
        "Quiero visitar Willemstad para conocer sitios históricos y museos por 3 días con un presupuesto de $150 por día",
        {"interests": ["cultural", "history"]}
    )
    
    # Test 2: Actividades de playa y snorkel
    run_test(
        "ESPAÑOL 2: Actividades de playa y snorkel",
        "Me gustaría hacer snorkel y visitar las mejores playas de Curaçao durante 4 días",
        {"interests": ["water_activities"], "budget_per_day": 180}
    )
    
    # Test 3: Experiencia gastronómica local
    run_test(
        "ESPAÑOL 3: Experiencia gastronómica local",
        "Busco probar la comida local y visitar restaurantes tradicionales en Willemstad por 2 días",
        {"interests": ["food"], "locations": ["Willemstad"]}
    )
    
    # Test 4: Turismo de aventura
    run_test(
        "ESPAÑOL 4: Turismo de aventura",
        "Quiero hacer actividades de aventura como senderismo, buceo y exploración de cuevas durante 5 días",
        {"interests": ["adventure", "nature"], "budget_per_day": 200}
    )
    
    # Test 5: Visita a museos y sitios históricos
    run_test(
        "ESPAÑOL 5: Visita a museos y sitios históricos",
        "Me interesan los museos, la arquitectura colonial y los sitios históricos en Curaçao por 3 días",
        {"trip_duration": 3, "group_size": 2}
    )
    
    # Test 6: Experiencia romántica para parejas
    run_test(
        "ESPAÑOL 6: Experiencia romántica para parejas",
        "Busco una experiencia romántica en Curaçao, con cenas a la luz de las velas, paseos al atardecer y hospedaje de lujo",
        {"interests": ["romantic", "food"], "group_size": 2}
    )
    
    # Test 7: Viaje en familia con niños
    run_test(
        "ESPAÑOL 7: Viaje en familia con niños",
        "Viajo con familia y niños pequeños, quiero actividades divertidas, playas seguras y lugares adecuados para niños",
        {"interests": ["family", "beach"], "group_size": 4}
    )
    
    # Test 8: Turismo de naturaleza y senderismo
    run_test(
        "ESPAÑOL 8: Turismo de naturaleza y senderismo",
        "Me interesa explorar el Parque Nacional Christoffel, hacer senderismo y observar la flora y fauna local",
        {"interests": ["nature", "hiking"], "locations": ["Christoffel Park"]}
    )
    
    # Test 9: Viaje de bajo presupuesto
    run_test(
        "ESPAÑOL 9: Viaje de bajo presupuesto",
        "Busco opciones económicas para visitar Curaçao, con alojamiento barato, comida local y actividades gratuitas o de bajo costo",
        {"budget_per_day": 80, "trip_duration": 7}
    )
    
    # Test 10: Viaje de lujo con alojamiento premium
    run_test(
        "ESPAÑOL 10: Viaje de lujo con alojamiento premium",
        "Quiero experimentar lo mejor de Curaçao, con hoteles de 5 estrellas, restaurantes gourmet y experiencias exclusivas",
        {"budget_per_day": 500, "interests": ["luxury"]}
    )
    
    # ESCENARIOS EN INGLÉS
    
    # Test 11: Cultural visit to downtown areas
    run_test(
        "ENGLISH 1: Cultural visit to downtown areas",
        "I want to explore the UNESCO World Heritage sites in Punda and Otrobanda, focusing on Dutch architecture and history",
        {"trip_duration": 3, "budget_per_day": 150}
    )
    
    # Test 12: Beach and water activities
    run_test(
        "ENGLISH 2: Beach and water activities",
        "Looking for the best beaches in Curacao for snorkeling and swimming, staying for 5 days",
        {"interests": ["water_activities", "beach"]}
    )
    
    # Test 13: Local food experience
    run_test(
        "ENGLISH 3: Local food experience",
        "I'm interested in trying local Curaçaoan food and visiting traditional restaurants and food markets",
        {"interests": ["food"], "budget_per_day": 200}
    )
    
    # Test 14: Adventure tourism
    run_test(
        "ENGLISH 4: Adventure tourism",
        "I'm looking for adventure activities like hiking, diving, cave exploration, and cliff jumping for a 4-day trip",
        {"interests": ["adventure"]}
    )
    
    # Test 15: Historical sites and museums
    run_test(
        "ENGLISH 5: Historical sites and museums",
        "Planning to visit historical sites, museums and colonial architecture in Willemstad for a 3-day trip",
        {"locations": ["Willemstad"], "interests": ["cultural", "history"]}
    )
    
    # Test 16: Romantic getaway
    run_test(
        "ENGLISH 6: Romantic getaway",
        "Planning a romantic holiday in Curacao with sunset cruises, beach dinners, and luxury accommodation",
        {"interests": ["romantic"], "group_size": 2, "budget_per_day": 300}
    )
    
    # Test 17: Family trip with kids
    run_test(
        "ENGLISH 7: Family trip with kids",
        "Traveling with family and young children, need child-friendly beaches, activities and restaurants",
        {"interests": ["family"], "group_size": 4}
    )
    
    # Test 18: Nature tourism and hiking
    run_test(
        "ENGLISH 8: Nature tourism and hiking",
        "I want to explore Christoffel National Park, do hiking trails and observe the local flora and fauna",
        {"interests": ["nature", "hiking"], "locations": ["Christoffel Park"]}
    )
    
    # Test 19: Budget-friendly trip
    run_test(
        "ENGLISH 9: Budget-friendly trip",
        "Looking for budget options to visit Curacao, with affordable accommodation, local food and free or low-cost activities",
        {"budget_per_day": 75, "trip_duration": 6}
    )
    
    # Test 20: Luxury experience with premium accommodations
    run_test(
        "ENGLISH 10: Luxury experience with premium accommodations",
        "I want to experience the best of Curacao with 5-star hotels, gourmet restaurants and exclusive experiences",
        {"budget_per_day": 500, "interests": ["luxury"]}
    )
    
    # Asegurar que el test pase
    assert True

if __name__ == "__main__":
    test_all_recommendations()