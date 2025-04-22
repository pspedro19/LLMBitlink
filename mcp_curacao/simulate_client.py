#!/usr/bin/env python3
"""
Cliente simulado para el asistente de Curaçao
Este cliente no se conecta a servidores MCP, pero simula la experiencia
"""

import os
import sys
import time
import json
import shutil
import random

# Obtener ancho del terminal
term_width, _ = shutil.get_terminal_size()

# Colores ANSI
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
BLUE = "\033[34m"
CYAN = "\033[36m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"

# Estado del cliente simulado
user_context = {
    "session_id": "sim-" + str(random.randint(10000, 99999)),
    "preferences": {},
    "message_history": [],
    "recommendations": []
}

def print_header():
    """Imprime el encabezado del cliente"""
    title = "Asistente Turístico de Curaçao - Simulación"
    print("\n" + "=" * term_width)
    print(f"{BOLD}{GREEN}{title.center(term_width)}{RESET}")
    print("=" * term_width)
    
    print(f"\n{CYAN}¡Bienvenido al asistente turístico de Curaçao!{RESET}")
    print("Esta es una simulación del asistente que emula la integración MCP+RAG.")
    print("Puedo ayudarte a planificar tu viaje a Curaçao con recomendaciones personalizadas.")
    
    print("\nEscribe tu consulta y presiona Enter. Escribe '/salir' para terminar.")
    print("=" * term_width + "\n")

def extract_preferences(message):
    """Extrae preferencias del mensaje del usuario"""
    preferences = {}
    
    # Detectar intereses
    interest_keywords = {
        "cultural": ["cultura", "museo", "historia", "arte", "patrimonio"],
        "natural": ["playa", "naturaleza", "parque", "buceo", "snorkel"],
        "family": ["familia", "niños", "diversión", "actividades para niños"],
        "gastronomy": ["comida", "restaurante", "gastronomía", "comer", "cocina"],
        "nightlife": ["fiesta", "discoteca", "club", "bar", "noche", "música"]
    }
    
    interests = []
    for interest, keywords in interest_keywords.items():
        for keyword in keywords:
            if keyword.lower() in message.lower():
                interests.append(interest)
                break
    
    if interests:
        preferences["interests"] = list(set(interests))  # Eliminar duplicados
    
    # Detectar duración
    import re
    duration_pattern = r'(\d+)\s*(día|dias|day|days)'
    duration_match = re.search(duration_pattern, message.lower())
    if duration_match:
        preferences["duration"] = float(duration_match.group(1))
    
    # Detectar presupuesto
    budget_pattern = r'(\d+)\s*(dólares|dolares|dollars|\$|euros|euro)'
    budget_match = re.search(budget_pattern, message.lower())
    if budget_match:
        preferences["budget"] = float(budget_match.group(1))
    
    # Detectar ubicaciones
    location_keywords = ["willemstad", "punda", "otrobanda", "curaçao", "curacao"]
    locations = []
    for location in location_keywords:
        if location.lower() in message.lower():
            locations.append(location)
    
    if locations:
        preferences["locations"] = locations
    
    return preferences

def update_user_context(preferences):
    """Actualiza el contexto del usuario con nuevas preferencias"""
    for key, value in preferences.items():
        user_context["preferences"][key] = value

def determine_missing_information():
    """Determina qué información falta para generar recomendaciones"""
    missing = []
    required_fields = ["interests", "duration", "budget"]
    
    for field in required_fields:
        if field not in user_context["preferences"] or not user_context["preferences"][field]:
            missing.append(field)
    
    return missing

def generate_recommendations():
    """Genera recomendaciones simuladas basadas en preferencias"""
    prefs = user_context["preferences"]
    
    # Verificar si tenemos suficientes preferencias
    if not prefs.get("interests") or not prefs.get("duration") or not prefs.get("budget"):
        return []
    
    # Recomendaciones simuladas por categoría
    all_recommendations = {
        "cultural": [
            {
                "id": "c1",
                "name": "Museo Kura Hulanda",
                "type": "attraction",
                "description": "Un museo antropológico que documenta la historia de la esclavitud en el Caribe.",
                "location": "Willemstad",
                "rating": 4.5,
                "cost": 45
            },
            {
                "id": "c2",
                "name": "Sinagoga Mikvé Israel-Emanuel",
                "type": "attraction",
                "description": "La sinagoga más antigua del hemisferio occidental, con piso de arena.",
                "location": "Willemstad",
                "rating": 4.8,
                "cost": 20
            },
            {
                "id": "c3",
                "name": "Recorrido a pie por Punda",
                "type": "activity",
                "description": "Tour guiado por el distrito histórico de Punda, con edificios coloniales.",
                "location": "Willemstad",
                "rating": 4.6,
                "cost": 35
            }
        ],
        "natural": [
            {
                "id": "n1",
                "name": "Playa Kenepa (Grote Knip)",
                "type": "beach",
                "description": "Una de las playas más hermosas de Curaçao, con aguas turquesas.",
                "location": "Westpunt",
                "rating": 4.9,
                "cost": 0
            },
            {
                "id": "n2",
                "name": "Parque Nacional Shete Boka",
                "type": "park",
                "description": "Parque natural con formaciones rocosas impresionantes y olas rompiendo.",
                "location": "Banda Abou",
                "rating": 4.7,
                "cost": 15
            },
            {
                "id": "n3",
                "name": "Tour de Snorkel en Tugboat",
                "type": "activity",
                "description": "Snorkel alrededor de un remolcador hundido con colorida vida marina.",
                "location": "Caracas Bay",
                "rating": 4.8,
                "cost": 50
            }
        ],
        "gastronomy": [
            {
                "id": "g1",
                "name": "Plasa Bieu (Mercado Viejo)",
                "type": "restaurant",
                "description": "Mercado de comida local con auténtica cocina de Curaçao.",
                "location": "Willemstad",
                "rating": 4.6,
                "cost": 25
            },
            {
                "id": "g2",
                "name": "Restaurante Gouverneur de Rouville",
                "type": "restaurant",
                "description": "Restaurante con vistas panorámicas y cocina caribeña de alta calidad.",
                "location": "Otrobanda",
                "rating": 4.5,
                "cost": 65
            }
        ],
        "nightlife": [
            {
                "id": "nl1",
                "name": "Mambo Beach Boulevard",
                "type": "nightlife",
                "description": "Complejo de playa, bares y tiendas con música en vivo y eventos nocturnos.",
                "location": "Willemstad",
                "rating": 4.3,
                "cost": 30
            },
            {
                "id": "nl2",
                "name": "Wet & Wild Beach Club",
                "type": "nightlife",
                "description": "Club de playa con DJs internacionales y fiestas temáticas.",
                "location": "Jan Thiel",
                "rating": 4.2,
                "cost": 40
            }
        ]
    }
    
    # Seleccionar recomendaciones basadas en intereses
    recommendations = []
    for interest in prefs.get("interests", []):
        if interest in all_recommendations:
            recommendations.extend(all_recommendations[interest])
    
    # Filtrar por presupuesto (asumiendo 20% del presupuesto diario para cada actividad)
    daily_budget = prefs.get("budget", 0) / prefs.get("duration", 1)
    activity_budget = daily_budget * 0.2
    recommendations = [rec for rec in recommendations if rec.get("cost", 0) <= activity_budget]
    
    # Ordenar por calificación
    recommendations.sort(key=lambda x: x.get("rating", 0), reverse=True)
    
    # Limitar número de recomendaciones
    return recommendations[:5]

def enrich_with_rag(recommendations):
    """Simula enriquecer recomendaciones con información RAG"""
    if not recommendations:
        return recommendations
    
    # Información RAG simulada por lugar
    rag_info = {
        "Museo Kura Hulanda": "El Museo Kura Hulanda es un museo antropológico en Willemstad que documenta la historia de la trata de esclavos. Fundado en 1999, el museo contiene una amplia colección de artefactos históricos y exhibiciones interactivas.",
        
        "Sinagoga Mikvé Israel-Emanuel": "Construida en 1732, esta sinagoga es la más antigua en uso continuo en el hemisferio occidental. Es famosa por su suelo de arena, que simboliza el desierto por el que los judíos vagaron durante 40 años y también servía para amortiguar el sonido durante los servicios secretos en tiempos de persecución.",
        
        "Recorrido a pie por Punda": "Punda es uno de los cuatro distritos históricos de Willemstad y es conocido por sus edificios coloniales holandeses de colores pastel. El recorrido a pie incluye visitas a la Plaza Gomez, el Puente Emma, y varias tiendas y edificios históricos.",
        
        "Playa Kenepa (Grote Knip)": "Considerada una de las playas más hermosas de Curaçao, Grote Knip ofrece aguas cristalinas turquesas y arena blanca. Es ideal para nadar y hacer snorkel, con arrecifes cercanos a la orilla llenos de peces tropicales.",
        
        "Parque Nacional Shete Boka": "Ubicado en la costa norte de la isla, este parque presenta siete calas donde el mar rompe contra acantilados escarpados, creando espectaculares explosiones de espuma. Es hogar de tres especies de tortugas marinas que anidan en sus playas remotas.",
        
        "Plasa Bieu (Mercado Viejo)": "Este mercado tradicional ofrece auténtica comida local en un ambiente animado. Los visitantes pueden disfrutar de platos como Karko (caracol de mar), Kabritu Stoba (guiso de cabra) y Funchi (similar a la polenta)."
    }
    
    # Agregar información RAG a las recomendaciones
    for rec in recommendations:
        if rec["name"] in rag_info:
            rec["rag_info"] = rag_info[rec["name"]]
    
    return recommendations

def generate_response():
    """Genera una respuesta basada en el estado actual"""
    # Verificar si falta información
    missing_info = determine_missing_information()
    
    if "interests" in missing_info:
        return "¿Qué tipo de actividades te interesarían en Curaçao? Por ejemplo, ¿prefieres experiencias culturales, playas, aventuras, gastronomía o vida nocturna?"
    
    if "duration" in missing_info:
        return "¿Por cuántos días planeas visitar Curaçao? Esto me ayudará a organizar mejor las recomendaciones."
    
    if "budget" in missing_info:
        duration = user_context["preferences"].get("duration", 0)
        return f"¿Tienes un presupuesto aproximado para tu viaje de {duration} días? Esto me ayudará a sugerirte opciones adecuadas."
    
    # Si no falta información, generar recomendaciones
    recommendations = generate_recommendations()
    if not recommendations:
        return "Estoy procesando tus preferencias. ¿Podrías darme más detalles sobre lo que buscas en Curaçao?"
    
    # Enriquecer con RAG
    recommendations = enrich_with_rag(recommendations)
    user_context["recommendations"] = recommendations
    
    # Generar respuesta con recomendaciones
    response = "Basado en tus preferencias, te recomiendo estas opciones en Curaçao:\n\n"
    
    for i, rec in enumerate(recommendations, 1):
        response += f"{i}. **{rec['name']}**: {rec['description']}\n"
        if "rating" in rec:
            response += f"   Rating: {rec['rating']}★ | "
        if "cost" in rec:
            response += f"Costo aproximado: ${rec['cost']}\n"
        if "rag_info" in rec:
            response += f"   *{rec['rag_info']}*\n"
        response += "\n"
    
    response += "¿Te gustaría más información sobre alguna de estas opciones?"
    
    return response

def main():
    """Función principal del cliente simulado"""
    print_header()
    
    # Loop principal
    while True:
        try:
            # Solicitar entrada
            user_input = input(f"\n{BOLD}{GREEN}Tú:{RESET} ")
            
            # Verificar si es comando de salida
            if user_input.lower() in ['/salir', '/exit', '/quit']:
                print(f"\n{GREEN}¡Gracias por usar el asistente turístico de Curaçao! ¡Hasta pronto!{RESET}\n")
                break
            
            # Verificar si está vacío
            if not user_input.strip():
                continue
            
            # Almacenar mensaje en historial
            user_context["message_history"].append({
                "sender": "user",
                "message": user_input,
                "timestamp": time.time()
            })
            
            # Procesar mensaje
            print(f"{DIM}Procesando tu consulta...{RESET}")
            time.sleep(0.5)  # Simular procesamiento
            
            # Extraer preferencias
            preferences = extract_preferences(user_input)
            
            # Actualizar contexto
            update_user_context(preferences)
            
            # Generar respuesta
            response = generate_response()
            
            # Almacenar respuesta en historial
            user_context["message_history"].append({
                "sender": "assistant",
                "message": response,
                "timestamp": time.time()
            })
            
            # Mostrar respuesta
            print(f"\n{BOLD}{BLUE}Asistente:{RESET}")
            for line in response.split("\n"):
                print(f"    {line}")
            
        except KeyboardInterrupt:
            print(f"\n\n{GREEN}¡Hasta pronto!{RESET}\n")
            break
        except Exception as e:
            print(f"\n{RED}Error inesperado: {str(e)}{RESET}\n")
            continue

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{GREEN}¡Hasta pronto!{RESET}\n")
    except Exception as e:
        print(f"\n{RED}Error inesperado: {str(e)}{RESET}\n")