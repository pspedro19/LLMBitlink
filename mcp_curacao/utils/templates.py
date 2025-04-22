# mcp_curacao/utils/templates.py
import random

class ResponseTemplates:
    """Plantillas para respuestas naturales y conversacionales"""
    
    def __init__(self):
        """Inicializa las plantillas de respuesta"""
        self._init_greeting_templates()
        self._init_question_templates()
        self._init_recommendation_templates()
        self._init_follow_up_templates()
        self._init_itinerary_templates()
    
    def _init_greeting_templates(self):
        """Inicializa plantillas de saludo"""
        self.greeting_templates = [
            "¡Hola! Soy tu asistente turístico de Curaçao. ¿En qué puedo ayudarte hoy?",
            "¡Bienvenido! Estoy aquí para ayudarte a planificar tu viaje a Curaçao. ¿Qué te gustaría saber?",
            "¡Hola! ¿Cómo puedo ayudarte con tu visita a Curaçao?",
            "¡Bienvenido al asistente de viajes de Curaçao! ¿En qué puedo asistirte?"
        ]
    
    def _init_question_templates(self):
        """Inicializa plantillas de preguntas para obtener información"""
        self.interest_questions = [
            "¿Qué tipo de actividades te interesarían en Curaçao? Por ejemplo, ¿prefieres experiencias culturales, playas, aventuras, gastronomía o vida nocturna?",
            "Para personalizar tus recomendaciones, ¿qué tipo de experiencias buscas en Curaçao? ¿Te interesan los aspectos culturales, naturales, culinarios o de entretenimiento?",
            "¿Qué te gustaría experimentar durante tu visita a Curaçao? ¿Te atraen más las actividades culturales, acuáticas, gastronómicas o de ocio nocturno?"
        ]
        
        self.duration_questions = [
            "¿Por cuántos días planeas visitar Curaçao? Esto me ayudará a organizar mejor las recomendaciones.",
            "¿Cuánto tiempo estarás en Curaçao? Así podré sugerirte un itinerario adecuado.",
            "¿Cuál es la duración de tu estancia en Curaçao? Esto es importante para ajustar mis recomendaciones."
        ]
        
        self.budget_questions = [
            "¿Tienes un presupuesto aproximado para tu viaje de {duration} días? Esto me ayudará a sugerirte opciones adecuadas.",
            "Para poder recomendarte opciones que se ajusten a tus necesidades, ¿podrías indicarme un presupuesto aproximado para tu estancia de {duration} días?",
            "¿Cuál es tu presupuesto para estos {duration} días en Curaçao? Así podré filtrar las mejores opciones para ti."
        ]
        
        self.fallback_responses = [
            "Entiendo. ¿Hay algo específico sobre Curaçao que te gustaría saber?",
            "Gracias por la información. ¿Hay algún aspecto particular de Curaçao que te interese?",
            "Perfecto. ¿Hay alguna otra preferencia que deba tener en cuenta para tus recomendaciones?"
        ]
        
        self.processing_messages = [
            "¡Perfecto! Estoy buscando las mejores recomendaciones para ti. Dame un momento...",
            "Gracias por la información. Estoy preparando algunas sugerencias personalizadas para ti.",
            "¡Excelente! Con estos datos puedo encontrar opciones que te encantarán. Un momento por favor..."
        ]
    
    def _init_recommendation_templates(self):
        """Inicializa plantillas para recomendaciones"""
        self.recommendation_intros = [
            "Basado en tus preferencias, te recomiendo estas opciones en Curaçao:",
            "He encontrado estas recomendaciones que creo que te encantarán:",
            "Considerando lo que buscas, estas son mis mejores sugerencias para tu visita a Curaçao:",
            "Para tu viaje a Curaçao, estas opciones se ajustan perfectamente a lo que estás buscando:"
        ]
        
        self.recommendation_item_templates = [
            "**{name}**: {description}",
            "**{name}** - {rating}★: {description}",
            "**{name}** ({location}): {description}"
        ]
        
        self.recommendation_endings = [
            "¿Te gustaría más información sobre alguna de estas opciones?",
            "¿Alguna de estas sugerencias te llama especialmente la atención?",
            "Si quieres detalles adicionales sobre cualquiera de estas recomendaciones, no dudes en preguntarme.",
            "¿Necesitas más información o tienes otras preguntas sobre tu viaje a Curaçao?"
        ]
    
    def _init_follow_up_templates(self):
        """Inicializa plantillas para preguntas de seguimiento"""
        self.follow_up_questions = {
            "general": [
                "¿Hay algún tipo específico de cocina que te gustaría probar en Curaçao?",
                "¿Te interesaría conocer sobre las mejores playas de Curaçao?",
                "¿Quieres información sobre opciones de transporte en la isla?",
                "¿Te gustaría conocer más sobre la historia y cultura de Curaçao?"
            ],
            "cultural": [
                "¿Te interesaría una visita guiada por el centro histórico de Willemstad?",
                "¿Quieres saber más sobre los museos y sitios patrimoniales de Curaçao?",
                "¿Te gustaría conocer sobre el arte local y las galerías de Curaçao?"
            ],
            "natural": [
                "¿Te interesaría hacer snorkel o buceo en los arrecifes de coral?",
                "¿Quieres información sobre parques naturales o reservas en Curaçao?",
                "¿Te gustaría saber cuáles son las playas menos concurridas?"
            ],
            "gastronomy": [
                "¿Prefieres restaurantes de lujo o experiencias más auténticas y locales?",
                "¿Te interesaría probar la cocina tradicional de Curaçao?",
                "¿Quieres recomendaciones de restaurantes con las mejores vistas?"
            ],
            "nightlife": [
                "¿Buscas lugares para bailar o prefieres bares más tranquilos?",
                "¿Te interesarían eventos de música en vivo durante tu estancia?",
                "¿Quieres información sobre casinos y entretenimiento nocturno?"
            ]
        }
    
    def _init_itinerary_templates(self):
        """Inicializa plantillas para itinerarios."""
        self.itinerary_intros = [
            "Aquí tienes un itinerario sugerido para tu viaje a Curaçao:",
            "He preparado este itinerario basado en tus preferencias:",
            "Para aprovechar al máximo tu estancia en Curaçao, te propongo el siguiente itinerario:",
            "Considerando tus intereses, este es un itinerario que podría encantarte:"
        ]
        
        self.itinerary_day_intros = [
            "**Día {day}:**",
            "**Día {day} - {weekday}:**",
            "**Jornada {day}:**"
        ]
        
        self.itinerary_conclusions = [
            "Este itinerario está diseñado para maximizar tu experiencia en Curaçao. ¿Te gustaría hacer algún cambio?",
            "¿Qué te parece este plan? Podemos ajustarlo según tus preferencias.",
            "Este es un plan flexible que puedes adaptar según el clima y tu energía. ¿Necesitas más detalles sobre alguna actividad?",
            "¿Hay alguna parte de este itinerario sobre la que te gustaría más información?"
        ]
    
    def get_greeting(self) -> str:
        """Devuelve un saludo aleatorio"""
        return random.choice(self.greeting_templates)
    
    def get_interests_question(self) -> str:
        """Devuelve una pregunta para obtener intereses"""
        return random.choice(self.interest_questions)
    
    def get_duration_question(self) -> str:
        """Devuelve una pregunta para obtener duración"""
        return random.choice(self.duration_questions)
    
    def get_budget_question(self, duration: int) -> str:
        """Devuelve una pregunta para obtener presupuesto"""
        question = random.choice(self.budget_questions)
        return question.format(duration=duration if duration else "")
    
    def get_fallback_message(self) -> str:
        """Devuelve un mensaje genérico cuando no hay suficiente contexto"""
        return random.choice(self.fallback_responses)
    
    def get_processing_message(self) -> str:
        """Devuelve un mensaje de procesamiento mientras se buscan recomendaciones"""
        return random.choice(self.processing_messages)
    
    def format_recommendations(self, recommendations, rag_info=None) -> str:
        """
        Formatea recomendaciones en texto conversacional.
        
        Args:
            recommendations: Lista de recomendaciones
            rag_info: Información RAG por recomendación
            
        Returns:
            Texto formateado con recomendaciones
        """
        if not recommendations:
            return "Lo siento, no he podido encontrar recomendaciones que se ajusten a tus preferencias. ¿Podrías darme más detalles sobre lo que buscas?"
        
        # Seleccionar una introducción aleatoria
        intro = random.choice(self.recommendation_intros)
        
        # Construir el cuerpo de las recomendaciones
        body = ""
        for i, rec in enumerate(recommendations[:5], 1):
            name = rec.get("name", "")
            description = rec.get("description", "Sin descripción disponible")
            location = rec.get("location", "Curaçao")
            rating = rec.get("rating", "")
            
            # Limitar descripción a longitud razonable
            if len(description) > 150:
                description = description[:147] + "..."
            
            # Seleccionar plantilla para este ítem
            item_template = random.choice(self.recommendation_item_templates)
            item_text = item_template.format(
                name=name,
                description=description,
                location=location,
                rating=rating
            )
            
            body += f"{i}. {item_text}\n\n"
            
            # Agregar información RAG si está disponible
            if "rag_info" in rec:
                rag_text = rec["rag_info"]
                if len(rag_text) > 200:
                    rag_text = rag_text[:197] + "..."
                body += f"   *{rag_text}*\n\n"
        
        # Agregar cierre
        ending = random.choice(self.recommendation_endings)
        
        # Compilar respuesta completa
        full_response = f"{intro}\n\n{body}{ending}"
        return full_response
    
    def format_itinerary(self, duration, grouped_recommendations, start_date=None):
        """
        Formatea un itinerario basado en recomendaciones agrupadas.
        
        Args:
            duration: Duración en días
            grouped_recommendations: Recomendaciones agrupadas por tipo
            start_date: Fecha de inicio (opcional)
            
        Returns:
            Texto formateado con el itinerario
        """
        # Extraer recomendaciones por grupo
        activities = grouped_recommendations.get("activities", [])
        spots = grouped_recommendations.get("spots", [])
        restaurants = grouped_recommendations.get("restaurants", [])
        nightlife = grouped_recommendations.get("nightlife", [])
        
        # Generar itinerario
        intro = random.choice(self.itinerary_intros)
        response = f"{intro}\n\n"
        
        # Generar día a día
        for day in range(1, int(duration) + 1):
            day_intro = random.choice(self.itinerary_day_intros).format(
                day=day, 
                weekday="" # Sería necesario calcular el día de la semana si se proporciona start_date
            )
            response += f"{day_intro}\n\n"
            
            # Actividad de mañana
            if activities and len(activities) > (day-1) % max(1, len(activities)):
                act = activities[(day-1) % len(activities)]
                response += f"- **Mañana:** {act.get('name')} - {act.get('description', '')[:100]}...\n"
            elif spots and len(spots) > (day-1) % max(1, len(spots)):
                spot = spots[(day-1) % len(spots)]
                response += f"- **Mañana:** Visita a {spot.get('name')} - {spot.get('description', '')[:100]}...\n"
            
            # Almuerzo
            if restaurants and len(restaurants) > (day-1) % max(1, len(restaurants)):
                rest = restaurants[(day-1) % len(restaurants)]
                response += f"- **Almuerzo:** {rest.get('name')}\n"
            
            # Actividad de tarde
            offset = day % max(1, len(spots) if spots else 1)
            if spots and len(spots) > offset:
                spot = spots[offset]
                response += f"- **Tarde:** Visita a {spot.get('name')} - {spot.get('description', '')[:100]}...\n"
            elif activities and len(activities) > offset:
                act = activities[offset]
                response += f"- **Tarde:** {act.get('name')} - {act.get('description', '')[:100]}...\n"
            
            # Cena y noche
            if restaurants and len(restaurants) > (day+1) % max(1, len(restaurants)):
                rest = restaurants[(day+1) % len(restaurants)]
                response += f"- **Cena:** {rest.get('name')}\n"
            
            # Actividad nocturna
            if day % 2 == 0 and nightlife and len(nightlife) > 0:  # Cada dos días
                night = nightlife[day % len(nightlife)]
                response += f"- **Noche:** {night.get('name')} - {night.get('description', '')[:100]}...\n"
            
            response += "\n"
        
        # Añadir conclusión
        response += random.choice(self.itinerary_conclusions)
        
        return response
    
    def get_follow_up_questions(self, interests) -> list:
        """
        Genera preguntas de seguimiento contextuales.
        
        Args:
            interests: Lista de intereses del usuario
            
        Returns:
            Lista de posibles preguntas de seguimiento
        """
        questions = []
        
        # Agregar preguntas basadas en intereses detectados
        for interest in interests:
            mapped_interest = None
            if interest == "cultural":
                mapped_interest = "cultural"
            elif interest in ["natural", "playa"]:
                mapped_interest = "natural"
            elif interest in ["gastronomy", "comida"]:
                mapped_interest = "gastronomy"
            elif interest in ["nightlife", "fiesta"]:
                mapped_interest = "nightlife"
            
            if mapped_interest and mapped_interest in self.follow_up_questions:
                questions.extend(self.follow_up_questions[mapped_interest])
        
        # Si no hay intereses específicos o hay pocos, agregar preguntas generales
        if len(questions) < 3:
            questions.extend(self.follow_up_questions["general"])
        
        # Limitar a 3 preguntas y mezclar
        questions = list(set(questions))  # Eliminar duplicados
        random.shuffle(questions)
        return questions[:3]