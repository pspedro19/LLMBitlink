from typing import Dict, List, Any
import logging
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

class HTMLFormatter:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)
           
        self.template = """
            <div class="recommendation-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px; max-width: 400px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
                <img src="{image_url}" alt="Imagen de {name}" style="width: 100%; border-radius: 8px 8px 0 0;">
                <div style="padding: 12px;">
                    <h3 style="margin: 0 0 8px; font-size: 1.5em; color: #333;">{name}</h3>
                    <p style="margin: 0 0 8px; color: #555;"><strong>Ubicación:</strong> {location}</p>
                    <p style="margin: 0 0 8px; color: #555;">{description}</p>
                    <p style="margin: 0 0 8px; color: #555;"><strong>Calificación:</strong> {rating}</p>
                    <p style="margin: 0 0 8px; color: #555;"><strong>Costo:</strong> ${cost}</p>
                    <p style="margin: 0 0 8px; color: #555;"><strong>Relevancia:</strong> {relevance}</p>
                    <p style="margin: 0 0 8px; color: #555;"><strong>Horario:</strong> {opening_hours}</p>
                    <p style="margin: 0 0 8px; color: #555;"><strong>Ideal para:</strong> {ideal_for}</p>
                    {facilities}
                </div>
            </div>
        """
       
    def format_to_html(self, recommendations: List[Dict], preferences: Dict) -> str:
        intro_text = self._generate_ai_intro(recommendations, preferences) if self.openai_api_key else self._generate_intro(preferences)
        cards = [self._format_card(rec) for rec in recommendations]
        
        return f"""
            <div>
                <p style="font-size: 1.2em; color: #333; margin-bottom: 16px;">{intro_text}</p>
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px;">
                    {''.join(cards)}
                </div>
            </div>
        """

    def _generate_ai_intro(self, recommendations: List[Dict], preferences: Dict) -> str:
        try:
            prompt = f"""Genera una introducción profesional y atractiva en español para un guía turístico de Curaçao. Detalles:
            - Destinos: {', '.join(preferences['locations'])}
            - Duración: {preferences['trip_duration']} días
            - Grupo: {preferences.get('group_size', 1)} personas
            - Intereses: {', '.join(preferences['interests'])}
            - Presupuesto: ${preferences.get('budget_per_day', 0)} por día"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "system",
                    "content": "Eres un experto guía turístico de Curaçao."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating AI intro: {str(e)}")
            return self._generate_intro(preferences)
           
    def _format_card(self, rec: Dict) -> str:
        facilities_list = self._format_facilities(rec)
        return self.template.format(
            image_url=self._get_image_url(rec),
            name=rec["name"],
            location=rec["location"],
            description=rec.get("description", ""),
            rating=self._format_rating(rec.get("rating", 0)),
            cost=self._format_cost(rec),
            relevance=self._calculate_relevance(rec["_scores"]["total"]),
            opening_hours=rec.get("opening_hours", "No disponible"),
            ideal_for=rec.get("recommended_for", rec.get("ideal_for", "Todos los visitantes")),
            facilities=facilities_list
        )

    def _format_facilities(self, rec: Dict) -> str:
        facilities = []
        if rec.get("accessibility"):
            facilities.append(f"<strong>Accesibilidad:</strong> {rec['accessibility']}")
        if rec.get("parking"):
            facilities.append(f"<strong>Estacionamiento:</strong> {rec['parking']}")
        if rec.get("payment_options"):
            facilities.append(f"<strong>Formas de pago:</strong> {rec['payment_options']}")
            
        if facilities:
            return '<div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;">' + \
                    '<p style="margin: 0 0 8px; color: #555;">' + \
                    ' | '.join(facilities) + '</p></div>'
        return ""

    def _get_image_url(self, rec: Dict) -> str:
        return "https://via.placeholder.com/400"

    def _format_rating(self, rating: float) -> str:
        rating = float(rating)
        stars = '★' * int(rating) + '☆' * (5 - int(rating))
        return f"{stars} ({rating:.1f}/5)"

    def _format_cost(self, rec: Dict) -> str:
        cost = rec.get("cost", rec.get("entry_fee", rec.get("average_person_expense", 0)))
        if cost == 0:
            return "Gratis"
        return f"{cost:.2f}"

    def _calculate_relevance(self, score: float) -> str:
        if score > 0.8:
            return "Muy Alta"
        elif score > 0.6:
            return "Alta"
        elif score > 0.4:
            return "Media"
        return "Baja"
        
    def _generate_intro(self, preferences: Dict) -> str:
        locations = preferences.get("locations", [])
        duration = preferences.get("trip_duration", 0)
        group_size = preferences.get("group_size", 1)
        interests = preferences.get("interests", [])
        
        intro = f"¡Descubre las mejores experiencias en {' y '.join(locations)}! "
        intro += f"Hemos seleccionado estas recomendaciones para tu viaje de {duration} días"
        
        if group_size > 1:
            intro += f" para un grupo de {group_size} personas"
        
        if interests:
            intro += f", enfocándonos en {', '.join(interests)}"
            
        intro += "."
        return intro