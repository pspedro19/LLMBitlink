import openai
from typing import List, Dict

class OpenAIHelper:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        
    def generate_tour_guide_intro(self, recommendations: List[Dict], preferences: Dict) -> str:
        try:
            # Crear prompt con detalles relevantes
            prompt = self._create_prompt(recommendations, preferences)
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "system",
                    "content": "Eres un guía turístico experto en Curaçao"
                }, {
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return self._generate_fallback_intro(preferences)
            
    def _create_prompt(self, recommendations: List[Dict], preferences: Dict) -> str:
        return f"""
        Genera una introducción profesional y atractiva para estas recomendaciones turísticas:
        
        Destinos: {', '.join(preferences['locations'])}
        Duración: {preferences['trip_duration']} días
        Intereses: {', '.join(preferences['interests'])}
        Presupuesto: ${preferences['budget_per_day']} por día
        
        Principales recomendaciones:
        {self._format_recommendations_for_prompt(recommendations[:3])}
        """
        
    def _format_recommendations_for_prompt(self, recommendations: List[Dict]) -> str:
        return "\n".join([
            f"- {rec['name']}: {rec['description']}" 
            for rec in recommendations
        ])
        
    def _generate_fallback_intro(self, preferences: Dict) -> str:
        return f"¡Descubre las mejores experiencias en {' y '.join(preferences['locations'])}!"