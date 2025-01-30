from openai import OpenAI
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class OpenAIHelper:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
    def generate_tour_guide_intro(self, recommendations: List[Dict], preferences: Dict) -> str:
        """Genera una introducción para las recomendaciones turísticas."""
        try:
            # Crear prompt con detalles relevantes
            prompt = self._create_prompt(recommendations, preferences)
            
            return self._generate_completion(
                prompt,
                "Eres un guía turístico experto en Curaçao"
            )
            
        except Exception as e:
            logger.error(f"Error generating tour guide intro: {str(e)}")
            return self._generate_fallback_intro(preferences)
            
    def generate_tour_guide_response(
        self, 
        user_text: str, 
        system_message: Optional[str] = None,
        lang: str = 'es'
    ) -> str:
        """
        Genera una respuesta conversacional para consultas turísticas.
        
        Args:
            user_text: Texto del usuario
            system_message: Mensaje de sistema personalizado
            lang: Código de idioma ('es' o 'en')
                
        Returns:
            str: Respuesta generada por OpenAI
        """
        try:
            if not system_message:
                system_message = "Eres un guía turístico experto en Curaçao"
                if lang == 'en':
                    system_message = "You are an expert tour guide in Curaçao"
                    
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_text}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating tour guide response: {str(e)}")
            default_error = "Lo siento, no pude procesar tu solicitud en este momento. ¿Podrías intentarlo de nuevo?"
            return "Sorry, I couldn't process your request at this time. Could you try again?" if lang == 'en' else default_error
            
    def _generate_completion(self, prompt: str, system_message: str) -> str:
        """
        Método común para generar completaciones.
        
        Args:
            prompt: Prompt para OpenAI
            system_message: Mensaje de sistema para establecer el rol
            
        Returns:
            str: Respuesta generada por OpenAI
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        return response.choices[0].message.content
            
    def _create_prompt(self, recommendations: List[Dict], preferences: Dict) -> str:
        """
        Crea un prompt para generar la introducción de las recomendaciones.
        
        Args:
            recommendations: Lista de recomendaciones
            preferences: Preferencias del usuario
            
        Returns:
            str: Prompt formateado
        """
        return f"""
        Genera una introducción profesional y atractiva para estas recomendaciones turísticas:
        
        Destinos: {', '.join(preferences['locations'])}
        Duración: {preferences['trip_duration']} días
        Intereses: {', '.join(preferences['interests'])}
        Presupuesto: ${preferences.get('budget_per_day', 'flexible')} por día
        
        Principales recomendaciones:
        {self._format_recommendations_for_prompt(recommendations[:3])}
        """
        
    def _format_recommendations_for_prompt(self, recommendations: List[Dict]) -> str:
        """
        Formatea las recomendaciones para incluirlas en el prompt.
        
        Args:
            recommendations: Lista de recomendaciones a formatear
            
        Returns:
            str: Recomendaciones formateadas
        """
        return "\n".join([
            f"- {rec['name']}: {rec['description']}" 
            for rec in recommendations
        ])
        
    def _generate_fallback_intro(self, preferences: Dict) -> str:
        """
        Genera una introducción de respaldo en caso de error.
        
        Args:
            preferences: Preferencias del usuario
            
        Returns:
            str: Mensaje de respaldo
        """
        return f"¡Descubre las mejores experiencias en {' y '.join(preferences['locations'])}!"