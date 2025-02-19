import pytest
import asyncio
from typing import Dict, Any
import re

class TestTemplateRendering:
    @pytest.fixture
    async def chat_agent_with_preferences(self, chat_agent):
        """Fixture que prepara un agente con preferencias establecidas"""
        preferences = [
            "Somos 4 personas",
            "Presupuesto de $5000",
            "7 días de estancia",
            "Sin restricciones alimenticias"
        ]
        
        for pref in preferences:
            await chat_agent.invoke(pref, "test_templates")
            
        return chat_agent

    async def test_response_formatting(self, chat_agent_with_preferences):
        """Test de formato de respuestas"""
        test_inputs = [
            "Hola, quiero información",
            "¿Qué playas recomiendan?",
            "Me interesa reservar"
        ]
        
        for input_text in test_inputs:
            response = await chat_agent_with_preferences.invoke(
                input_text,
                "test_format"
            )
            
            # Verificar estructura básica
            assert "🏝️" in response["response"]  # Tiene emojis
            assert "Curaçao" in response["response"]  # Menciona el destino
            
            # Verificar formato
            paragraphs = response["response"].split("\n\n")
            assert len(paragraphs) >= 2  # Al menos dos párrafos
            
            # Verificar capitalización
            sentences = re.split(r'[.!?]+', response["response"])
            for sentence in sentences:
                if sentence.strip():
                    assert sentence.strip()[0].isupper()

    async def test_template_variables(self, chat_agent_with_preferences):
        """Test de sustitución de variables en templates"""
        response = await chat_agent_with_preferences.invoke(
            "¿Qué actividades recomiendan?",
            "test_variables"
        )
        
        # Verificar que no hay variables sin reemplazar
        assert "{" not in response["response"]
        assert "}" not in response["response"]
        
        # Verificar que se incluyeron las preferencias
        response_text = response["response"].lower()
        assert "4 personas" in response_text
        assert "$5000" in response_text
        assert "7 días" in response_text

    async def test_neuro_technique_application(self, chat_agent):
        """Test de aplicación de técnicas de neuroventas"""
        # Probar cada técnica
        techniques = {
            "scarcity": "Últimos lugares disponibles",
            "social_proof": "Más de 5,000 viajeros",
            "urgency": "Oferta especial"
        }
        
        for technique, expected_text in techniques.items():
            response = await chat_agent.invoke(
                f"Me interesa el tour de {technique}",
                f"test_neuro_{technique}"
            )
            
            assert expected_text in response["response"]

    async def test_emotional_responses(self, chat_agent):
        """Test de respuestas emocionales apropiadas"""
        test_cases = [
            {
                "input": "¡Me encanta la idea!",
                "sentiment": "POSITIVE",
                "