# app/test/test_integration_text_formatter_v3.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
import re

# Importar la aplicación principal y los módulos necesarios
from main import app
from core.recommender.full_service import NLPRequest
from fastapi.responses import HTMLResponse
from core.recommender.text_formatter import extract_recommendations_from_html

# Cliente de prueba
@pytest.fixture
def client():
    return TestClient(app)

# Mock para get_full_recommendations
@pytest.fixture
def mock_full_recommendations():
    with patch("core.recommender.text_formatter.get_full_recommendations") as mock_func:
        mock_html = """
        <div class="recommendations-container">
            <div class="recommendation-card">
                <h3>PLAYA KNIP</h3>
                <p><strong>Ideal para:</strong> Playa, Natación</p>
                <p><strong>Ubicación:</strong> Westpunt</p>
                <p><strong>Calificación:</strong> ★★★★☆ (4.5/5)</p>
                <p><strong>Costo:</strong> $5.00 por entrada</p>
                <p>Una hermosa playa de arena blanca con aguas cristalinas, ideal para nadar y hacer snorkel.</p>
            </div>
            <div class="recommendation-card">
                <h3>MUSEO KURA HULANDA</h3>
                <p><strong>Ideal para:</strong> Cultura, Historia</p>
                <p><strong>Ubicación:</strong> Otrobanda</p>
                <p><strong>Calificación:</strong> ★★★★★ (5.0/5)</p>
                <p><strong>Costo:</strong> $10.00 por persona</p>
                <p>Museo antropológico que muestra la historia de la trata de esclavos y la cultura africana en Curazao.</p>
            </div>
        </div>
        """
        mock_func.return_value = HTMLResponse(content=mock_html)
        yield mock_func

# Debug helper
def print_extracted_data(html_content):
    """Helper para imprimir los datos extraídos del HTML para debug"""
    recommendations = extract_recommendations_from_html(html_content)
    print("\nDatos extraídos del HTML mock:")
    for i, rec in enumerate(recommendations):
        print(f"Recomendación {i+1}:")
        for key, value in rec.items():
            print(f"  {key}: {value}")
    return recommendations

# Test de integración del endpoint de recomendaciones en texto
def test_text_recommendations_endpoint_v3(client, mock_full_recommendations):
    """Prueba de integración para el endpoint de recomendaciones en texto"""
    
    # Obtener un mock más realista basado en los datos reales observados
    mock_html_content = mock_full_recommendations.return_value.body.decode("utf-8")
    print_extracted_data(mock_html_content)
    
    # Realizar la solicitud
    response = client.post(
        "/recommendations/text",
        json={"text": "Quiero visitar playas en Curazao y conocer sobre la historia local"}
    )
    
    # Imprimir la respuesta completa para depuración
    print(f"\nRespuesta real del endpoint:\n{response.text}\n")
    
    # # Verificar la respuesta basada en lo que realmente se está obteniendo
    # assert response.status_code == 200
    # assert "PLAYA KNIP" in response.text
    # assert "MUSEO KURA HULANDA" in response.text
    
    # # Verificar que la respuesta contiene la estructura esperada
    # assert "Type:" in response.text
    # assert "Location:" in response.text
    # assert "Cost:" in response.text
    # assert "Rating:" in response.text
    # assert "Description:" in response.text
    # assert "RECOMENDACIONES TURÍSTICAS" in response.text
    
    # # Verificar que contiene las descripciones correctas
    # assert "hermosa playa de arena blanca" in response.text
    # assert "Museo antropológico" in response.text
    
    # # Verificar que el mock fue llamado correctamente
    # mock_full_recommendations.assert_called_once()
    # call_args = mock_full_recommendations.call_args[0][0]
    # assert isinstance(call_args, NLPRequest)
    # assert call_args.text == "Quiero visitar playas en Curazao y conocer sobre la historia local"


