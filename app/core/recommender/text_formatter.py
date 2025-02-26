# app/core/recommender/text_formatter.py
from typing import Dict, Any, List
import logging
from bs4 import BeautifulSoup
import re
from fastapi import HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

# Importar el servicio existente
from app.core.recommender.full_service import get_full_recommendations, NLPRequest

# Configurar logging
logger = logging.getLogger(__name__)

class TextFormattedRequest(BaseModel):
    text: str = Field(..., description="Natural language input text")

async def get_text_formatted_recommendations(request: TextFormattedRequest) -> PlainTextResponse:
    """
    Endpoint que combina el análisis NLP con generación de recomendaciones en formato texto plano
    """
    try:
        # Utilizar el servicio existente para obtener recomendaciones en HTML
        html_response = await get_full_recommendations(NLPRequest(text=request.text))
        html_content = html_response.body.decode('utf-8')
        
        # Extraer las recomendaciones del HTML
        recommendations = extract_recommendations_from_html(html_content)
        
        # Aplicar el nuevo formato de texto
        formatted_text = format_recommendations_list(recommendations)
        
        return PlainTextResponse(content=formatted_text)
        
    except Exception as e:
        logger.error(f"Error processing text formatted recommendation request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def extract_recommendations_from_html(html_content: str) -> List[Dict[str, Any]]:
    """
    Extrae las recomendaciones del HTML generado por el servicio original
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    recommendation_cards = soup.find_all('div', class_='recommendation-card')
    
    recommendations = []
    for card in recommendation_cards:
        # Extraer la información de cada tarjeta
        name = card.find('h3').text.strip() if card.find('h3') else "UNNAMED"
        
        # Extraer ubicación
        location_p = card.find('p', string=lambda s: 'Ubicación' in s if s else False)
        location = location_p.text.replace('Ubicación:', '').strip() if location_p else 'N/A'
        
        # Extraer descripción
        description_p = next((p for p in card.find_all('p') 
                             if not p.find('strong') and not 'Ideal para' in p.text 
                             and not 'Ubicación' in p.text), None)
        description = description_p.text.strip() if description_p else 'No description available'
        
        # Extraer calificación
        rating_p = card.find('p', string=lambda s: 'Calificación' in s if s else False)
        rating = 0
        if rating_p:
            rating_text = rating_p.text
            # Extraer el número de la calificación (asumiendo formato "(X.X/5)")
            rating_match = re.search(r'\((\d+\.\d+)\/5\)', rating_text)
            if rating_match:
                try:
                    rating = float(rating_match.group(1))
                except ValueError:
                    rating = 0
        
        # Extraer costo
        cost_p = card.find('p', string=lambda s: 'Costo' in s if s else False)
        cost = 0
        if cost_p:
            cost_text = cost_p.text
            # Extraer el número del costo (asumiendo formato "$X.XX")
            cost_match = re.search(r'\$(\d+\.\d+)', cost_text)
            if cost_match:
                try:
                    cost = float(cost_match.group(1))
                except ValueError:
                    cost = 0
        
        # Extraer tipo
        # Podríamos asumirlo basado en otros elementos o usar un valor predeterminado
        type_p = card.find('p', string=lambda s: 'Ideal para' in s if s else False)
        rec_type = type_p.text.replace('Ideal para:', '').strip() if type_p else 'Attraction'
        
        recommendations.append({
            'name': name,
            'location': location,
            'description': description,
            'rating': rating,
            'cost': cost,
            'type': rec_type
        })
    
    return recommendations

def format_recommendation(rec: Dict[str, Any]) -> str:
    """Format a single recommendation in a readable way"""
    name = rec.get('name', 'UNNAMED').upper()
    divider = '-' * len(name)
    
    # Format various fields
    cost_field = rec.get('cost') or rec.get('entry_fee') or rec.get('average_person_expense', 0)
    cost = float(cost_field) if cost_field else 0
    cost_str = f"${cost:.2f}"
    
    # Format rating with stars
    rating = float(rec.get('rating', 0))
    stars = '★' * int(rating) + '☆' * (5 - int(rating))
    
    # Format description
    description = rec.get('description', 'No description available')
    if len(description) > 60:
        description = '\n'.join(description[i:i+60] for i in range(0, len(description), 60))

    return f"""
        {name}
        {divider}
        Type: {rec.get('type', 'N/A')}
        Location: {rec.get('location', 'N/A')}
        Cost: {cost_str}
        Rating: {stars} ({rating}/5)

        Description:
        {description}
    """

def format_recommendations_list(recommendations: List[Dict[str, Any]]) -> str:
    """Formats the entire list of recommendations"""
    if not recommendations:
        return "No se encontraron recomendaciones que coincidan con tus criterios."
    
    # Encabezado
    header = """
        RECOMENDACIONES TURÍSTICAS
        ==========================
    """
    
    # Formatear cada recomendación
    formatted_recs = [format_recommendation(rec) for rec in recommendations]
    
    # Unir todo con separadores
    separator = "\n\n" + "=" * 40 + "\n\n"
    formatted_text = header + separator.join(formatted_recs)
    
    return formatted_text