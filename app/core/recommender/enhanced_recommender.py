from fastapi import HTTPException
from pydantic import BaseModel
import spacy
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Any, Optional, Tuple
from utils.logger import get_logger
from utils.openai_helper import OpenAIHelper
from functools import lru_cache
from datetime import datetime, timedelta
from langdetect import detect
import openai
from openai import OpenAIError

logger = get_logger(__name__)

class SimpleCache:
    def __init__(self):
        self.cache = {}
        self.ttl = timedelta(hours=24)

    def get(self, key: str) -> Optional[str]:
        """Get value from cache if not expired"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: str):
        """Set value in cache with current timestamp"""
        self.cache[key] = (value, datetime.now())

    def cleanup(self):
        """Remove expired entries"""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp >= self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]

class EnhancedRecommender:
    # Datos multilingües para Curazao
    CURACAO_DATA = {
        'en': {
            'food': [
                "Keshi Yena", "Stoba", "Funchi", "Pastechi", "Sopi Mondongo",
                "Kabritu", "Piska Kruda", "Guiambo", "Arepa di Pampuna"
            ],
            'events': [
                "Carnival", "Seú Parade", "Harvest Festival",
                "Curaçao North Sea Jazz Festival", "Blue Marlin Release Tournament"
            ]
        },
        'es': {
            'food': [
                "Keshi Yena", "Estofado", "Funchi", "Pastechi", "Sopa de Mondongo",
                "Cabrito", "Pescado Crudo", "Quiambo", "Arepa de Calabaza"
            ],
            'events': [
                "Carnaval", "Desfile Seú", "Festival de la Cosecha",
                "Festival de Jazz del Mar del Norte", "Torneo de Pesca del Marlin Azul"
            ]
        }
    }

    def __init__(self, openai_helper: OpenAIHelper):
        """Initialize the enhanced recommender with multilingual capabilities"""
        self.openai_helper = openai_helper
        self.logger = logging.getLogger(__name__)
        self.cache = SimpleCache()
        
        # Initialize NLP models
        self.nlp_models = {}
        self._load_nlp_models()

    def _load_nlp_models(self):
        """Load available NLP models for different languages"""
        try:
            self.nlp_models['en'] = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("English model not available")

        try:
            self.nlp_models['es'] = spacy.load("es_core_news_sm")
        except OSError:
            self.logger.warning("Spanish model not available")

        if not self.nlp_models:
            try:
                self.nlp_models['xx'] = spacy.load("xx_ent_wiki_sm")
                self.logger.info("Using multilingual model as fallback")
            except OSError:
                raise HTTPException(
                    status_code=500,
                    detail="No NLP models available. Please install required language models."
                )

    def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        try:
            lang = detect(text)
            return 'en' if lang not in ['es'] else lang
        except:
            return 'en'  # Default to English

    def get_nlp_model(self, language: str):
        """Get appropriate NLP model for language"""
        if language in self.nlp_models:
            return self.nlp_models[language]
        return self.nlp_models.get('xx', self.nlp_models['en'])

    def detect_entities(self, text: str, lang: str) -> Dict[str, List[str]]:
        """Detect food items and events in specified language"""
        text_lower = text.lower()
        data = self.CURACAO_DATA[lang if lang in self.CURACAO_DATA else 'en']
        
        return {
            'food': [
                food for food in data['food']
                if food.lower() in text_lower
            ],
            'events': [
                event for event in data['events']
                if event.lower() in text_lower
            ]
        }

    def extract_entities(self, text: str) -> Tuple[Dict[str, List[str]], str]:
        """Extract named entities from text using appropriate language model"""
        # Detect language
        lang = self.detect_language(text)
        
        # Get appropriate NLP model
        nlp = self.get_nlp_model(lang)
        
        # Process text
        doc = nlp(text)
        
        entities = {
            "locations": [],
            "attractions": [],
            "food": [],
            "events": [],
            "other": []
        }

        # Map spaCy entity labels to our categories
        entity_mapping = {
            "GPE": "locations",
            "FAC": "attractions",
            "EVENT": "events",
            "NORP": "other",
            "PRODUCT": "other",
            "ORG": "attractions",
            "LOC": "locations"
        }

        # Extract entities from spaCy
        for ent in doc.ents:
            category = entity_mapping.get(ent.label_, "other")
            if ent.text not in entities[category]:
                entities[category].append(ent.text)

        # Add custom detections
        custom_entities = self.detect_entities(text, lang)
        entities["food"].extend(custom_entities["food"])
        entities["events"].extend(custom_entities["events"])

        return entities, lang

    def generate_prompt(self, context: Dict[str, Any], lang: str) -> str:
        """Generate language-appropriate prompt for OpenAI"""
        prompts = {
            'es': f"""
            Actúa como un guía turístico experto de Curazao. Crea un párrafo introductorio que:
            1. Mencione las ubicaciones: {context['locations']}
            2. Haga referencia a las atracciones: {context['attractions']}
            3. Incluya elementos culturales como comida ({context['food_items']}) o eventos ({context['events']})
            4. Conecte con los intereses del usuario: "{context['user_query']}"
            Hazlo personal y atractivo, en 2-3 oraciones.
            """,
            'en': f"""
            Act as an expert Curaçao tour guide. Create an engaging introduction paragraph that:
            1. Mentions the locations: {context['locations']}
            2. References attractions: {context['attractions']}
            3. Includes cultural elements like food ({context['food_items']}) or events ({context['events']})
            4. Connects with user interests: "{context['user_query']}"
            Make it personal and engaging, about 2-3 sentences long.
            """
        }
        return prompts.get(lang, prompts['en'])

    def generate_card_prompt(self, title: str, context: Dict[str, Any], lang: str) -> str:
        """Generate language-appropriate prompt for recommendation cards"""
        prompts = {
            'es': f"""
            Como experto en turismo de Curazao, mejora esta recomendación para {title}.
            Incluye:
            1. Importancia histórica o cultural
            2. Conexión con los intereses del usuario: {context['user_query']}
            3. Consejos prácticos para visitantes
            4. Elementos relevantes de comida, eventos o cultura de: {context['food_items']}, {context['events']}
            Mantenlo conciso pero informativo, aproximadamente 2-3 oraciones.
            """,
            'en': f"""
            As a Curaçao tourism expert, enhance this recommendation for {title}.
            Include:
            1. Historical or cultural significance
            2. Connection to user interests: {context['user_query']}
            3. Practical tips for visitors
            4. Any relevant food, events, or cultural elements from: {context['food_items']}, {context['events']}
            Keep it concise but informative, about 2-3 sentences.
            """
        }
        return prompts.get(lang, prompts['en'])

    def generate_tip_prompt(self, title: str, lang: str) -> str:
        """Generate language-appropriate prompt for expert tips"""
        prompts = {
            'es': f"Proporciona un consejo específico de experto para visitantes de {title}, basado en conocimiento local.",
            'en': f"Provide one specific expert tip for visitors to {title}, based on local knowledge."
        }
        return prompts.get(lang, prompts['en'])

    async def process_recommendations(
        self, 
        html_content: str, 
        user_query: str, 
        preferences: Dict[str, Any]
    ) -> str:
        """Process and enhance recommendations using NER and OpenAI"""
        try:
            # Extract entities and detect language
            query_entities, detected_lang = self.extract_entities(user_query)
            
            # Additional entities from preferences
            if preferences.get("specific_sites"):
                query_entities["attractions"].extend([
                    site for site in preferences["specific_sites"] 
                    if site not in query_entities["attractions"]
                ])
            
            if preferences.get("cuisine_preferences"):
                query_entities["food"].extend([
                    food for food in preferences["cuisine_preferences"] 
                    if food not in query_entities["food"]
                ])

            # Enrich HTML content
            enhanced_html = await self.enrich_html_content(
                html_content,
                query_entities,
                user_query,
                detected_lang
            )

            return enhanced_html

        except HTTPException as http_err:
            raise http_err
        except OpenAIError as openai_err:
            self.logger.error(f"OpenAI API error: {str(openai_err)}")
            raise HTTPException(
                status_code=502,
                detail=f"Error en el servicio de OpenAI: {str(openai_err)}"
            )
        except Exception as e:
            self.logger.error(f"Error inesperado: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Ocurrió un error inesperado al procesar tu solicitud."
            )
            
    async def enrich_html_content(
        self, 
        html_content: str, 
        entities: Dict[str, List[str]], 
        user_query: str,
        lang: str
    ) -> str:
        """Enrich HTML content with additional context using OpenAI"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            context = {
                "entities": entities,
                "user_query": user_query,
                "locations": ", ".join(entities["locations"]),
                "attractions": ", ".join(entities["attractions"]),
                "food_items": ", ".join(entities["food"]),
                "events": ", ".join(entities["events"])
            }

            # Generate enhanced introduction
            intro_prompt = self.generate_prompt(context, lang)
            cache_key = f"intro_{hash(intro_prompt)}"
            
            enhanced_intro = self.cache.get(cache_key)
            if not enhanced_intro:
                try:
                    enhanced_intro = self.openai_helper.generate_tour_guide_response(
                        user_text=intro_prompt,
                        system_message=None,
                        lang=lang
                    )
                    if enhanced_intro:
                        self.cache.set(cache_key, enhanced_intro)
                except OpenAIError as e:
                    self.logger.error(f"OpenAI error generating intro: {str(e)}")
                    raise HTTPException(
                        status_code=502,
                        detail="Error al generar el contenido mejorado. Por favor, intenta de nuevo."
                    )

            # Update introduction
            intro_div = soup.find('div', class_='intro')
            if not intro_div:
                intro_div = soup.new_tag('div', attrs={'class': 'intro'})
                if soup.body:
                    soup.body.insert(0, intro_div)
                else:
                    soup.append(intro_div)

            intro_p = soup.new_tag('p')
            intro_p['style'] = 'font-size: 1.2em; color: #333; margin-bottom: 16px;'
            intro_p.string = enhanced_intro

            intro_div.clear()
            intro_div.append(intro_p)

            # Process recommendation cards
            await self._process_recommendation_cards(soup, context, lang)

            # Clean up old cache entries periodically
            self.cache.cleanup()

            return str(soup)

        except Exception as e:
            self.logger.error(f"Error enriching HTML content: {str(e)}")
            raise HTTPException(
                status_code=502,
                detail="Error al generar el contenido mejorado. Por favor, intenta de nuevo."
            )

    async def _process_recommendation_cards(
        self, 
        soup: BeautifulSoup, 
        context: Dict[str, Any],
        lang: str
    ):
        """Process and enhance individual recommendation cards"""
        try:
            for card in soup.find_all('div', class_='recommendation-card'):
                title_elem = card.find('h3')
                if not title_elem:
                    continue
                    
                title = title_elem.text.strip()
                
                # Generate enhanced description
                desc_prompt = self.generate_card_prompt(title, context, lang)
                cache_key = f"card_{hash(desc_prompt)}"
                
                enhanced_description = self.cache.get(cache_key)
                if not enhanced_description:
                    enhanced_description = self.openai_helper.generate_tour_guide_response(
                        user_text=desc_prompt,
                        system_message=None,
                        lang=lang
                    )
                    if enhanced_description:
                        self.cache.set(cache_key, enhanced_description)
                
                if enhanced_description:
                    desc_div = card.find('div', class_='description-content')
                    if not desc_div:
                        desc_div = soup.new_tag('div', attrs={'class': 'description-content'})
                        card.append(desc_div)
                    
                    desc_p = soup.new_tag('p', attrs={'class': 'description'})
                    desc_p.string = enhanced_description
                    desc_div.clear()
                    desc_div.append(desc_p)

                # Add expert tip
                if not card.find('p', class_='expert-tip'):
                    tip_prompt = self.generate_tip_prompt(title, lang)
                    cache_key = f"tip_{hash(tip_prompt)}"
                    
                    expert_tip = self.cache.get(cache_key)
                    if not expert_tip:
                        expert_tip = self.openai_helper.generate_tour_guide_response(
                            user_text=tip_prompt,
                            system_message=None,
                            lang=lang
                        )
                        if expert_tip:
                            self.cache.set(cache_key, expert_tip)
                    
                    if expert_tip:
                        tip_p = soup.new_tag('p', attrs={'class': 'expert-tip'})
                        tip_p['style'] = 'font-style: italic; color: #666;'
                        tip_p.string = f"{'🎯 Pro Tip:' if lang == 'en' else '🎯 Consejo Pro:'} {expert_tip}"
                        card.append(tip_p)
                        
        except Exception as e:
            self.logger.error(f"Error processing recommendation cards: {str(e)}")
            # No lanzamos la excepción aquí para permitir un fallo parcial