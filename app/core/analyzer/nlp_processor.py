"""Enhanced NLP processor with improved pattern matching and intent classification."""

import logging
import re
from typing import Dict, List, Optional, Any, Set
from functools import lru_cache
import spacy
from spacy.tokens import Doc
from fuzzywuzzy import process, fuzz
from datetime import datetime

# App imports
from app.utils.logger import get_logger
from app.core.utils.validators import DataValidator

logger = get_logger(__name__)

class ImprovedNLPProcessor:
    """Enhanced NLP processor with improved pattern matching and intent classification"""

    def __init__(self):
        """Initialize NLP processor with required components and models"""
        try:
            self.nlp = spacy.load("es_core_news_sm")
        except OSError:
            logger.error("Failed to load Spanish language model. Please install it using: python -m spacy download es_core_news_sm")
            raise
        except Exception as e:
            logger.error(f"Error initializing NLP model: {str(e)}")
            raise

        self.VALID_CATEGORIES = DataValidator.VALID_CATEGORIES
        
# Enhanced interest context with broader categories
        self.interest_context = {
            'cultural': [
                'museum', 'history', 'art', 'culture', 'heritage', 'monument', 'architecture',
                'theater', 'festival', 'ruins', 'temple', 'palace', 'historic_site', 'tradition',
                'ceremony', 'archaeological_site', 'folk_art', 'customs', 'music', 'dance',
                'religious_site', 'library', 'cultural_center', 'performing_arts',
                # Spanish terms
                'museo', 'historia', 'arte', 'cultura', 'patrimonio', 'monumento', 'arquitectura',
                'teatro', 'festival', 'ruinas', 'templo', 'palacio', 'sitio histórico', 'tradición',
                'ceremonia', 'sitio arqueológico', 'arte popular', 'costumbres', 'música', 'baile',
                'sitio religioso', 'biblioteca', 'centro cultural', 'artes escénicas'
            ],
            'adventure': [
                'hiking', 'climbing', 'adventure', 'trek', 'expedition', 'outdoor',
                'zipline', 'rappelling', 'canyoning', 'rafting', 'kayaking', 'paragliding',
                'rock_climbing', 'mountain_biking', 'horseback_riding', 'caving', 'camping',
                'off_road', 'survival_skills', 'bungee_jumping', 'skydiving', 'orienteering',
                # Spanish terms
                'senderismo', 'escalada', 'aventura', 'trekking', 'expedición', 'aire libre',
                'tirolina', 'rápel', 'barranquismo', 'rafting', 'kayak', 'parapente',
                'escalada en roca', 'ciclismo de montaña', 'montar a caballo', 'espeleología', 'acampar',
                'todoterreno', 'habilidades de supervivencia', 'puenting', 'paracaidismo', 'orientación'
            ],
            'nature': [
                'park', 'beach', 'nature', 'wildlife', 'flora', 'fauna', 'ecosystem',
                'forest', 'mountain', 'waterfall', 'lake', 'river', 'desert', 'canyon',
                'volcano', 'hot_springs', 'wetlands', 'cave', 'reef', 'lagoon', 'valley',
                'sanctuary', 'reserve', 'botanical_garden', 'geyser', 'natural_bridge',
                # Spanish terms
                'parque', 'playa', 'naturaleza', 'vida silvestre', 'flora', 'fauna', 'ecosistema',
                'bosque', 'montaña', 'cascada', 'lago', 'río', 'desierto', 'cañón',
                'volcán', 'aguas termales', 'humedales', 'cueva', 'arrecife', 'laguna', 'valle',
                'santuario', 'reserva', 'jardín botánico', 'géiser', 'puente natural'
            ],
            'food': [
                'restaurant', 'dining', 'food', 'cuisine', 'culinary', 'gastronomy', 'taste',
                'food_tour', 'cooking_class', 'wine_tasting', 'brewery', 'market', 'street_food',
                'farm_to_table', 'food_festival', 'local_produce', 'seafood', 'vegetarian',
                'traditional_cooking', 'food_market', 'distillery', 'coffee_shop', 'bakery',
                # Spanish terms
                'restaurante', 'comida', 'gastronomía', 'culinario', 'sabor',
                'tour gastronómico', 'clase de cocina', 'cata de vinos', 'cervecería', 'mercado', 'comida callejera',
                'de la granja a la mesa', 'festival gastronómico', 'productos locales', 'mariscos', 'vegetariano',
                'cocina tradicional', 'mercado de comida', 'destilería', 'cafetería', 'panadería'
            ],
            'water_activities': [
                'diving', 'snorkel', 'swim', 'marine', 'underwater', 'beach',
                'surfing', 'paddleboarding', 'jet_skiing', 'sailing', 'windsurfing',
                'fishing', 'boat_tour', 'whale_watching', 'dolphin_watching', 'waterpark',
                'scuba_diving', 'freediving', 'submarine_tour', 'coral_reef', 'aquarium',
                # Spanish terms
                'buceo', 'snorkel', 'nadar', 'marino', 'submarino', 'playa',
                'surf', 'paddle', 'moto acuática', 'navegación', 'windsurf',
                'pesca', 'paseo en barco', 'avistamiento de ballenas', 'avistamiento de delfines', 'parque acuático',
                'submarinismo', 'apnea', 'tour submarino', 'arrecife de coral', 'acuario'
            ],
            'local_experiences': [
                'traditional', 'authentic', 'local', 'cultural', 'indigenous',
                'homestay', 'village_visit', 'local_festival', 'community_tourism',
                'traditional_dance', 'folklore', 'local_market', 'artisan_workshop',
                'farming_experience', 'fishing_village', 'tribal_visit', 'local_guide',
                'traditional_music', 'cultural_exchange', 'local_ceremonies',
                # Spanish terms
                'tradicional', 'auténtico', 'local', 'cultural', 'indígena',
                'estancia en casa local', 'visita a pueblo', 'festival local', 'turismo comunitario',
                'baile tradicional', 'folklore', 'mercado local', 'taller artesanal',
                'experiencia agrícola', 'pueblo pesquero', 'visita tribal', 'guía local',
                'música tradicional', 'intercambio cultural', 'ceremonias locales'
            ]
        }

        # Enhanced patterns for various extractions with multilingual support
        self.enhanced_patterns = {
            'budget': [
                # English patterns
                r'(?:USD|\$|€)\s*(\d+(?:\.\d{2})?)\s*(?:per|a|each)?\s*(?:day|person|pax)?',
                r'budget.*?(\d+)(?:\s*(?:dollars|USD|€))?',
                r'(?:spend|cost|price).*?(\d+)(?:\s*(?:dollars|USD|€))?',
                r'around\s*(?:USD|\$|€)?\s*(\d+)',
                r'(\d+)\s*(?:USD|\$|€)?\s*(?:budget|per day|daily)',
                # Spanish patterns
                r'(?:USD|\$|€)\s*(\d+(?:\.\d{2})?)\s*(?:por|al|cada)?\s*(?:día|persona|pax)?',
                r'presupuesto.*?(\d+)(?:\s*(?:dólares|USD|€))?',
                r'(?:gastar|costo|precio).*?(\d+)(?:\s*(?:dólares|USD|€))?',
                r'alrededor\s*(?:de)?\s*(?:USD|\$|€)?\s*(\d+)',
                r'(\d+)\s*(?:USD|\$|€)?\s*(?:presupuesto|por día|diario|diarios)',
                r'(?:tengo|dispongo de) un presupuesto de\s*(?:USD|\$|€)?\s*(\d+)'
            ],
            'duration': [
                # English patterns
                r'(\d+)\s*(?:days?|nights?|evenings?)',
                r'stay(?:ing)?\s*(?:for)?\s*(\d+)\s*(?:days?|nights?)',
                r'(?:duration|period|time)\s*(?:of)?\s*(\d+)\s*(?:days?|nights?)',
                r'(?:plan|trip|vacation|holiday)\s*(?:for)?\s*(\d+)\s*(?:days?|nights?)',
                r'(\d+)-day(?:\s+trip|\s+tour|\s+vacation)?',
                # Spanish patterns
                r'(\d+)\s*(?:días?|noches?|tardes?)',
                r'estar(?:é|emos)?\s*(?:por|durante)?\s*(\d+)\s*(?:días?|noches?)',
                r'(?:duración|periodo|tiempo)\s*(?:de)?\s*(\d+)\s*(?:días?|noches?)',
                r'(?:plan|viaje|vacaciones)\s*(?:por|durante|de)?\s*(\d+)\s*(?:días?|noches?)',
                r'(\d+)(?:-|\s)?días?(?:\s+de\s+)?(?:viaje|tour|vacaciones)?',
                r'(?:voy|estaré|quiero estar) (?:por|durante) (\d+) (?:días?|noches?)'
            ],
            'location': [
                # English patterns
                r'(?:in|at|near|around|visit(?:ing)?)\s+([\w\s]+?)(?=\s+(?:and|or|,|\.|\b))',
                r'stay(?:ing)?\s+(?:in|at|near)\s+([\w\s]+?)(?=\s+(?:and|or|,|\.|\b))',
                r'(?:to|in|at)\s+([\w\s]+?)\s+(?:area|region|district|neighborhood)',
                r'explore\s+([\w\s]+?)(?=\s+(?:and|or|,|\.|\b))',
                r'interested\s+in\s+([\w\s]+?)(?=\s+(?:and|or|,|\.|\b))',
                # Spanish patterns
                r'(?:en|cerca de|alrededor de|visitar?)\s+([\w\s]+?)(?=\s+(?:y|o|,|\.|\b))',
                r'(?:alojarse|quedar(?:se)?|hospedar(?:se)?)\s+(?:en|cerca de)\s+([\w\s]+?)(?=\s+(?:y|o|,|\.|\b))',
                r'(?:a|en)\s+([\w\s]+?)\s+(?:área|región|distrito|barrio|zona)',
                r'explorar\s+([\w\s]+?)(?=\s+(?:y|o|,|\.|\b))',
                r'interesad[oa]\s+en\s+([\w\s]+?)(?=\s+(?:y|o|,|\.|\b))',
                r'(?:quiero|me gustaría) (?:visitar|conocer|ir a) ([\w\s]+?)(?=\s+(?:y|o|,|\.|\b))'
            ]
        }

        # Enhanced intent patterns with weighted keywords and Spanish support
        self.intent_patterns = {
            'activity_search': {
                'keywords': ['explore', 'visit', 'see', 'do', 'activities', 'experience',
                           'sightseeing', 'tour', 'discover', 'adventure', 'participate',
                           # Spanish keywords
                           'explorar', 'visitar', 'ver', 'hacer', 'actividades', 'experiencia',
                           'turismo', 'tour', 'descubrir', 'aventura', 'participar'],
                'weight': 1.5
            },
            'food_search': {
                'keywords': ['eat', 'restaurant', 'food', 'dining', 'cuisine', 'gastronomy',
                           'culinary', 'vegetarian', 'meal', 'lunch', 'dinner', 'taste',
                           # Spanish keywords
                           'comer', 'restaurante', 'comida', 'gastronomía', 'cocina',
                           'culinario', 'vegetariano', 'plato', 'almuerzo', 'cena', 'sabor'],
                'weight': 1.2
            },
            'cultural_interest': {
                'keywords': ['culture', 'history', 'museum', 'art', 'heritage', 'traditional',
                           'local', 'authentic', 'architecture', 'landmark',
                           # Spanish keywords
                           'cultura', 'historia', 'museo', 'arte', 'patrimonio', 'tradicional',
                           'local', 'auténtico', 'arquitectura', 'monumento'],
                'weight': 1.3
            },
            'nature_adventure': {
                'keywords': ['nature', 'hiking', 'outdoor', 'wildlife', 'park', 'beach',
                           'mountain', 'trek', 'adventure', 'exploration',
                           # Spanish keywords
                           'naturaleza', 'senderismo', 'aire libre', 'fauna', 'parque', 'playa',
                           'montaña', 'trekking', 'aventura', 'exploración'],
                'weight': 1.4
            },
            'planning_logistics': {
                'keywords': ['plan', 'schedule', 'itinerary', 'organize', 'book', 'reserve',
                           'arrangement', 'timing', 'duration', 'dates',
                           # Spanish keywords
                           'plan', 'horario', 'itinerario', 'organizar', 'reservar',
                           'arreglo', 'tiempo', 'duración', 'fechas'],
                'weight': 1.1
            }
        }

        # Activity patterns with detailed classifications and Spanish support
        self.activity_patterns = {
            'walking_tour': {
                'pattern': r'walk(?:ing)?\s*tour|guided\s*walk|city\s*walk|tour\s*(?:a\s*)?pie|visita\s*guiada|paseo\s*(?:por\s*la\s*)?ciudad',
                'category': 'cultural'
            },
            'water_sports': {
                'pattern': r'div(?:ing|e)|snorkel(?:ing)?|swim(?:ming)?|kayak(?:ing)?|buce(?:ar|o)|snorkel(?:ear)?|nad(?:ar|ando)|kayak',
                'category': 'adventure'
            },
            'cultural_activities': {
                'pattern': r'museum|gallery|exhibition|monument|heritage|historical|museo|galería|exposición|monumento|patrimonio|histórico',
                'category': 'cultural'
            },
            'nature_activities': {
                'pattern': r'hik(?:ing|e)|trek(?:king)?|nature\s*walk|wildlife|bird(?:watching)?|senderismo|caminata|paseo\s*(?:por\s*la\s*)?naturaleza|vida\s*silvestre|observación\s*de\s*aves',
                'category': 'nature'
            },
            'food_experiences': {
                'pattern': r'food.*tour|culinary.*experience|cooking\s*class|tasting|tour.*gastronómic[oa]|experiencia.*culinaria|clase.*cocina|degustación|cata',
                'category': 'food'
            }
        }

        self.location_cache = {}
        self.preference_cache = {}

    def extract_preferences(self, text: str) -> Dict[str, Any]:
        """
        Extract preferences from user input text with comprehensive pattern matching
        
        Args:
            text (str): User input text
            
        Returns:
            Dict[str, Any]: Extracted preferences
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid input text")
            return self._get_empty_preferences()

        try:
            # Create cache key
            cache_key = hash(text.lower().strip())
            if cache_key in self.preference_cache:
                return self.preference_cache[cache_key].copy()

            # Normalize text and process with spaCy
            text = text.lower().strip()
            doc = self.nlp(text)

            # Initialize preferences structure
            preferences = {
                'interests': [],
                'locations': [],
                'budget': None,
                'duration': None,
                'special_interests': [],
                'activity_types': [],
                'accommodation_preferences': [],
                'transportation_preferences': [],
                'dietary_restrictions': [],
                'accessibility_requirements': [],
                'language_preferences': ['english']  # Default language
            }

            # Extract locations
            potential_locations = self._extract_locations(doc, text)
            preferences['locations'] = [
                loc for loc in map(self._normalize_and_validate_location, potential_locations)
                if loc is not None
            ]

            # Extract interests and activities
            self._extract_interests_and_activities(text, preferences)

            # Extract budget with enhanced matching
            preferences['budget'] = self._extract_budget(text)

            # Extract duration with validation
            preferences['duration'] = self._extract_duration(text)

            # Extract special requirements
            self._extract_special_requirements(text, preferences)

            # Normalize and validate
            preferences = self._normalize_preferences(preferences)

            # Cache the results
            self.preference_cache[cache_key] = preferences.copy()

            return preferences

        except Exception as e:
            logger.error(f"Error extracting preferences: {str(e)}")
            return self._get_empty_preferences()

    def _extract_locations(self, doc: Doc, text: str) -> List[str]:
        """
        Extract locations from text using multiple methods
        
        Args:
            doc (Doc): Processed spaCy document
            text (str): Original text
            
        Returns:
            List[str]: Extracted locations
        """
        locations: Set[str] = set()

        try:
            # Extract from named entities
            for ent in doc.ents:
                if ent.label_ in {'GPE', 'LOC', 'FAC'}:
                    locations.add(ent.text)

            # Extract from location patterns
            for pattern in self.enhanced_patterns['location']:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    locations.add(match.group(1).strip())

            # Add basic location validation
            valid_locations = {
                loc for loc in locations 
                if len(loc) >= 2 and not any(char.isdigit() for char in loc)
            }

            return list(valid_locations)

        except Exception as e:
            logger.error(f"Error extracting locations: {str(e)}")
            return []

    def _extract_budget(self, text: str) -> Optional[float]:
        """
        Extract budget information using multiple patterns
        
        Args:
            text (str): Input text
            
        Returns:
            Optional[float]: Extracted budget amount
        """
        try:
            for pattern in self.enhanced_patterns['budget']:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        amount = float(match.group(1))
                        # Basic validation
                        if 0 < amount <= 10000:  # Reasonable budget range
                            return amount
                    except ValueError:
                        continue
            return None
        except Exception as e:
            logger.error(f"Error extracting budget: {str(e)}")
            return None

    def _extract_duration(self, text: str) -> Optional[int]:
        """
        Extract duration information using multiple patterns
        
        Args:
            text (str): Input text
            
        Returns:
            Optional[int]: Extracted duration in days
        """
        try:
            for pattern in self.enhanced_patterns['duration']:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        duration = int(match.group(1))
                        # Basic validation
                        if 1 <= duration <= 30:  # Reasonable duration range
                            return duration
                    except ValueError:
                        continue
            return None
        except Exception as e:
            logger.error(f"Error extracting duration: {str(e)}")
            return None

    def _extract_interests_and_activities(self, text: str, preferences: Dict[str, Any]) -> None:
        """
        Extract interests and activities from text
        
        Args:
            text (str): Input text
            preferences (Dict[str, Any]): Preferences dictionary to update
        """
        try:
            # Extract main interests
            for category, keywords in self.interest_context.items():
                if any(keyword in text for keyword in keywords):
                    preferences['interests'].append(category)

            # Extract activities with enhanced matching
            for activity, info in self.activity_patterns.items():
                if re.search(info['pattern'], text, re.IGNORECASE):
                    preferences['activity_types'].append(activity)
                    if info['category'] not in preferences['interests']:
                        preferences['interests'].append(info['category'])

            # Extract special interests with context
            for category, details in self.intent_patterns.items():
                if any(keyword in text for keyword in details['keywords']):
                    if category not in preferences['special_interests']:
                        preferences['special_interests'].append(category)

        except Exception as e:
            logger.error(f"Error extracting interests and activities: {str(e)}")

    def _extract_special_requirements(self, text: str, preferences: Dict[str, Any]) -> None:
        """
        Extract special requirements and preferences from text
        
        Args:
            text (str): Input text to analyze
            preferences (Dict[str, Any]): Preferences dictionary to update
        """
        try:
            # Dietary restrictions with enhanced patterns including Spanish support
            dietary_patterns = {
                'vegetarian': r'vegetarian|no\s*meat|meat[\s-]free|vegetariano|sin\s*carne',
                'vegan': r'vegan|plant[\s-]based|no\s*animal|vegano|a\s*base\s*de\s*plantas',
                'gluten_free': r'gluten[\s-]free|no\s*gluten|celiac|sin\s*gluten|celiaco',
                'halal': r'halal|muslim\s*friendly|islamic\s*dietary|halal|comida\s*musulmana',
                'kosher': r'kosher|jewish\s*dietary|kosher|comida\s*judía',
                'dairy_free': r'dairy[\s-]free|lactose[\s-]free|no\s*dairy|sin\s*lácteos|sin\s*lactosa',
                'nut_free': r'nut[\s-]free|no\s*nuts|peanut[\s-]free|sin\s*nueces|sin\s*frutos\s*secos'
            }

            # Accommodation preferences with enhanced patterns including Spanish support
            accommodation_patterns = {
                'luxury': r'luxury|high[\s-]end|upscale|premium|lujo|alta\s*gama|exclusivo',
                'budget': r'budget|cheap|affordable|economical|económico|barato|accesible|bajo\s*costo',
                'apartment': r'apartment|flat|condo|suite|apartamento|piso|condominio|suite',
                'hotel': r'hotel|resort|lodging|hotel|resort|alojamiento',
                'beachfront': r'beach[\s-]front|by\s*the\s*beach|ocean[\s-]view|frente\s*a\s*la\s*playa|vista\s*al\s*mar',
                'central': r'central|downtown|city[\s-]center|heart\s*of|centro|centro\s*de\s*la\s*ciudad|céntrico',
                'quiet': r'quiet|peaceful|serene|tranquil|tranquilo|pacífico|sereno'
            }

            # Transportation preferences including Spanish support
            transportation_patterns = {
                'car': r'car|driving|rent\s*a\s*car|rental\s*car|coche|conducir|alquilar\s*un\s*coche|carro',
                'public': r'public\s*transport|bus|train|metro|transporte\s*público|autobús|tren|metro',
                'taxi': r'taxi|cab|uber|ride[\s-]sharing|taxi|uber|cabify|compartir\s*viaje',
                'walking': r'walk|walking\s*distance|on\s*foot|caminar|a\s*pie|distancia\s*a\s*pie',
                'bicycle': r'bike|bicycle|cycling|bici|bicicleta|ciclismo'
            }

            # Initialize preference lists if they don't exist
            preferences.setdefault('dietary_restrictions', [])
            preferences.setdefault('accommodation_preferences', [])
            preferences.setdefault('transportation_preferences', [])
            preferences.setdefault('accessibility_requirements', [])
            preferences.setdefault('language_preferences', [])

            # Process dietary restrictions
            for diet, pattern in dietary_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    preferences['dietary_restrictions'].append(diet)

            # Process accommodation preferences
            for acc_type, pattern in accommodation_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    preferences['accommodation_preferences'].append(acc_type)

            # Process transportation preferences
            for trans_type, pattern in transportation_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    preferences['transportation_preferences'].append(trans_type)

            # Process special accessibility requirements with Spanish support
            accessibility_patterns = {
                'wheelchair': r'wheelchair|accessible|mobility\s*impaired|silla\s*de\s*ruedas|accesible|movilidad\s*reducida',
                'hearing': r'hearing\s*impaired|deaf|hard\s*of\s*hearing|discapacidad\s*auditiva|sordo|problemas\s*de\s*audición',
                'visual': r'visually\s*impaired|blind|sight\s*impaired|discapacidad\s*visual|ciego|problemas\s*de\s*visión',
                'elderly': r'elderly|senior|older\s*adults?|anciano|mayor|tercera\s*edad',
                'stroller': r'stroller|baby\s*carriage|pram|cochecito|carrito\s*de\s*bebé'
            }

            for req_type, pattern in accessibility_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    preferences['accessibility_requirements'].append(req_type)

            # Process language preferences with Spanish support
            language_patterns = {
                'english': r'english|eng|inglés',
                'spanish': r'spanish|español|esp',
                'dutch': r'dutch|nederlands|holandés',
                'papiamento': r'papiamento|papiamentu',
                'portuguese': r'portuguese|português|port|portugués'
            }

            for lang, pattern in language_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    preferences['language_preferences'].append(lang)

            # Default to English if no language preference specified
            if not preferences['language_preferences']:
                preferences['language_preferences'] = ['english']

            # Remove duplicates while preserving order
            for key in ['dietary_restrictions', 'accommodation_preferences', 
                    'transportation_preferences', 'accessibility_requirements', 
                    'language_preferences']:
                if key in preferences:
                    preferences[key] = list(dict.fromkeys(preferences[key]))

        except Exception as e:
            logger.error(f"Error extracting special requirements: {str(e)}")
            # Initialize empty lists for all preference types if error occurs
            preferences.update({
                'dietary_restrictions': [],
                'accommodation_preferences': [],
                'transportation_preferences': [],
                'accessibility_requirements': [],
                'language_preferences': ['english']  # Default to English
            })

    def _normalize_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and deduplicate preferences with enhanced processing
        
        Args:
            preferences (Dict[str, Any]): Raw preferences dictionary
            
        Returns:
            Dict[str, Any]: Normalized and deduplicated preferences
        """
        normalized = preferences.copy()

        # Remove duplicates while preserving order
        list_fields = [
            'interests', 'locations', 'activity_types', 'special_interests',
            'accommodation_preferences', 'transportation_preferences',
            'dietary_restrictions', 'language_preferences'
        ]

        for field in list_fields:
            if field in normalized:
                # Convert to lowercase for case-insensitive deduplication
                items = [item.lower() for item in normalized[field] if item]
                # Remove duplicates while preserving order
                seen = set()
                unique_items = []
                for item in items:
                    if item not in seen:
                        seen.add(item)
                        unique_items.append(item)
                normalized[field] = unique_items

        # Ensure all required fields exist
        for field in list_fields:
            normalized.setdefault(field, [])

        # Normalize budget if present
        if 'budget' in normalized and normalized['budget'] is not None:
            try:
                normalized['budget'] = float(normalized['budget'])
            except (ValueError, TypeError):
                normalized['budget'] = None

        # Normalize duration if present
        if 'duration' in normalized and normalized['duration'] is not None:
            try:
                normalized['duration'] = int(normalized['duration'])
                if normalized['duration'] < 1:  # Ensure positive duration
                    normalized['duration'] = None
            except (ValueError, TypeError):
                normalized['duration'] = None

        # Add timestamp for tracking
        normalized['timestamp'] = datetime.now().isoformat()

        return normalized

    def _add_preference_weights(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add weight scores to preferences based on confidence and priority
        
        Args:
            preferences (Dict[str, Any]): Normalized preferences
            
        Returns:
            Dict[str, Any]: Preferences with weights
        """
        weighted = preferences.copy()
        weights = {}

        try:
            # Weight locations based on specificity
            if weighted.get('locations'):
                location_weights = {}
                for loc in weighted['locations']:
                    # Higher weight for exact matches
                    if loc in self.VALID_CATEGORIES['locations']:
                        location_weights[loc] = 1.0
                    # Lower weight for fuzzy matches
                    else:
                        match = self._normalize_and_validate_location(loc)
                        if match:
                            location_weights[match] = 0.8
                weights['locations'] = location_weights

            # Weight interests based on context and priority
            if weighted.get('interests'):
                interest_weights = {}
                for interest in weighted['interests']:
                    weight = 1.0
                    # Increase weight for interests with multiple keywords
                    if any(all(kw in interest for kw in keywords) 
                          for keywords in self.interest_context.values()):
                        weight *= 1.2
                    interest_weights[interest] = weight
                weights['interests'] = interest_weights

            weighted['_weights'] = weights
            return weighted

        except Exception as e:
            logger.error(f"Error adding preference weights: {str(e)}")
            return preferences  # Return original preferences if weighting fails

    def _validate_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean preferences with comprehensive checks
        
        Args:
            preferences (Dict[str, Any]): Preferences to validate
            
        Returns:
            Dict[str, Any]: Validated and cleaned preferences
        """
        validated = {}
        try:
            # Validate locations
            if preferences.get('locations'):
                validated['locations'] = [
                    loc for loc in preferences['locations']
                    if self._normalize_and_validate_location(loc)
                ]

            # Validate interests
            if preferences.get('interests'):
                validated['interests'] = [
                    interest for interest in preferences['interests']
                    if DataValidator.normalize_interest(interest)
                ]

            # Validate budget
            if preferences.get('budget'):
                validated['budget'] = DataValidator.normalize_budget(preferences['budget'])

            # Validate duration
            if preferences.get('duration'):
                try:
                    duration = int(preferences['duration'])
                    if 1 <= duration <= 30:  # Reasonable range check
                        validated['duration'] = duration
                except (ValueError, TypeError):
                    pass

            # Add other validated fields
            for key, value in preferences.items():
                if key not in validated and value is not None:
                    validated[key] = value

            return validated

        except Exception as e:
            logger.error(f"Error validating preferences: {str(e)}")
            return preferences  # Return original preferences if validation fails

    def _cache_preferences(self, text: str, preferences: Dict[str, Any]) -> None:
        """
        Cache processed preferences for future retrieval
        
        Args:
            text (str): Original query text
            preferences (Dict[str, Any]): Processed preferences
        """
        try:
            cache_key = hash(text.lower().strip())
            self.preference_cache[cache_key] = preferences.copy()

            # Maintain cache size
            if len(self.preference_cache) > 1000:  # Arbitrary limit
                # Remove oldest entries
                oldest_keys = sorted(self.preference_cache.keys())[:100]
                for key in oldest_keys:
                    self.preference_cache.pop(key, None)

        except Exception as e:
            logger.error(f"Error caching preferences: {str(e)}")

    def classify_intent(self, text: str) -> Dict[str, float]:
        """Classify user intent with enhanced pattern matching"""
        try:
            doc = self.nlp(text.lower())
            intent_scores = {intent: 0.0 for intent in self.intent_patterns}

            # Calculate raw scores
            for sent in doc.sents:
                for intent, details in self.intent_patterns.items():
                    score = sum(details['weight'] for keyword in details['keywords']
                              if keyword in sent.text)
                    intent_scores[intent] += score

            # Normalize scores
            total_score = sum(intent_scores.values())
            if total_score > 0:
                intent_scores = {k: round(v/total_score, 3)
                               for k, v in intent_scores.items()}

            return intent_scores

        except Exception as e:
            logger.error(f"Error in intent classification: {str(e)}")
            return {intent: 0.0 for intent in self.intent_patterns}

    def _get_empty_preferences(self) -> Dict[str, Any]:
        """Return a template for empty preferences"""
        return {
            'interests': [],
            'locations': [],
            'budget': None,
            'duration': None,
            'special_interests': [],
            'activity_types': [],
            'accommodation_preferences': [],
            'transportation_preferences': [],
            'dietary_restrictions': []
        }

    @lru_cache(maxsize=128)
    def _normalize_and_validate_location(self, location: str) -> Optional[str]:
        """Normalize and validate location with caching"""
        if not location:
            return None
        location = location.lower().strip()
        if not DataValidator.validate_location(location):
            return None

        # Add fuzzy matching with DataValidator's locations
        matches = process.extractBests(
            location,
            self.VALID_CATEGORIES['locations'],
            score_cutoff=85,
            scorer=fuzz.token_sort_ratio
        )
        return matches[0][0] if matches else None