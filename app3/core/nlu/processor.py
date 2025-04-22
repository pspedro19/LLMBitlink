from typing import Dict, Any, List, Optional
import spacy
import re
from datetime import datetime
from app3.utils.logger import get_logger

logger = get_logger(__name__)

class NLUProcessor:
    """
    Procesador de lenguaje natural que extrae preferencias estructuradas
    de consultas en lenguaje natural.
    """
    def __init__(self):
        """Inicializa el procesador NLU con modelos y patrones"""
        try:
            # Cargar modelo de español por defecto
            self.nlp = spacy.load("es_core_news_sm")
            
            # Cargar modelo de inglés si está disponible
            try:
                self.nlp_en = spacy.load("en_core_web_sm")
            except:
                self.nlp_en = None
                logger.warning("Modelo de inglés no disponible")
                
            # Inicializar patrones para extracción
            self._init_patterns()
            
            # Inicializar contexto de intereses
            self._init_interest_context()
            
        except Exception as e:
            logger.error(f"Error inicializando NLUProcessor: {e}")
            raise
            
    def _init_patterns(self):
        """Inicializa patrones regex para extracción de preferencias"""
        self.patterns = {
            "budget": [
                # Patrones en español
                r'(?:USD|\$|€)\s*(\d+(?:\.\d{2})?)\s*(?:por|al|cada)?\s*(?:día|persona|pax)?',
                r'presupuesto.*?(\d+)(?:\s*(?:dólares|USD|€))?',
                r'(?:gastar|costo|precio).*?(\d+)(?:\s*(?:dólares|USD|€))?',
                r'(\d+)\s*(?:USD|\$|€)?\s*(?:presupuesto|por día|diario|diarios)',
                
                # Patrones en inglés
                r'(?:USD|\$|€)\s*(\d+(?:\.\d{2})?)\s*(?:per|a|each)?\s*(?:day|person|pax)?',
                r'budget.*?(\d+)(?:\s*(?:dollars|USD|€))?',
                r'(?:spend|cost|price).*?(\d+)(?:\s*(?:dollars|USD|€))?',
                r'(\d+)\s*(?:USD|\$|€)?\s*(?:daily|per day|budget)',
            ],
            "duration": [
                # Patrones en español
                r'(\d+)\s*(?:días?|noches?|tardes?)',
                r'estar(?:é|emos)?\s*(?:por|durante)?\s*(\d+)\s*(?:días?|noches?)',
                r'(?:duración|periodo|tiempo)\s*(?:de)?\s*(\d+)\s*(?:días?|noches?)',
                r'(?:plan|viaje|vacaciones)\s*(?:por|durante|de)?\s*(\d+)\s*(?:días?|noches?)',
                r'(\d+)(?:-|\s)?días?(?:\s+de\s+)?(?:viaje|tour|vacaciones)?',
                
                # Patrones en inglés
                r'(\d+)\s*(?:days?|nights?)',
                r'stay(?:ing)?\s*(?:for)?\s*(\d+)\s*(?:days?|nights?)',
                r'(?:duration|period|time)\s*(?:of)?\s*(\d+)\s*(?:days?|nights?)',
                r'(\d+)[-\s]?day(?:\s+trip|\s+tour|\s+vacation)?',
            ],
            "group_size": [
                # Patrones en español
                r'(?:somos|iremos|vamos|viajamos|viajo con|viajaré con)\s*(\d+)\s*(?:personas?|gente|viajeros?)?',
                r'grupo\s*(?:de)?\s*(\d+)',
                r'(\d+)\s*(?:personas?|viajeros?|turistas?|visitantes?)',
                
                # Patrones en inglés
                r'(?:we are|there are|group of)\s*(\d+)\s*(?:people|persons|travelers)?',
                r'(\d+)\s*(?:of us|people|travelers|tourists|visitors)',
            ],
            "special_occasion": [
                # Patrones para ocasiones especiales
                r'(?:aniversario|cumpleaños|boda|luna de miel|honeymoon|anniversary|wedding|birthday)',
            ]
        }
        
    def _init_interest_context(self):
        """Inicializa el contexto de intereses con categorías y términos relacionados"""
        self.interest_context = {
            'cultural': [
                # Español
                'museo', 'historia', 'arte', 'cultura', 'patrimonio', 'monumento', 'arquitectura',
                'teatro', 'festival', 'ruinas', 'templo', 'palacio', 'sitio histórico', 'tradición',
                'biblioteca', 'centro cultural', 'arte popular', 'colonial', 'histórico',
                
                # Inglés
                'museum', 'history', 'art', 'culture', 'heritage', 'monument', 'architecture',
                'theater', 'festival', 'ruins', 'temple', 'palace', 'historic site', 'tradition',
                'library', 'cultural center', 'folk art', 'colonial', 'historical', 'photography'
            ],
            'naturaleza': [
                # Español
                'parque', 'playa', 'naturaleza', 'vida silvestre', 'flora', 'fauna', 'ecosistema',
                'bosque', 'montaña', 'cascada', 'lago', 'río', 'arrecife', 'laguna', 'valle',
                'reserva', 'jardín botánico', 'cueva', 'paisaje', 'escénico', 'natural',
                
                # Inglés
                'park', 'beach', 'nature', 'wildlife', 'flora', 'fauna', 'ecosystem',
                'forest', 'mountain', 'waterfall', 'lake', 'river', 'reef', 'lagoon', 'valley',
                'reserve', 'botanical garden', 'cave', 'landscape', 'scenic', 'natural', 'retreat'
            ],
            'gastronomía': [
                # Español
                'restaurante', 'comida', 'gastronomía', 'culinario', 'sabor', 'cena', 'especial',
                'tour gastronómico', 'clase de cocina', 'cata', 'mercado', 'comida callejera',
                'festival gastronómico', 'productos locales', 'mariscos', 'cocina tradicional',
                
                # Inglés
                'restaurant', 'food', 'cuisine', 'culinary', 'taste', 'dinner', 'special',
                'food tour', 'cooking class', 'tasting', 'market', 'street food',
                'food festival', 'local produce', 'seafood', 'traditional cooking'
            ],
            'actividades_acuáticas': [
                # Español
                'buceo', 'snorkel', 'nadar', 'marino', 'submarino', 'playa',
                'surf', 'paddle', 'moto acuática', 'navegación', 'windsurf',
                'pesca', 'paseo en barco', 'avistamiento de ballenas', 'parque acuático',
                
                # Inglés
                'diving', 'snorkel', 'swim', 'marine', 'underwater', 'beach',
                'surfing', 'paddleboarding', 'jet skiing', 'sailing', 'windsurfing',
                'fishing', 'boat tour', 'whale watching', 'waterpark'
            ],
            'aventura': [
                # Español
                'senderismo', 'escalada', 'aventura', 'trekking', 'expedición', 'aire libre',
                'tirolina', 'ciclismo de montaña', 'montar a caballo', 'acampar', 'todoterreno',
                
                # Inglés
                'hiking', 'climbing', 'adventure', 'trek', 'expedition', 'outdoor',
                'zipline', 'mountain biking', 'horseback riding', 'camping', 'off road'
            ],
            'vida_nocturna': [
                # Español
                'fiesta', 'discoteca', 'club', 'bar', 'música en vivo', 'concierto', 'baile',
                'vida nocturna', 'casino', 'espectáculo', 'festival nocturno',
                
                # Inglés
                'party', 'nightclub', 'club', 'bar', 'live music', 'concert', 'dance',
                'nightlife', 'casino', 'show', 'night festival'
            ],
            'romántico': [
                # Español
                'romántico', 'romance', 'pareja', 'luna de miel', 'aniversario', 'íntimo',
                'especial', 'cena romántica', 'atardecer', 'vistas', 'exclusivo',
                
                # Inglés
                'romantic', 'romance', 'couple', 'honeymoon', 'anniversary', 'intimate',
                'special', 'romantic dinner', 'sunset', 'views', 'exclusive'
            ]
        }
    
    def process_query(self, query: str, language: str = "auto") -> Dict[str, Any]:
        """
        Procesa una consulta en lenguaje natural y extrae información estructurada.
        
        Args:
            query (str): Texto de la consulta del usuario
            language (str): Código de idioma o "auto" para detección
            
        Returns:
            Dict[str, Any]: Información estructurada extraída de la consulta
        """
        try:
            # Normalizar texto
            query = query.strip()
            if not query:
                return self._empty_result(query)
                
            # Detectar idioma si es automático
            if language == "auto":
                language = self._detect_language(query)
                
            # Procesar con el modelo NLP adecuado
            doc = self._process_with_language_model(query, language)
                
            # Extraer preferencias
            preferences = self._extract_preferences(query, doc, language)
                
            # Clasificar intención
            intent = self._classify_intent(query, doc, language)
                
            # Estructurar resultado
            return {
                "query": query,
                "language": language,
                "intent": intent,
                "preferences": preferences,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error procesando consulta: {e}")
            return self._empty_result(query)
            
    def _process_with_language_model(self, text: str, language: str) -> Any:
        """Procesa el texto con el modelo de lenguaje adecuado"""
        if language == "en" and self.nlp_en:
            return self.nlp_en(text)
        return self.nlp(text)
    
    def _detect_language(self, text: str) -> str:
        """Detecta el idioma del texto"""
        # Detección básica basada en palabras clave
        spanish_words = ['que', 'como', 'para', 'con', 'por', 'quiero', 'días', 'viaje']
        english_words = ['the', 'and', 'for', 'with', 'want', 'days', 'travel', 'trip']
        
        text_lower = text.lower()
        spanish_count = sum(1 for word in spanish_words if word in text_lower)
        english_count = sum(1 for word in english_words if word in text_lower)
        
        return "es" if spanish_count >= english_count else "en"
    
    def _extract_preferences(self, text: str, doc: Any, language: str) -> Dict[str, Any]:
        """Extrae preferencias estructuradas del texto y documento procesado"""
        preferences = {
            "interests": [],
            "locations": [],
            "budget": None,
            "duration": None,
            "group_size": None,
            "with_children": self._has_children(text),
            "accommodation_type": None,
            "transportation": None,
            "special_occasion": self._extract_special_occasion(text)
        }
        
        # Extraer presupuesto
        preferences["budget"] = self._extract_numeric_with_patterns(text, self.patterns["budget"])
        
        # Extraer duración
        preferences["duration"] = self._extract_numeric_with_patterns(text, self.patterns["duration"])
        
        # Extraer tamaño de grupo
        preferences["group_size"] = self._extract_numeric_with_patterns(text, self.patterns["group_size"])
        
        # Extraer ubicaciones
        preferences["locations"] = self._extract_locations(text, doc)
        
        # Extraer intereses
        preferences["interests"] = self._extract_interests(text)
        
        # Inferir intereses adicionales basados en contexto
        self._infer_additional_interests(text, preferences)
        
        return preferences
        
    def _extract_numeric_with_patterns(self, text: str, patterns: List[str]) -> Optional[float]:
        """Extrae valores numéricos usando patrones regex"""
        for pattern in patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    value = float(matches.group(1))
                    if value > 0:
                        return value
                except (ValueError, IndexError):
                    continue
        return None
        
    def _extract_locations(self, text: str, doc: Any) -> List[str]:
        """Extrae ubicaciones del texto"""
        locations = []
        
        # Extraer entidades geográficas
        for ent in doc.ents:
            if ent.label_ in ['LOC', 'GPE']:
                locations.append(ent.text)
                
        # Lista de ubicaciones conocidas en Curaçao
        known_locations = [
            "willemstad", "punda", "otrobanda", "westpunt", "christoffel",
            "jan thiel", "mambo beach", "playa kalki", "playa kenepa", "curaçao",
            "curacao", "kura hulanda", "pietermaai", "caracasbaai"
        ]
        
        # Buscar ubicaciones conocidas
        text_lower = text.lower()
        for location in known_locations:
            if location in text_lower and location not in [loc.lower() for loc in locations]:
                locations.append(location)
                
        return locations
        
    def _extract_interests(self, text: str) -> List[str]:
        """Extrae intereses del texto de manera más robusta"""
        interests = []
        text_lower = text.lower()
        
        # Verificar cada categoría de interés
        for category, keywords in self.interest_context.items():
            if any(keyword in text_lower for keyword in keywords):
                interests.append(category)
        
        # Detección directa de categorías de viaje comunes
        common_interests = {
            'cultural': ['cultural', 'museos', 'histórico', 'historia', 'museums', 'historical'],
            'naturaleza': ['naturaleza', 'playas', 'nature', 'beaches', 'landscape'],
            'aventura': ['aventura', 'senderismo', 'hiking', 'adventure', 'trekking'],
            'gastronomía': ['comida', 'restaurantes', 'gastronomía', 'food', 'culinary'],
            'actividades_acuáticas': ['buceo', 'snorkel', 'nadar', 'diving', 'swimming'],
            'vida_nocturna': ['fiesta', 'música', 'bares', 'party', 'nightlife', 'bar'],
            'romántico': ['romántico', 'pareja', 'romance', 'couple', 'intimate']
        }
        
        for category, keywords in common_interests.items():
            if any(keyword in text_lower for keyword in keywords) and category not in interests:
                interests.append(category)
        
        # Inferir intereses basados en palabras clave específicas
        if any(word in text_lower for word in ['aniversario', 'anniversary', 'pareja', 'couple']):
            if 'romántico' not in interests:
                interests.append('romántico')
        
        if any(word in text_lower for word in ['familia', 'niños', 'family', 'kids', 'children']):
            if 'naturaleza' not in interests:
                interests.append('naturaleza')
        
        if any(word in text_lower for word in ['foto', 'fotografía', 'photo', 'photography']):
            if 'cultural' not in interests:
                interests.append('cultural')
            if 'naturaleza' not in interests:
                interests.append('naturaleza')
        
        # Asegurar al menos un interés predeterminado si no se detectó ninguno
        if not interests:
            interests = ['cultural']  # Valor predeterminado
        
        return interests
    
    def _extract_special_occasion(self, text: str) -> Optional[str]:
        """Extrae ocasión especial del texto si existe"""
        text_lower = text.lower()
        
        # Detectar ocasiones especiales
        if "aniversario" in text_lower or "anniversary" in text_lower:
            return "anniversary"
        elif "luna de miel" in text_lower or "honeymoon" in text_lower:
            return "honeymoon"
        elif "cumpleaños" in text_lower or "birthday" in text_lower:
            return "birthday"
        elif "boda" in text_lower or "wedding" in text_lower:
            return "wedding"
            
        # Buscar con patrones
        for pattern in self.patterns.get("special_occasion", []):
            if re.search(pattern, text_lower):
                # Extraer el tipo de ocasión
                match = re.search(pattern, text_lower)
                if match:
                    return match.group(0)
                    
        return None
        
    def _infer_additional_interests(self, text: str, preferences: Dict[str, Any]) -> None:
        """Infiere intereses adicionales basados en el contexto de la consulta"""
        text_lower = text.lower()
        
        # Inferir interés romántico si es para pareja o aniversario
        if any(word in text_lower for word in ["romántico", "romantic", "pareja", "couple", "aniversario", "anniversary"]):
            if "romántico" not in preferences["interests"]:
                preferences["interests"].append("romántico")
                # Añadir interés gastronómico si hay menciones a cenas
                if any(word in text_lower for word in ["cena", "dinner", "restaurante", "restaurant"]):
                    if "gastronomía" not in preferences["interests"]:
                        preferences["interests"].append("gastronomía")
                        
        # Inferir interés cultural para fotografía
        if any(word in text_lower for word in ["fotografía", "photography", "foto", "photo", "escénico", "scenic"]):
            if "cultural" not in preferences["interests"]:
                preferences["interests"].append("cultural")
            if "naturaleza" not in preferences["interests"]:
                preferences["interests"].append("naturaleza")
                
        # Si no se detectaron intereses pero hay duración, añadir interés cultural como predeterminado
        if not preferences["interests"] and preferences["duration"]:
            preferences["interests"].append("cultural")
        
    def _has_children(self, text: str) -> bool:
        """Detecta si el viaje incluye niños"""
        text_lower = text.lower()
        child_indicators = [
            'niño', 'niños', 'hijo', 'hijos', 'familia', 'pequeño', 'pequeños', 'bebé',
            'child', 'children', 'kid', 'kids', 'family', 'baby', 'infant'
        ]
        
        return any(indicator in text_lower for indicator in child_indicators)
    
    def _classify_intent(self, text: str, doc: Any, language: str) -> str:
        """Clasifica la intención del usuario"""
        text_lower = text.lower()
        
        # Patrones de palabras clave por intención
        intent_patterns = {
            "information_seeking": [
                'información', 'info', 'saber', 'conocer', 'información sobre',
                'information', 'info', 'know', 'learn', 'tell me about'
            ],
            "recommendation_seeking": [
                'recomendar', 'sugerir', 'recomendación', 'sugerencia', 'recomienden',
                'recommend', 'suggest', 'recommendation', 'suggestion', 'advice'
            ],
            "planning": [
                'planear', 'planificar', 'organizar', 'preparar', 'plan', 'itinerario',
                'plan', 'schedule', 'organize', 'prepare', 'itinerary'
            ],
            "booking": [
                'reservar', 'reserva', 'disponibilidad', 'disponible', 'precio',
                'book', 'booking', 'availability', 'available', 'price', 'cost'
            ]
        }
        
        # Puntuar intenciones
        intent_scores = {intent: 0 for intent in intent_patterns}
        
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    intent_scores[intent] += 1
                    
        # Encontrar la intención con mayor puntuación
        max_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        # Si no hay una intención clara, usar recommendation_seeking como predeterminada
        if max_intent[1] == 0:
            return "recommendation_seeking"
            
        return max_intent[0]
    
    def _empty_result(self, query: str) -> Dict[str, Any]:
        """Devuelve un resultado vacío para consultas inválidas"""
        return {
            "query": query,
            "language": "unknown",
            "intent": "unknown",
            "preferences": {
                "interests": [],
                "locations": [],
                "budget": None,
                "duration": None,
                "group_size": None
            },
            "timestamp": datetime.now().isoformat()
        }