from enum import Enum
from typing import Tuple, Dict, Set, List, Optional, Any
import spacy
from functools import lru_cache
import logging
from dataclasses import dataclass
import os
import yaml
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class IntentType(Enum):
    GREETING = "greeting"
    IDENTITY = "identity"
    RECOMMENDATION = "recommendation"
    SPECIFIC_QUESTION = "specific_question"
    MIXED = "mixed"
    UNKNOWN = "unknown"

@dataclass
class IntentCombination:
    """Define una combinación específica de intenciones."""
    primary: IntentType
    secondary: IntentType
    weight: float = 1.0
    require_both: bool = False

@dataclass
class IntentResult:
    """Estructura para almacenar los resultados del análisis de intenciones."""
    primary_intent: IntentType
    confidence: float
    secondary_intents: List[Tuple[IntentType, float]]
    detected_patterns: Dict[str, List[str]]
    intent_combination: Optional[IntentCombination] = None
    analysis_metadata: Dict[str, Any] = None

class IntentAnalyzerConfig:
    """Configuración para el analizador de intenciones."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa la configuración del analizador.
        
        Args:
            config_path: Ruta opcional al archivo de configuración YAML
        """
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Carga la configuración desde YAML o usa valores por defecto."""
        default_config = {
            'confidence_thresholds': {
                'high': 0.8,
                'medium': 0.5,
                'low': 0.3
            },
            'intent_combinations': [
                {
                    'primary': 'GREETING',
                    'secondary': 'RECOMMENDATION',
                    'weight': 1.2,
                    'require_both': False
                },
                {
                    'primary': 'SPECIFIC_QUESTION',
                    'secondary': 'RECOMMENDATION',
                    'weight': 1.5,
                    'require_both': True
                }
            ],
            'model_settings': {
                'spacy_model': 'es_core_news_sm',
                'use_transformer': False,
                'cache_size': 1000
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                return {**default_config, **loaded_config}
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        return default_config

class PatternLoader:
    """Cargador de patrones desde archivos externos."""
    
    @staticmethod
    def load_patterns(patterns_dir: Optional[str] = None) -> Dict[str, Dict[str, Set[str]]]:
        """
        Carga patrones desde archivos JSON o usa valores por defecto.
        
        Args:
            patterns_dir: Directorio opcional con archivos de patrones
        """
        default_patterns = {
            'greeting': {
                'basic': {
                    'hola', 'hello', 'hi', 'buenos días', 'buenas tardes',
                    'buenas noches', 'saludos', 'que tal'
                }
            },
            'identity': {
                'questions': {
                    'quien eres', 'quién eres', 'que eres', 'qué eres',
                    'como te llamas', 'cómo te llamas'
                }
            },
            'recommendation': {
                'general': {
                    'recomendar', 'sugerir', 'proponer', 'aconsejar'
                },
                'activities': {
                    'hacer', 'visitar', 'ver', 'explorar', 'conocer'
                },
                'places': {
                    'lugar', 'sitio', 'punto', 'zona', 'área'
                }
            }
        }
        
        if patterns_dir and os.path.exists(patterns_dir):
            try:
                patterns = {}
                for file in Path(patterns_dir).glob('*.json'):
                    with open(file, 'r') as f:
                        category = file.stem
                        patterns[category] = json.load(f)
                return patterns
            except Exception as e:
                logger.error(f"Error loading patterns: {e}")
        
        return default_patterns

class IntentAnalyzer:
    """Analizador avanzado de intenciones usando spaCy y patrones."""
    
    def __init__(self, config_path: Optional[str] = None, patterns_dir: Optional[str] = None):
        """
        Inicializa el analizador con configuración y patrones personalizables.
        
        Args:
            config_path: Ruta opcional al archivo de configuración
            patterns_dir: Directorio opcional con archivos de patrones
        """
        # Cargar configuración
        self.config = IntentAnalyzerConfig(config_path).config
        
        # Cargar patrones
        self.patterns = PatternLoader.load_patterns(patterns_dir)
        
        # Inicializar spaCy con manejo de errores mejorado
        self.nlp = self._initialize_nlp()
        
        # Configurar combinaciones de intenciones
        self.intent_combinations = [
            IntentCombination(
                primary=IntentType[combo['primary']],
                secondary=IntentType[combo['secondary']],
                weight=combo['weight'],
                require_both=combo['require_both']
            )
            for combo in self.config['intent_combinations']
        ]
        
        # Métricas y estado
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'intent_distribution': {intent.value: 0 for intent in IntentType}
        }
    
    def _initialize_nlp(self) -> spacy.language.Language:
        """
        Inicializa el modelo spaCy con mejor manejo de errores.
        
        Returns:
            spacy.language.Language: Modelo spaCy inicializado
        
        Raises:
            RuntimeError: Si no se puede cargar ni instalar el modelo
        """
        model_name = self.config['model_settings']['spacy_model']
        try:
            return spacy.load(model_name)
        except OSError:
            logger.warning(f"Modelo {model_name} no encontrado")
            try:
                spacy.cli.download(model_name)
                return spacy.load(model_name)
            except Exception as e:
                logger.error(f"Error instalando modelo spaCy: {str(e)}")
                # Intentar cargar modelo más básico como fallback
                try:
                    return spacy.load('es_core_news_md')
                except:
                    raise RuntimeError(
                        f"No se pudo cargar ningún modelo spaCy. Error original: {str(e)}"
                    )
                    
    @lru_cache(maxsize=1000)
    def analyze_intent(self, text: str) -> IntentResult:
        """
        Analiza el texto para determinar intenciones con manejo mejorado de casos mixtos.
        """
        try:
            # Incrementar métricas
            self.metrics['total_requests'] += 1
            
            text = text.lower().strip()
            doc = self.nlp(text)
            
            # Detectar todos los patrones
            detected_patterns = self._detect_all_patterns(text, doc)
            
            # Analizar intenciones y calcular scores
            intent_scores = self._calculate_intent_scores(text, doc, detected_patterns)
            
            # Ordenar intenciones por puntuación
            sorted_intents = sorted(
                [(intent, score) for intent, score in intent_scores.items() if score > 0],
                key=lambda x: x[1],
                reverse=True
            )
            
            if not sorted_intents:
                return IntentResult(
                    primary_intent=IntentType.UNKNOWN,
                    confidence=0.0,
                    secondary_intents=[],
                    detected_patterns=detected_patterns,
                    intent_combination=None,
                    analysis_metadata={
                        'spacy_entities': [(ent.text, ent.label_) for ent in doc.ents],
                        'confidence_by_category': self._get_confidence_by_category(intent_scores)
                    }
                )
            
            # Verificar combinaciones predefinidas
            for combination in self.intent_combinations:
                primary_score = intent_scores[combination.primary]
                secondary_score = intent_scores[combination.secondary]
                
                # Si ambas intenciones tienen puntuación significativa
                if (primary_score > self.config['confidence_thresholds']['medium'] and 
                    secondary_score > self.config['confidence_thresholds']['low']):
                    
                    if (combination.require_both and 
                        primary_score > self.config['confidence_thresholds']['high'] and 
                        secondary_score > self.config['confidence_thresholds']['medium']):
                        # Combinar puntuaciones según el peso de la combinación
                        combined_score = (primary_score + secondary_score * combination.weight) / (1 + combination.weight)
                        
                        return IntentResult(
                            primary_intent=IntentType.MIXED,
                            confidence=combined_score,
                            secondary_intents=sorted_intents[1:],
                            detected_patterns=detected_patterns,
                            intent_combination=combination,
                            analysis_metadata={
                                'spacy_entities': [(ent.text, ent.label_) for ent in doc.ents],
                                'confidence_by_category': self._get_confidence_by_category(intent_scores)
                            }
                        )
            
            # Si no hay combinación específica pero hay múltiples intenciones fuertes
            if (len(sorted_intents) > 1 and 
                sorted_intents[1][1] > self.config['confidence_thresholds']['medium']):
                return IntentResult(
                    primary_intent=IntentType.MIXED,
                    confidence=sorted_intents[0][1],
                    secondary_intents=sorted_intents[1:],
                    detected_patterns=detected_patterns,
                    intent_combination=None,
                    analysis_metadata={
                        'spacy_entities': [(ent.text, ent.label_) for ent in doc.ents],
                        'confidence_by_category': self._get_confidence_by_category(intent_scores)
                    }
                )
            
            # Caso simple: una intención dominante
            primary_intent, confidence = sorted_intents[0]
            secondary_intents = sorted_intents[1:] if len(sorted_intents) > 1 else []
            
            return IntentResult(
                primary_intent=primary_intent,
                confidence=confidence,
                secondary_intents=secondary_intents,
                detected_patterns=detected_patterns,
                intent_combination=None,
                analysis_metadata={
                    'spacy_entities': [(ent.text, ent.label_) for ent in doc.ents],
                    'confidence_by_category': self._get_confidence_by_category(intent_scores)
                }
            )
            
        except Exception as e:
            logger.error(f"Error en análisis de intención: {str(e)}")
            return IntentResult(
                primary_intent=IntentType.UNKNOWN,
                confidence=0.0,
                secondary_intents=[],
                detected_patterns={},
                intent_combination=None,
                analysis_metadata={
                    'error': str(e),
                    'spacy_entities': [],
                    'confidence_by_category': {}
                }
            )
    
    def _detect_all_patterns(self, text: str, doc: spacy.tokens.Doc) -> Dict[str, List[str]]:
        """
        Detecta patrones en el texto con mejor organización y manejo de errores.
        
        Args:
            text: Texto a analizar
            doc: Documento spaCy procesado
            
        Returns:
            Dict[str, List[str]]: Patrones detectados por categoría
        """
        patterns = {}
        
        try:
            for category, subcategories in self.patterns.items():
                patterns[category] = {}
                for subcategory, pattern_set in subcategories.items():
                    matches = []
                    for pattern in pattern_set:
                        if self._pattern_matches(pattern, text, doc):
                            matches.append(pattern)
                    if matches:
                        patterns[category][subcategory] = matches
        except Exception as e:
            logger.error(f"Error detectando patrones: {str(e)}")
            # Retornar diccionario vacío en caso de error
            return {}
            
        return patterns
    
    def _pattern_matches(self, pattern: str, text: str, doc: spacy.tokens.Doc) -> bool:
        """
        Verifica si un patrón coincide con el texto usando diferentes métodos.
        
        Args:
            pattern: Patrón a buscar
            text: Texto donde buscar
            doc: Documento spaCy procesado
            
        Returns:
            bool: True si hay coincidencia
        """
        # Coincidencia exacta
        if pattern in text:
            return True
            
        # Coincidencia por lema
        pattern_tokens = self.nlp(pattern)
        for token in pattern_tokens:
            if any(t.lemma_ == token.lemma_ for t in doc):
                return True
                
        return False
    
    def _calculate_intent_scores(self, text: str, doc: spacy.tokens.Doc,
                               patterns: Dict[str, Dict[str, List[str]]]) -> Dict[IntentType, float]:
        """
        Calcula puntuaciones para cada tipo de intención con pesos ajustables.
        
        Args:
            text: Texto original
            doc: Documento spaCy procesado
            patterns: Patrones detectados
            
        Returns:
            Dict[IntentType, float]: Puntuaciones por tipo de intención
        """
        scores = {intent_type: 0.0 for intent_type in IntentType}
        
        try:
            # Calcular scores base por patrones
            for category, subcategories in patterns.items():
                if category == 'greeting' and subcategories:
                    scores[IntentType.GREETING] = self._calculate_category_score(subcategories)
                elif category == 'identity' and subcategories:
                    scores[IntentType.IDENTITY] = self._calculate_category_score(subcategories)
                elif category == 'recommendation' and subcategories:
                    scores[IntentType.RECOMMENDATION] = (
                        self._calculate_recommendation_score(subcategories)
                    )
            
            # Detectar preguntas específicas
            question_score = self._calculate_question_score(text, doc)
            scores[IntentType.SPECIFIC_QUESTION] = question_score
            
        except Exception as e:
            logger.error(f"Error calculando scores: {str(e)}")
            # En caso de error, retornar scores en 0
            return scores
        
        return scores
    
    def _calculate_category_score(self, subcategories: Dict[str, List[str]]) -> float:
        """
        Calcula el score para una categoría basado en sus subcategorías.
        
        Args:
            subcategories: Subcategorías con sus coincidencias
            
        Returns:
            float: Score calculado
        """
        total_matches = sum(len(matches) for matches in subcategories.values())
        max_matches = max(len(matches) for matches in subcategories.values())
        
        # Combinar cantidad de coincidencias con intensidad por subcategoría
        return min(total_matches * 0.3 + max_matches * 0.7, 1.0)
    
    def _calculate_recommendation_score(self, subcategories: Dict[str, List[str]]) -> float:
        """
        Calcula el score para recomendaciones con pesos por subcategoría.
        
        Args:
            subcategories: Subcategorías con sus coincidencias
            
        Returns:
            float: Score calculado
        """
        weights = {
            'general': 1.0,
            'activities': 0.8,
            'places': 0.8,
            'experiences': 0.7
        }
        
        weighted_score = sum(
            len(matches) * weights.get(subcat, 0.5)
            for subcat, matches in subcategories.items()
        )
        
        return min(weighted_score / 3, 1.0)
    
    def _calculate_question_score(self, text: str, doc: spacy.tokens.Doc) -> float:
        """
        Calcula el score para preguntas específicas.
        
        Args:
            text: Texto original
            doc: Documento spaCy procesado
            
        Returns:
            float: Score calculado
        """
        question_indicators = {
            'marks': '?' in text,
            'words': any(token.text.lower() in self.patterns.get('question', {}).get('words', set())
                        for token in doc),
            'structure': any(token.dep_ == 'ROOT' and token.pos_ == 'VERB'
                           for token in doc)
        }
        
        score = sum([
            0.6 if question_indicators['marks'] else 0,
            0.3 if question_indicators['words'] else 0,
            0.1 if question_indicators['structure'] else 0
        ])
        
        return min(score, 1.0)
    
    def _determine_intents_with_combinations(
        self, 
        intent_scores: Dict[IntentType, float]
    ) -> Tuple[IntentType, float, List[Tuple[IntentType, float]], Optional[IntentCombination]]:
        """
        Determina las intenciones primarias y secundarias, considerando combinaciones predefinidas.
        
        Args:
            intent_scores: Puntuaciones por tipo de intención
        
        Returns:
            Tuple con:
            - Intención primaria
            - Nivel de confianza
            - Lista de intenciones secundarias con sus niveles de confianza
            - Combinación de intenciones detectada (si existe)
        """
        try:
            # Ordenar intenciones por puntuación
            sorted_intents = sorted(
                [(intent, score) for intent, score in intent_scores.items() if score > 0],
                key=lambda x: x[1],
                reverse=True
            )
            
            if not sorted_intents:
                return IntentType.UNKNOWN, 0.0, [], None
            
            # Verificar combinaciones predefinidas
            for combination in self.intent_combinations:
                primary_score = intent_scores[combination.primary]
                secondary_score = intent_scores[combination.secondary]
                
                # Si ambas intenciones tienen puntuación significativa
                if primary_score > self.config['confidence_thresholds']['medium'] and \
                   secondary_score > self.config['confidence_thresholds']['low']:
                    
                    if combination.require_both and \
                       primary_score > self.config['confidence_thresholds']['high'] and \
                       secondary_score > self.config['confidence_thresholds']['medium']:
                        # Combinar puntuaciones según el peso de la combinación
                        combined_score = (primary_score + secondary_score * combination.weight) / (1 + combination.weight)
                        return (
                            IntentType.MIXED,
                            combined_score,
                            sorted_intents[1:],
                            combination
                        )
            
            # Si no hay combinación específica pero hay múltiples intenciones fuertes
            if len(sorted_intents) > 1 and \
               sorted_intents[1][1] > self.config['confidence_thresholds']['medium']:
                return (
                    IntentType.MIXED,
                    sorted_intents[0][1],
                    sorted_intents[1:],
                    None
                )
            
            # Caso simple: una intención dominante
            primary_intent, confidence = sorted_intents[0]
            secondary_intents = sorted_intents[1:] if len(sorted_intents) > 1 else []
            
            return primary_intent, confidence, secondary_intents, None
            
        except Exception as e:
            logger.error(f"Error determinando intenciones: {str(e)}")
            return IntentType.UNKNOWN, 0.0, [], None
    
    def _get_confidence_by_category(self, intent_scores: Dict[IntentType, float]) -> Dict[str, Dict[str, float]]:
        """
        Organiza los niveles de confianza por categoría para análisis detallado.
        
        Args:
            intent_scores: Puntuaciones por tipo de intención
            
        Returns:
            Dict[str, Dict[str, float]]: Niveles de confianza organizados por categoría
        """
        confidence_levels = {
            'high': self.config['confidence_thresholds']['high'],
            'medium': self.config['confidence_thresholds']['medium'],
            'low': self.config['confidence_thresholds']['low']
        }
        
        result = {
            'by_threshold': {
                'high': [],
                'medium': [],
                'low': []
            },
            'by_intent': {
                intent_type.value: score 
                for intent_type, score in intent_scores.items()
            }
        }
        
        # Clasificar intenciones por nivel de confianza
        for intent_type, score in intent_scores.items():
            if score >= confidence_levels['high']:
                result['by_threshold']['high'].append(intent_type.value)
            elif score >= confidence_levels['medium']:
                result['by_threshold']['medium'].append(intent_type.value)
            elif score >= confidence_levels['low']:
                result['by_threshold']['low'].append(intent_type.value)
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas de uso del analizador.
        
        Returns:
            Dict[str, Any]: Métricas recopiladas
        """
        return {
            'total_requests': self.metrics['total_requests'],
            'cache_hits': self.metrics['cache_hits'],
            'intent_distribution': self.metrics['intent_distribution'],
            'cache_hit_rate': (
                self.metrics['cache_hits'] / self.metrics['total_requests']
                if self.metrics['total_requests'] > 0 else 0
            )
        }
    
    def reset_metrics(self):
        """Reinicia las métricas a sus valores iniciales."""
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'intent_distribution': {intent.value: 0 for intent in IntentType}
        }

# Configuración por defecto en YAML
DEFAULT_CONFIG = """
confidence_thresholds:
  high: 0.8
  medium: 0.5
  low: 0.3

intent_combinations:
  - primary: GREETING
    secondary: RECOMMENDATION
    weight: 1.2
    require_both: false
  
  - primary: SPECIFIC_QUESTION
    secondary: RECOMMENDATION
    weight: 1.5
    require_both: true
  
  - primary: IDENTITY
    secondary: GREETING
    weight: 1.0
    require_both: false

model_settings:
  spacy_model: es_core_news_sm
  use_transformer: false
  cache_size: 1000
"""