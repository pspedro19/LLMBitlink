import json
import re
from typing import Dict, Any, Optional, List
from openai import OpenAI
from openai import APITimeoutError, OpenAIError
from utils.logger import get_logger

logger = get_logger(__name__)

class OpenAINLPProcessor:
    def __init__(self, api_key: str):
        """Initialize the OpenAI NLP processor with API key"""
        self.client = OpenAI(api_key=api_key)
        self.timeout = 20  # Timeout en segundos para llamadas a OpenAI

    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process natural language text to extract travel preferences using OpenAI
        
        Args:
            text (str): User input text
            
        Returns:
            Dict[str, Any]: Extracted preferences in structured format
        """
        try:
            # Primer intento con prompt completo
            logger.info(f"Processing text (length: {len(text.split())} words)")
            result = self._try_openai_request(text, use_simplified=False)
            
            # Verificar si el resultado tiene información útil antes de intentar fallback
            if result and not self._is_empty_response(result):
                logger.debug("First attempt successful with valid data")
                return result
                    
            # Solo intentar fallback si el primer intento no produjo datos útiles
            logger.info("First attempt produced empty results, trying simplified prompt")
            result = self._try_openai_request(text, use_simplified=True)
            
            if result:
                return result
                
            logger.warning("All attempts failed, returning default response")
            return self._get_default_response(text)
            
        except Exception as e:
            logger.error(f"Unexpected error in process_text: {e}")
            return self._get_default_response(text)

    def _is_empty_response(self, result: Dict[str, Any]) -> bool:
        """Check if response is effectively empty"""
        preferences = result.get("preferences", {})
        return not any(
            val for val in preferences.values()
            if val not in (None, [], 0, "", 1)  # 1 es el valor por defecto para group_size
        )

    def _try_openai_request(self, text: str, use_simplified: bool = False) -> Optional[Dict[str, Any]]:
        """Attempt to process text with OpenAI with error handling"""
        try:
            # Calculate dynamic max_tokens based on input length
            words = len(text.split())
            max_tokens = min(1000, max(200, words * 5))  # Entre 200 y 1000 tokens
            
            logger.debug(f"Using max_tokens={max_tokens} for text of {words} words")
            
            # Create appropriate prompt based on attempt type
            prompt = (
                self._create_simplified_prompt(text) if use_simplified 
                else self._create_extraction_prompt(text)
            )
            
            # Make API call to OpenAI using client instance
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "Eres un experto en extracción de información sobre preferencias de viaje."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=max_tokens,
                timeout=self.timeout
            )
            
            result = response.choices[0].message.content
            logger.debug("OpenAI response received")
            
            parsed_result = self._extract_json(result)
            parsed_result["query"] = text
            
            return self._normalize_response(parsed_result)
            
        except APITimeoutError:
            logger.error("OpenAI API timeout")
            return None
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error in OpenAI request: {e}")
            return None

    def _create_simplified_prompt(self, text: str) -> str:
        """Create a simplified prompt for fallback attempts"""
        return f"""
        Analiza el siguiente texto y extrae la información relevante. Devuelve SOLO el siguiente JSON:

        {{
            "preferences": {{
                "locations": [],
                "budget_per_day": null,
                "trip_duration": null,
                "group_size": 1,
                "specific_sites": []
            }}
        }}

        Texto: {text}

        IMPORTANTE:
        - Usa valores predeterminados si falta información.
        - Devuelve SOLO el JSON en el formato exacto.
        """

    def _create_extraction_prompt(self, text: str) -> str:
        """Create a detailed and optimized prompt for OpenAI"""
        return f"""
        Analiza el siguiente texto y extrae información sobre preferencias de viaje. Devuelve SOLO el siguiente JSON:

        {{
            "preferences": {{
                "interests": [],
                "locations": [],
                "budget_per_day": null,
                "trip_duration": null,
                "group_size": 1,
                "activity_types": [],
                "specific_sites": [],
                "cuisine_preferences": []
            }}
        }}

        Texto: {text}

        IMPORTANTE:
        - Usa valores predeterminados si falta información.
        - Infiere intereses y actividades si no están explícitos.
        - Devuelve SOLO el JSON en el formato exacto.
        - Los campos numéricos deben ser números, no strings.
        - Incluye todos los sitios específicos mencionados.
        """

    def _extract_json(self, result: str) -> Dict[str, Any]:
        """Extract JSON block from a string response using regex"""
        try:
            json_match = re.search(r"\{.*\}", result, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
                
            return json.loads(json_match.group())
            
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            raise ValueError("Invalid JSON format in response")

    def _normalize_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and validate the response with improved validation"""
        try:
            preferences = response.get("preferences", {})
            
            # Validar tipo de preferences
            if not isinstance(preferences, dict):
                logger.warning("Field 'preferences' is not a dictionary, resetting to empty dict")
                preferences = {}
            
            # Normalize numeric fields with validation
            budget = self._to_float(preferences.get("budget_per_day"))
            if budget is not None and (budget < 0 or budget > 10000):
                logger.warning(f"Invalid budget_per_day value: {budget}, adjusting to valid range")
                preferences["budget_per_day"] = max(0, min(budget, 10000))
            else:
                preferences["budget_per_day"] = budget
            
            duration = self._to_int(preferences.get("trip_duration"))
            if duration is not None and (duration < 1 or duration > 90):
                logger.warning(f"Invalid trip_duration value: {duration}, adjusting to valid range")
                preferences["trip_duration"] = max(1, min(duration, 90))
            else:
                preferences["trip_duration"] = duration
            
            group_size = self._to_int(preferences.get("group_size"))
            if group_size is not None and (group_size < 1 or group_size > 50):
                logger.warning(f"Invalid group_size value: {group_size}, adjusting to valid range")
                preferences["group_size"] = max(1, min(group_size, 50))
            else:
                preferences["group_size"] = group_size if group_size is not None else 1
            
            # Normalize list fields efficiently
            list_fields = [
                "interests", "locations", "activity_types",
                "specific_sites", "cuisine_preferences"
            ]
            
            for field in list_fields:
                current_value = preferences.get(field)
                if not isinstance(current_value, list):
                    logger.warning(f"Field '{field}' is not a list, resetting to empty list")
                    preferences[field] = []
                else:
                    preferences[field] = self._to_list(current_value)
            
            logger.debug("Response normalized successfully")
            return {
                "query": response.get("query", ""),
                "preferences": preferences
            }
            
        except Exception as e:
            logger.error(f"Error in _normalize_response: {e}")
            raise ValueError(f"Failed to normalize response: {str(e)}")

    def _to_float(self, value: Any) -> Optional[float]:
        """Convert value to float if possible"""
        try:
            if value is not None:
                return float(value)
            return None
        except (ValueError, TypeError):
            return None

    def _to_int(self, value: Any) -> Optional[int]:
        """Convert value to integer if possible"""
        try:
            if value is not None:
                return int(value)
            return None
        except (ValueError, TypeError):
            return None

    def _to_list(self, value: Any) -> List[str]:
        """Convert value to list of unique, normalized strings"""
        if isinstance(value, list) and value:
            # Convertir a set para eliminar duplicados y volver a lista
            return list(set(
                str(item).strip().lower()  # Normalizar strings
                for item in value
                if item is not None and str(item).strip()
            ))
        return []

    def _get_default_response(self, text: str) -> Dict[str, Any]:
        """Return a default response structure"""
        return {
            "query": text,
            "preferences": {
                "interests": [],
                "locations": [],
                "budget_per_day": None,
                "trip_duration": None,
                "group_size": 1,
                "activity_types": [],
                "specific_sites": [],
                "cuisine_preferences": []
            }
        }