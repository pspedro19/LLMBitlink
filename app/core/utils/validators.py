import re
from typing import Dict, List, Optional, Union, Any
from functools import lru_cache
from datetime import date
from pydantic import BaseModel, Field, validator
from fuzzywuzzy import process, fuzz
from utils.logger import get_logger




class DataValidator:
    """Enhanced validator with improved normalization and matching"""

    VALID_CATEGORIES = {
        'interests': [
            'cultural', 'adventure', 'nature', 'food', 'water_sports',
            'history', 'shopping', 'nightlife', 'relaxation', 'gastronomy',
            'culinary', 'walking', 'historical', 'diving', 'snorkeling',
            'hiking', 'photography', 'art', 'crafts', 'museums', 'architecture',
            'local_cuisine', 'watersports', 'beach', 'marine_life'
        ],
        'locations': [
            'willemstad', 'punda', 'otrobanda', 'westpunt', 'christoffel park',
            'spanish water', 'pietermaai', 'jan thiel', 'banda abou', 'shete boka',
            'piscadera bay', 'caracasbaai', 'grote knip', 'playa kalki',
            'blue room cave', 'plasa bieu', 'floating market', 'soto', 'barber'
        ],
        'cuisine_types': [
            'local', 'international', 'seafood', 'vegetarian',
            'fusion', 'caribbean', 'krioyo', 'mediterranean', 'asian'
        ],
        'price_ranges': {
            'low': {'max': 50.0, 'description': 'Budget-friendly'},
            'medium': {'max': 150.0, 'description': 'Moderate'},
            'high': {'max': 300.0, 'description': 'Premium'}
        }
    }

    INTEREST_MAPPINGS = {
        'cultural': [
            'cultural', 'culture', 'historical', 'history', 'heritage', 'museum',
            'architecture', 'colonial', 'monuments', 'traditional', 'landmark',
            'archaeological', 'ruins', 'historical site', 'cultural heritage',
            'fort', 'castle', 'palace', 'historic district', 'old town',
            'historic center', 'cultural tour', 'guided tour', 'cultural experience'
        ],
        'gastronomy': [
            'gastronomy', 'gastronomic', 'food', 'culinary', 'dining', 'restaurant',
            'cuisine', 'eating', 'foodie', 'local food', 'traditional food', 'krioyo',
            'taste', 'flavors', 'cooking', 'food tour', 'food tasting', 'street food',
            'gourmet', 'authentic cuisine', 'local dishes', 'food market', 'fresh market',
            'seafood', 'fish market', 'food festival', 'wine tasting', 'cooking class',
            'chef', 'local specialties', 'food experience', 'farm to table'
        ],
        'history': [
            'history', 'historical', 'heritage', 'past', 'cultural', 'ancient',
            'traditional', 'colonial', 'landmark', 'archaeology', 'artifacts',
            'historic preservation', 'historic building', 'historic site',
            'plantation house', 'slavery history', 'maritime history', 'colonial era',
            'dutch heritage', 'trading post', 'historical museum', 'historic district',
            'cultural heritage', 'historic architecture', 'restoration'
        ],
        'walking': [
            'walking', 'walk', 'tour', 'tours', 'walking tour', 'on foot',
            'guided walk', 'city walk', 'sightseeing', 'exploration', 'stroll',
            'wandering', 'pedestrian', 'historic walk', 'nature walk', 'urban walk',
            'walking trail', 'guided tour', 'self-guided tour', 'walking path',
            'promenade', 'boardwalk', 'city exploration', 'walking route',
            'neighborhood walk', 'street art walk'
        ],
        'water_activities': [
            'diving', 'snorkel', 'swim', 'beach', 'marine', 'underwater', 'reef',
            'sea', 'ocean', 'watersports', 'kayak', 'sailing', 'boat', 'surfing',
            'paddleboarding', 'jet ski', 'waterski', 'fishing', 'deep sea fishing',
            'scuba', 'coral reef', 'marine life', 'boat tour', 'catamaran',
            'yacht', 'cruise', 'swimming', 'snorkeling', 'water sports',
            'aquatic activities', 'beach activities', 'maritime'
        ]
    }

    @classmethod
    @lru_cache(maxsize=128)
    def normalize_interest(cls, value: str) -> Optional[str]:
        """
        Enhanced interest normalization with fuzzy matching
        
        Args:
            value (str): Interest value to normalize
            
        Returns:
            Optional[str]: Normalized interest or None if invalid
        """
        if not value or not isinstance(value, str):
            return None

        value = value.lower().strip()

        # Direct category match
        if value in cls.VALID_CATEGORIES['interests']:
            return value

        # Check mappings with expanded variants
        for normalized, variants in cls.INTEREST_MAPPINGS.items():
            if value in variants or any(variant in value for variant in variants):
                return normalized

        # Fuzzy matching with higher threshold
        matches = process.extractBests(
            value,
            cls.VALID_CATEGORIES['interests'],
            score_cutoff=85,
            scorer=fuzz.token_sort_ratio
        )
        return matches[0][0] if matches else None

    @classmethod
    @lru_cache(maxsize=128)
    def normalize_location(cls, value: str) -> Optional[str]:
        """
        Enhanced location normalization with improved validation
        
        Args:
            value (str): Location value to normalize
            
        Returns:
            Optional[str]: Normalized location or None if invalid
        """
        if not value or not isinstance(value, str):
            return None

        value = value.lower().strip()
        value = value.replace('-', ' ').replace('_', ' ')

        if not cls.validate_location(value):
            return None

        if value in cls.VALID_CATEGORIES['locations']:
            return value

        for location in cls.VALID_CATEGORIES['locations']:
            if location in value:
                return location

        matches = process.extractBests(
            value,
            cls.VALID_CATEGORIES['locations'],
            score_cutoff=85,
            scorer=fuzz.token_sort_ratio
        )
        return matches[0][0] if matches else None

    @staticmethod
    def validate_location(location: str) -> bool:
        """
        Improved location validation with enhanced checks
        
        Args:
            location (str): Location to validate
            
        Returns:
            bool: True if location is valid, False otherwise
        """
        if not location or not isinstance(location, str):
            return False

        invalid_words = {
            'budget', 'cost', 'price', 'day', 'per', 'usd', 'dollar',
            'euro', 'cheap', 'expensive', 'free'
        }

        location_words = set(location.lower().split())
        if location_words.intersection(invalid_words):
            return False

        if len(location.strip()) < 2:
            return False

        valid_chars = set('abcdefghijklmnopqrstuvwxyz0123456789 -')
        if not all(char in valid_chars for char in location.lower()):
            return False

        return True

    @classmethod
    @lru_cache(maxsize=128)
    def normalize_budget(cls, budget_str: Union[str, float, int]) -> Optional[float]:
        """
        Enhanced budget normalization with comprehensive handling
        
        Args:
            budget_str: Budget value to normalize (string, float, or int)
            
        Returns:
            Optional[float]: Normalized budget value or None if invalid
        """
        if not budget_str:
            return None

        if isinstance(budget_str, (int, float)):
            return float(budget_str) if 0 < float(budget_str) <= 10000 else None

        budget_str = str(budget_str).lower().strip()

        if budget_str in cls.VALID_CATEGORIES['price_ranges']:
            return cls.VALID_CATEGORIES['price_ranges'][budget_str]['max']

        patterns = [
            r'(?:USD|\$|€)\s*(\d+(?:\.\d{2})?)',
            r'(\d+(?:\.\d{2})?)\s*(?:dollars|euro)',
            r'(?:budget|cost|price|spend).*?(\d+)',
            r'(\d+)(?:\s*(?:per|a|each))?\s*(?:day|person|pax)',
            r'(\d+(?:\.\d{2})?)'
        ]

        for pattern in patterns:
            match = re.search(pattern, budget_str, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    if 0 < value <= 10000:
                        return value
                except ValueError:
                    continue

        return None

    @classmethod
    def get_price_range(cls, amount: float) -> str:
        """
        Get price range category for a given amount
        
        Args:
            amount (float): Amount to categorize
            
        Returns:
            str: Price range category
        """
        for range_name, range_info in cls.VALID_CATEGORIES['price_ranges'].items():
            if amount <= range_info['max']:
                return range_name
        return 'high'

    @classmethod
    def validate_preferences(cls, preferences: Dict[str, Any]) -> Dict[str, Any]:
        validated = {}
        
        # Validar intereses
        if preferences.get('interests'):
            validated['interests'] = [
                interest for interest in preferences['interests']
                if cls.normalize_interest(interest)
            ]
        
        # Validar ubicaciones
        if preferences.get('locations'):
            validated['locations'] = [
                loc for loc in preferences['locations']
                if cls.normalize_location(loc)
            ]
        
        # Validar presupuesto
        if preferences.get('budget_per_day'):
            budget = cls.normalize_budget(preferences['budget_per_day'])
            if budget:
                validated['budget_per_day'] = budget
                
        # Validar campos numéricos
        numeric_fields = ['trip_duration', 'group_size']
        for field in numeric_fields:
            if value := preferences.get(field):
                try:
                    num_value = float(value)
                    if num_value > 0:
                        validated[field] = num_value
                except ValueError:
                    continue
                    
        # Mantener campos opcionales si existen
        optional_fields = ['specific_sites', 'cuisine_preferences']
        for field in optional_fields:
            if value := preferences.get(field):
                validated[field] = value

        return validated