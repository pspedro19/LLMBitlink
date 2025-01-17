"""
Data models and validation logic for the tourism recommendation system.
"""
import re
from typing import Dict, List, Optional, Union
from functools import lru_cache
from datetime import date
from pydantic import BaseModel, Field, validator
from fuzzywuzzy import process, fuzz
from app.utils.logger import get_logger
from app.core.utils.validators import DataValidator  # Cambiar la importación


logger = get_logger(__name__)

class TourismError(Exception):
    """Custom error class for tourism system"""
    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.message = message
        super().__init__(self.message)


class EnhancedPreferences(BaseModel):
    """Enhanced model for tourism preferences with validation"""
    interests: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    budget_per_day: Optional[float] = Field(None, ge=0)
    trip_duration: Optional[int] = Field(None, ge=1)
    accommodation_type: Optional[str] = None
    preferred_activities: List[str] = Field(default_factory=list)
    accessibility_needs: Optional[str] = None
    language_preferences: List[str] = Field(default_factory=lambda: ["Spanish"])
    cuisine_preferences: List[str] = Field(default_factory=list)
    group_size: int = Field(1, ge=1)
    has_children: bool = False
    travel_dates: Optional[Dict[str, str]] = None
    max_walking_distance: Optional[float] = None
    preferred_transportation: Optional[str] = None
    dietary_preferences: List[str] = Field(default_factory=list)

    @validator('interests')
    def validate_interests(cls, v: List[str]) -> List[str]:
        """Validate and normalize interests"""
        return [DataValidator.normalize_interest(interest) for interest in v if interest]

    @validator('locations')
    def validate_locations(cls, v: List[str]) -> List[str]:
        """Validate and normalize locations"""
        return [DataValidator.normalize_location(location) for location in v if location]

    @validator('budget_per_day')
    def validate_budget(cls, v: Optional[float]) -> Optional[float]:
        """Ensure budget is positive if provided"""
        if v is not None and v <= 0:
            raise ValueError("Budget must be positive")
        return v

    @validator('travel_dates')
    def validate_travel_dates(cls, v: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """Validate travel dates format and logical correctness"""
        if v:
            try:
                start_date = date.fromisoformat(v.get('start', ''))
                end_date = date.fromisoformat(v.get('end', ''))
                if start_date >= end_date:
                    raise ValueError("End date must be after start date")
            except Exception:
                raise ValueError("Invalid travel dates format. Use 'YYYY-MM-DD'")
        return v

    @validator('language_preferences', 'cuisine_preferences', 'dietary_preferences', pre=True, always=True)
    def ensure_lists(cls, v):
        """Ensure these fields are always lists"""
        if not isinstance(v, list):
            raise ValueError("Value must be a list")
        return v

    @validator('preferred_transportation')
    def validate_transportation(cls, v: Optional[str]) -> Optional[str]:
        """Validate preferred transportation method"""
        valid_transportation = ['car', 'bus', 'bike', 'walk', 'train', 'plane']
        if v and v.lower() not in valid_transportation:
            raise ValueError(f"Invalid transportation method: {v}")
        return v