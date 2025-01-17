"""
Enhanced preference model with comprehensive validation for tourism preferences.
"""

from typing import Dict, List, Optional, Any
from datetime import date
from pydantic import BaseModel, Field, validator
from app.core.utils.validators import DataValidator  # Cambiar la importación



class TourismPreferenceError(Exception):
    """Custom exception for tourism preference validation errors"""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class EnhancedPreferences(BaseModel):
    """Enhanced model for tourism preferences with validation"""
    
    # Core preferences
    interests: List[str] = Field(
        default_factory=list,
        description="List of tourist interests (e.g., cultural, adventure, nature)"
    )
    
    locations: List[str] = Field(
        default_factory=list,
        description="List of preferred locations to visit"
    )
    
    budget_per_day: Optional[float] = Field(
        None, 
        ge=0,
        description="Maximum budget per day in USD"
    )
    
    trip_duration: Optional[int] = Field(
        None, 
        ge=1,
        description="Duration of trip in days"
    )
    
    # Accommodation and accessibility
    accommodation_type: Optional[str] = Field(
        None,
        description="Preferred type of accommodation (e.g., hotel, apartment)"
    )
    
    accessibility_needs: Optional[str] = Field(
        None,
        description="Any specific accessibility requirements"
    )
    
    max_walking_distance: Optional[float] = Field(
        None,
        description="Maximum walking distance in kilometers"
    )
    
    # Group and preferences
    group_size: int = Field(
        1, 
        ge=1,
        description="Number of people in the travel group"
    )
    
    has_children: bool = Field(
        False,
        description="Whether children are part of the travel group"
    )
    
    # Activities and preferences
    preferred_activities: List[str] = Field(
        default_factory=list,
        description="List of specific activities of interest"
    )
    
    language_preferences: List[str] = Field(
        default_factory=lambda: ["Spanish"],
        description="Preferred languages for tours and activities"
    )
    
    cuisine_preferences: List[str] = Field(
        default_factory=list,
        description="Preferred types of cuisine"
    )
    
    dietary_preferences: List[str] = Field(
        default_factory=list,
        description="Special dietary requirements or preferences"
    )
    
    # Transportation
    preferred_transportation: Optional[str] = Field(
        None,
        description="Preferred mode of transportation"
    )
    
    # Travel dates
    travel_dates: Optional[Dict[str, str]] = Field(
        None,
        description="Travel dates in format {'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}"
    )

    @validator('interests')
    def validate_interests(cls, v: List[str]) -> List[str]:
        """Validate and normalize interests"""
        normalized = [
            interest for interest in v 
            if DataValidator.normalize_interest(interest)
        ]
        if not normalized and v:
            raise TourismPreferenceError("No valid interests found in the provided list")
        return normalized

    @validator('locations')
    def validate_locations(cls, v: List[str]) -> List[str]:
        """Validate and normalize locations"""
        normalized = [
            location for location in v 
            if DataValidator.normalize_location(location)
        ]
        if not normalized and v:
            raise TourismPreferenceError("No valid locations found in the provided list")
        return normalized

    @validator('budget_per_day')
    def validate_budget(cls, v: Optional[float]) -> Optional[float]:
        """Ensure budget is positive if provided"""
        if v is not None:
            if v <= 0:
                raise ValueError("Budget must be positive")
            if v > 10000:  # Reasonable upper limit
                raise ValueError("Budget exceeds maximum reasonable amount")
        return v

    @validator('trip_duration')
    def validate_duration(cls, v: Optional[int]) -> Optional[int]:
        """Validate trip duration"""
        if v is not None:
            if v <= 0:
                raise ValueError("Trip duration must be positive")
            if v > 90:  # Reasonable upper limit
                raise ValueError("Trip duration exceeds maximum reasonable length")
        return v

    @validator('travel_dates')
    def validate_travel_dates(cls, v: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """Validate travel dates format and logical correctness"""
        if v:
            required_keys = {'start', 'end'}
            if not all(key in v for key in required_keys):
                raise ValueError("Travel dates must include both 'start' and 'end' dates")
            
            try:
                start_date = date.fromisoformat(v.get('start', ''))
                end_date = date.fromisoformat(v.get('end', ''))
                
                if start_date >= end_date:
                    raise ValueError("End date must be after start date")
                
                # Validate reasonable date range
                date_difference = (end_date - start_date).days
                if date_difference > 90:  # 3 months max
                    raise ValueError("Travel period exceeds maximum reasonable duration")
                
            except ValueError as e:
                raise ValueError(f"Invalid travel dates format. Use 'YYYY-MM-DD': {str(e)}")
        return v

    @validator('language_preferences', 'cuisine_preferences', 'dietary_preferences', pre=True, always=True)
    def ensure_lists(cls, v) -> List[str]:
        """Ensure these fields are always lists"""
        if not isinstance(v, list):
            raise ValueError("Value must be a list")
        return [str(item).strip() for item in v if item]

    @validator('preferred_transportation')
    def validate_transportation(cls, v: Optional[str]) -> Optional[str]:
        """Validate preferred transportation method"""
        valid_transportation = ['car', 'bus', 'bike', 'walk', 'train', 'plane']
        if v:
            v_lower = v.lower()
            if v_lower not in valid_transportation:
                raise ValueError(
                    f"Invalid transportation method: {v}. "
                    f"Valid options are: {', '.join(valid_transportation)}"
                )
            return v_lower
        return v

    @validator('accommodation_type')
    def validate_accommodation(cls, v: Optional[str]) -> Optional[str]:
        """Validate accommodation type"""
        valid_accommodations = [
            'hotel', 'hostel', 'apartment', 'resort', 
            'guesthouse', 'villa', 'cottage'
        ]
        if v:
            v_lower = v.lower()
            if v_lower not in valid_accommodations:
                raise ValueError(
                    f"Invalid accommodation type: {v}. "
                    f"Valid options are: {', '.join(valid_accommodations)}"
                )
            return v_lower
        return v

    @validator('max_walking_distance')
    def validate_walking_distance(cls, v: Optional[float]) -> Optional[float]:
        """Validate maximum walking distance"""
        if v is not None:
            if v <= 0:
                raise ValueError("Walking distance must be positive")
            if v > 20:  # 20km as reasonable maximum
                raise ValueError("Walking distance exceeds maximum reasonable distance")
        return v

    @validator('group_size')
    def validate_group_size(cls, v: int) -> int:
        """Validate group size"""
        if v <= 0:
            raise ValueError("Group size must be positive")
        if v > 50:  # Reasonable maximum for a tourist group
            raise ValueError("Group size exceeds maximum reasonable size")
        return v

    class Config:
        """Pydantic model configuration"""
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {
            date: lambda v: v.isoformat()
        }
        
        schema_extra = {
            "example": {
                "interests": ["cultural", "nature"],
                "locations": ["willemstad", "punda"],
                "budget_per_day": 100.0,
                "trip_duration": 7,
                "accommodation_type": "hotel",
                "group_size": 2,
                "has_children": False,
                "language_preferences": ["Spanish", "English"],
                "cuisine_preferences": ["local", "seafood"],
                "travel_dates": {
                    "start": "2024-06-01",
                    "end": "2024-06-07"
                }
            }
        }