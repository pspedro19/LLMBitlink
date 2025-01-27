from pydantic import BaseModel, Field
from typing import List, Optional

class NLPPreferences(BaseModel):
    interests: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    budget_per_day: Optional[float] = Field(None)
    trip_duration: Optional[int] = Field(None)
    group_size: Optional[int] = Field(1)
    activity_types: List[str] = Field(default_factory=list)
    specific_sites: List[str] = Field(default_factory=list)
    cuisine_preferences: List[str] = Field(default_factory=list)

class NLPRequest(BaseModel):
    text: str = Field(..., description="Texto en lenguaje natural para procesar")

class NLPResponse(BaseModel):
    query: str
    preferences: NLPPreferences