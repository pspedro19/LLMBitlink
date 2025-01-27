from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from core.recommender.recommendation_engine import RecommendationEngine
from fastapi.responses import HTMLResponse
from core.recommender.formatter import HTMLFormatter
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"API Key configured: {'Yes' if OPENAI_API_KEY else 'No'}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tourism Recommendations API")
engine = RecommendationEngine()
# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Preferences(BaseModel):
    interests: List[str] = Field(..., description="List of travel interests")
    locations: List[str] = Field(..., description="Preferred locations")
    budget_per_day: Optional[float] = Field(None, description="Daily budget in USD")
    trip_duration: int = Field(..., description="Trip duration in days")
    group_size: int = Field(..., description="Number of travelers")
    activity_types: List[str] = Field(..., description="Preferred activity types")
    specific_sites: Optional[List[str]] = Field(None, description="Specific sites to visit")
    cuisine_preferences: Optional[List[str]] = Field(None, description="Food preferences")
    
    class Config:
        schema_extra = {
            "example": {
                "interests": ["cultural", "history", "architecture"],
                "locations": ["Punda", "Otrobanda"],
                "budget_per_day": 200.0,
                "trip_duration": 4,
                "group_size": 2,
                "activity_types": ["walking_tour", "museum_visits"],
                "specific_sites": ["Queen Emma Bridge"],
                "cuisine_preferences": ["local"]
            }
        }

class RecommendationRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    preferences: Preferences

@app.post("/recommendations/")
def get_recommendations(request: RecommendationRequest) -> JSONResponse:
    """Get personalized tourism recommendations"""
    try:
        # Get recommendations
        response = engine.get_recommendations(
            request.query, 
            request.preferences.dict()
        )
        
        if response["status"] == "error":
            raise HTTPException(
                status_code=500,
                detail=response["error"]
            )
            
        if response["status"] == "no_results":
            return JSONResponse(
                status_code=404,
                content={
                    "status": "no_results",
                    "message": response["message"],
                    "metadata": response["metadata"]
                }
            )
        
        # Format recommendations
        formatted_response = {
            "status": "success",
            "recommendations": [
                {
                    "name": rec["name"].upper(),
                    "type": rec["type"],
                    "location": rec["location"],
                    "cost": float(rec.get("cost", 0)),
                    "rating": {
                        "value": float(rec.get("rating", 0)),
                        "display": "★" * int(float(rec.get("rating", 0))) + 
                                 "☆" * (5 - int(float(rec.get("rating", 0))))
                    },
                    "description": rec.get("description", ""),
                    "relevance_score": rec["_scores"]["total"]
                }
                for rec in response["recommendations"]
            ],
            "metadata": {
                **response["metadata"],
                "currency": "USD",
                "timestamp": datetime.now().isoformat()
            },
            "validation": response["validation"]
        }
        
        return JSONResponse(content=formatted_response)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing recommendation request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/recommendations/html")
def get_html_recommendations(request: RecommendationRequest) -> HTMLResponse:
    try:
        if not OPENAI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured"
            )
            
        response = engine.get_recommendations(
            request.query,
            request.preferences.dict()
        )
        
        if response["status"] != "success":
            raise HTTPException(
                status_code=404 if response["status"] == "no_results" else 500,
                detail=response.get("message", "Error generating recommendations")
            )
            
        formatter = HTMLFormatter()
        html_content = formatter.format_to_html(
            response["recommendations"],
            request.preferences.dict()
        )
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"Error processing HTML recommendation request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
async def health_check() -> Dict[str, str]:
    """API health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)