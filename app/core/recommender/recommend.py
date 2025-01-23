from typing import Dict, Any, List
from .recommendation_engine import IntegratedTourismSystem
from utils.logger import get_logger
from datetime import datetime

logger = get_logger(__name__)

def recommend(query: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
    """
    High-level function to get recommendations based on query and preferences
    
    Args:
        query (str): User's query text
        preferences (Dict[str, Any]): User preferences
        
    Returns:
        Dict[str, Any]: Complete response including recommendations and metadata
    """
    try:
        logger.info(f"Processing recommendation request for query: {query[:100]}...")
        logger.debug(f"Preferences: {preferences}")

        # Initialize the tourism system
        tourism_system = IntegratedTourismSystem()
        
        # Process the query and get complete response
        response = tourism_system.process_query(query, preferences)
        
        # Validate recommendations format
        recommendations = response.get("recommendations", [])
        valid_recommendations = []
        
        for rec in recommendations:
            if isinstance(rec, dict):
                valid_recommendations.append(rec)
            else:
                logger.warning(f"Invalid recommendation format: {rec}")
        
        # Update the recommendations in the response with validated ones
        response["recommendations"] = valid_recommendations
        
        # Log results
        if not valid_recommendations:
            logger.warning("No valid recommendations found in response")
        else:
            logger.info(f"Found {len(valid_recommendations)} valid recommendations")
            logger.debug(f"First recommendation: {valid_recommendations[0] if valid_recommendations else 'None'}")

        # Ensure all required fields are present
        response.setdefault("validation", {})
        response.setdefault("query_analysis", {})
        response.setdefault("metadata", {
            "query_time": datetime.now().isoformat(),
            "recommendation_count": len(valid_recommendations),
            "preference_count": len(preferences)
        })
        response.setdefault("status", "success" if valid_recommendations else "no_results")
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}", exc_info=True)
        # Return a properly structured error response
        return {
            "status": "error",
            "error": str(e),
            "recommendations": [],
            "validation": {},
            "query_analysis": {},
            "metadata": {
                "query_time": datetime.now().isoformat(),
                "error_message": str(e),
                "preference_count": len(preferences),
                "recommendation_count": 0
            }
        }