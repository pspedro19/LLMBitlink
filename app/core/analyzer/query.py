from typing import Dict, Any, Optional
from datetime import datetime
import time

from .nlp_processor import ImprovedNLPProcessor
from ..recommender.recommendation_engine import IntegratedTourismSystem
from app.utils.config import API_CONFIG
from app.utils.logger import get_logger

# Solo necesitamos una instancia del logger
logger = get_logger(__name__)

class QueryProcessor:
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize query processor with NLP and recommendation components"""
        try:
            self.nlp_processor = ImprovedNLPProcessor()
            self.tourism_system = IntegratedTourismSystem(openai_api_key)
        except Exception as e:
            logger.error(f"Failed to initialize QueryProcessor: {e}")
            raise

    def process_query(
        self, 
        query: str, 
        preferences: Optional[Dict[str, Any]] = None,
        max_recommendations: int = API_CONFIG["max_recommendations"]
    ) -> Dict[str, Any]:
        """Process user query with enhanced error handling and metrics"""
        start_time = time.time()
        processing_times = {}

        try:
            # Validate input
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Invalid query format")

            # Extract and merge preferences with timing
            stage_start = time.time()
            extracted_prefs = self.nlp_processor.extract_preferences(query)
            processing_times['preference_extraction'] = time.time() - stage_start

            merged_preferences = self._merge_preferences(
                extracted_prefs, 
                preferences or {}
            )

            # Get recommendations with limit
            recommendations = self.tourism_system.get_recommendations(
                merged_preferences,
                limit=max_recommendations
            )

            # Get intent analysis
            intent_analysis = self.nlp_processor.classify_intent(query)

            return self._compile_response(
                query=query,
                recommendations=recommendations,
                preferences=merged_preferences,
                intent_analysis=intent_analysis,
                processing_times=processing_times,
                total_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return self._create_error_response(str(e), time.time() - start_time)

    def _merge_preferences(self, extracted: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge extracted preferences with existing preferences
        
        Args:
            extracted (Dict[str, Any]): Newly extracted preferences
            existing (Dict[str, Any]): Existing preferences
            
        Returns:
            Dict[str, Any]: Merged preferences
        """
        merged = existing.copy()
        
        # Merge list fields
        list_fields = ['interests', 'locations', 'preferred_activities']
        for field in list_fields:
            if field in extracted:
                existing_items = set(merged.get(field, []))
                new_items = set(extracted[field])
                merged[field] = list(existing_items.union(new_items))

        # Update scalar values if new ones are extracted
        scalar_fields = ['budget_per_day', 'trip_duration']
        for field in scalar_fields:
            if field in extracted and extracted[field] is not None:
                merged[field] = extracted[field]

        return merged

    def _compile_response(
        self,
        query: str,
        recommendations: list,
        preferences: dict,
        intent_analysis: dict,
        processing_times: dict,
        total_time: float
    ) -> Dict[str, Any]:
        """
        Compile the final response with all metrics and data
        
        Args:
            query (str): Original query text
            recommendations (list): List of recommendations
            preferences (dict): Processed preferences
            intent_analysis (dict): Intent analysis results
            processing_times (dict): Processing time metrics
            total_time (float): Total processing time
            
        Returns:
            Dict[str, Any]: Complete response with all data and metrics
        """
        return {
            "status": "success",
            "recommendations": recommendations,
            "query_analysis": {
                "intent_scores": intent_analysis,
                "extracted_preferences": preferences,
                "query_text": query
            },
            "metadata": {
                "query_time": datetime.now().isoformat(),
                "processing_times": processing_times,
                "total_processing_time": total_time
            }
        }

    def _create_error_response(self, error_message: str, total_time: float) -> Dict[str, Any]:
        """
        Create an error response
        
        Args:
            error_message (str): Description of the error
            total_time (float): Total processing time
            
        Returns:
            Dict[str, Any]: Error response with metadata
        """
        return {
            "status": "error",
            "error_message": error_message,
            "metadata": {
                "error_time": datetime.now().isoformat(),
                "total_processing_time": total_time
            }
        }