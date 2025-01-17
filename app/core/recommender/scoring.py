from typing import Dict, Any, List
import logging
from collections import Counter
import statistics

logger = logging.getLogger(__name__)

class RecommendationScoring:
    """Enhanced scoring system for tourism recommendations"""
    
    def __init__(self):
        self.scoring_weights = {
            'interest_match': 2.0,
            'location_match': 1.5,
            'budget_match': 1.3,
            'rating_bonus': 1.2,
            'diversity_bonus': 1.1
        }
        
        # Initialize tracking sets for diversity
        self._seen_types = set()
        self._seen_locations = set()
        
    def calculate_recommendation_score(self, item: Dict, preferences: Dict) -> float:
        """Calculate weighted score for a single recommendation"""
        try:
            base_score = 0.0
            
            # Base rating score (0-5 scale)
            rating = float(item.get('rating', 0))
            base_score = rating / 5.0
            
            # Interest matching with enhanced scoring
            if preferences.get('interests'):
                interest_score = self._calculate_interest_score(item, preferences['interests'])
                base_score += interest_score * self.scoring_weights['interest_match']
            
            # Location matching with proximity bonus
            if preferences.get('locations'):
                location_score = self._calculate_location_score(item, preferences['locations'])
                base_score += location_score * self.scoring_weights['location_match']
            
            # Budget matching with range consideration
            if preferences.get('budget_per_day'):
                budget_score = self._calculate_budget_score(item, preferences['budget_per_day'])
                base_score += budget_score * self.scoring_weights['budget_match']
            
            # Rating bonus for highly-rated items
            if rating >= 4.5:
                base_score *= self.scoring_weights['rating_bonus']
            
            # Diversity bonus based on unique attributes
            diversity_score = self._calculate_diversity_score(item)
            if diversity_score > 0:
                base_score *= self.scoring_weights['diversity_bonus']
            
            return base_score
            
        except Exception as e:
            logger.error(f"Error calculating recommendation score: {str(e)}")
            return 0.0
    
    def _calculate_interest_score(self, item: Dict, interests: List[str]) -> float:
        """Calculate interest match score"""
        try:
            item_text = (f"{item.get('type', '')} {item.get('description', '')} "
                        f"{item.get('recommended_for', '')}").lower()
            
            matches = sum(1.0 for interest in interests
                         if interest.lower() in item_text)
            return min(matches / len(interests), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating interest score: {str(e)}")
            return 0.0
    
    def _calculate_location_score(self, item: Dict, locations: List[str]) -> float:
        """Calculate location match score with proximity consideration"""
        try:
            item_location = item.get('location', '').lower()
            max_score = 0.0
            
            for location in locations:
                location = location.lower()
                if location == item_location:
                    max_score = max(max_score, 1.0)
                elif location in item_location or item_location in location:
                    max_score = max(max_score, 0.8)
                    
            return max_score
            
        except Exception as e:
            logger.error(f"Error calculating location score: {str(e)}")
            return 0.0
    
    def _calculate_budget_score(self, item: Dict, budget: float) -> float:
        """Calculate budget match score"""
        try:
            item_cost = float(item.get('cost', 0)) or float(item.get('price', 0))
            
            if item_cost <= budget:
                score = 1.0
                # Bonus for being well within budget
                if item_cost <= budget * 0.7:
                    score *= 1.2
                return min(score, 1.0)
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating budget score: {str(e)}")
            return 0.0
    
    def _calculate_diversity_score(self, item: Dict) -> float:
        """Calculate diversity score based on unique attributes"""
        try:
            score = 0.0
            item_type = str(item.get('type', '')).lower()
            item_location = str(item.get('location', '')).lower()
            
            if item_type not in self._seen_types:
                score += 0.5
                self._seen_types.add(item_type)
                
            if item_location not in self._seen_locations:
                score += 0.5
                self._seen_locations.add(item_location)
                
            return score
            
        except Exception as e:
            logger.error(f"Error calculating diversity score: {str(e)}")
            return 0.0
    
    def calculate_distribution_metrics(self, recommendations: List[Dict]) -> Dict[str, Any]:
        """Calculate distribution metrics for recommendations"""
        try:
            return {
                'type_distribution': self._get_type_distribution(recommendations),
                'price_distribution': self._get_price_distribution(recommendations),
                'location_distribution': self._get_location_distribution(recommendations),
                'rating_distribution': self._get_rating_distribution(recommendations)
            }
        except Exception as e:
            logger.error(f"Error calculating distribution metrics: {str(e)}")
            return {}
    
    def _get_type_distribution(self, recommendations: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of recommendation types"""
        return Counter(rec.get('type', 'unknown').lower() for rec in recommendations)
    
    def _get_price_distribution(self, recommendations: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of price ranges"""
        return Counter(self._get_price_category(rec) for rec in recommendations)
    
    def _get_location_distribution(self, recommendations: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of locations"""
        return Counter(rec.get('location', 'unknown').lower() for rec in recommendations)
    
    def _get_rating_distribution(self, recommendations: List[Dict]) -> Dict[str, float]:
        """Calculate rating statistics"""
        ratings = [float(rec.get('rating', 0)) for rec in recommendations if rec.get('rating')]
        if not ratings:
            return {'average': 0, 'median': 0, 'max': 0, 'min': 0}
            
        return {
            'average': statistics.mean(ratings),
            'median': statistics.median(ratings),
            'max': max(ratings),
            'min': min(ratings)
        }
    
    def _get_price_category(self, recommendation: Dict) -> str:
        """Determine price category for a recommendation"""
        try:
            cost = float(recommendation.get('cost', 0))
            if cost <= 50:
                return 'budget'
            elif cost <= 150:
                return 'moderate'
            return 'premium'
        except (ValueError, TypeError):
            return 'unknown'