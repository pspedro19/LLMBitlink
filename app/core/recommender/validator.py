import re
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

class RecommendationValidator:
    """Enhanced validator with improved metrics handling and validation"""

    def __init__(self):
        """Initialize validator with metrics and price categories"""
        self.validation_metrics = {
            'location_match': 0.0,
            'budget_match': 0.0,
            'interest_match': 0.0,
            'diversity_score': 0.0,
            'preference_coverage': 0.0
        }

        # Define price categories for normalization
        self.price_categories = {
            'low': 50.0,
            'medium': 150.0,
            'high': 300.0,
            'budget': 50.0,
            'moderate': 150.0,
            'premium': 300.0
        }

    def validate(self, recommendations: List[Dict], preferences: Dict[str, Any]) -> Dict[str, float]:
        """
        Validate recommendations with enhanced metrics
        
        Args:
            recommendations: List of recommendation dictionaries
            preferences: Dictionary of user preferences
            
        Returns:
            Dict[str, float]: Dictionary of validation metrics
        """
        try:
            if not recommendations:
                return self.validation_metrics

            # Remove duplicates before validation
            unique_recommendations = self._deduplicate_recommendations(recommendations)

            # Calculate all validation metrics
            self._validate_locations(unique_recommendations, preferences)
            self._validate_budget(unique_recommendations, preferences)
            self._validate_interests(unique_recommendations, preferences)
            self._calculate_diversity(unique_recommendations)
            self._calculate_coverage(unique_recommendations, preferences)

            return self.validation_metrics

        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return self.validation_metrics

    def _deduplicate_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Remove duplicate recommendations based on multiple fields"""
        unique_recs = {}
        for rec in recommendations:
            # Create a unique key using multiple fields
            key = f"{rec.get('name', '')}-{rec.get('location', '')}-{rec.get('type', '')}"
            if key not in unique_recs:
                unique_recs[key] = rec
        return list(unique_recs.values())

    def _validate_locations(self, recommendations: List[Dict], preferences: Dict[str, Any]) -> None:
        """Enhanced location validation with fuzzy matching"""
        try:
            if not preferences.get('locations'):
                self.validation_metrics['location_match'] = 0.0
                return

            locations = [loc.lower() for loc in preferences['locations']]
            total_score = 0.0

            for rec in recommendations:
                rec_location = rec.get('location', '').lower()
                location_score = 0.0

                # Exact match
                if any(loc == rec_location for loc in locations):
                    location_score = 1.0
                # Partial match
                elif any(loc in rec_location or rec_location in loc for loc in locations):
                    location_score = 0.8
                # Nearby location
                elif any(self._are_locations_nearby(loc, rec_location) for loc in locations):
                    location_score = 0.5

                total_score += location_score

            self.validation_metrics['location_match'] = total_score / len(recommendations) if recommendations else 0.0

        except Exception as e:
            logger.error(f"Location validation error: {str(e)}")
            self.validation_metrics['location_match'] = 0.0

    def _validate_budget(self, recommendations: List[Dict], preferences: Dict[str, Any]) -> None:
        """Enhanced budget validation with comprehensive cost handling"""
        try:
            if not preferences.get('budget_per_day'):
                self.validation_metrics['budget_match'] = 0.0
                return

            budget = float(preferences['budget_per_day'])
            matches = 0
            total_items = len(recommendations)

            for rec in recommendations:
                # Get all possible cost fields
                costs = [
                    self._normalize_price(rec.get('cost', 0)),
                    self._normalize_price(rec.get('price', 0)),
                    self._normalize_price(rec.get('entry_fee', 0)),
                    self._normalize_price(rec.get('price_range', 0))
                ]

                # Get price category if available
                price_category = str(rec.get('price_range', '')).lower().strip()
                if price_category in ['low', 'budget']:
                    costs.append(50.0)
                elif price_category in ['medium', 'moderate']:
                    costs.append(150.0)
                elif price_category in ['high', 'premium']:
                    costs.append(300.0)

                # Use minimum non-zero cost or default to 0
                valid_costs = [c for c in costs if c > 0]
                item_cost = min(valid_costs) if valid_costs else 0

                # Enhanced budget matching with flexible tolerance
                if item_cost <= budget:
                    matches += 1.0  # Full match
                elif item_cost <= budget * 1.1:  # 10% tolerance
                    matches += 0.8
                elif item_cost <= budget * 1.2:  # 20% tolerance
                    matches += 0.6
                elif item_cost <= budget * 1.3:  # 30% tolerance
                    matches += 0.4
                elif item_cost <= budget * 1.5:  # 50% tolerance
                    matches += 0.2

                # Bonus for significantly under budget
                if item_cost <= budget * 0.7:
                    matches += 0.2

                # Add weighted score for free activities
                if item_cost == 0 and any(tag in str(rec).lower() for tag in ['free', 'no cost', 'public']):
                    matches += 1.0

            # Calculate final score with normalization
            final_score = matches / total_items if total_items > 0 else 0.0
            self.validation_metrics['budget_match'] = min(final_score, 1.0)

        except Exception as e:
            logger.error(f"Budget validation error: {str(e)}", exc_info=True)
            self.validation_metrics['budget_match'] = 0.0

    def _validate_interests(self, recommendations: List[Dict], preferences: Dict[str, Any]) -> None:
        """Enhanced interest validation with semantic matching"""
        try:
            if not preferences.get('interests'):
                self.validation_metrics['interest_match'] = 0.0
                return

            interests = [i.lower() for i in preferences['interests']]

            # Define interest categories with semantic matching
            interest_categories = {
                'cultural': {
                    'keywords': ['museum', 'heritage', 'history', 'art', 'architecture', 'colonial', 'historical'],
                    'weight': 1.3
                },
                'nature': {
                    'keywords': ['park', 'hiking', 'wildlife', 'botanical', 'landscape', 'trail', 'mountain'],
                    'weight': 1.2
                },
                'water_activities': {
                    'keywords': ['beach', 'diving', 'snorkeling', 'swimming', 'boat', 'marine', 'sea'],
                    'weight': 1.2
                },
                'adventure': {
                    'keywords': ['hiking', 'climbing', 'sports', 'kayaking', 'biking', 'adventure', 'explore'],
                    'weight': 1.3
                },
                'food': {
                    'keywords': ['restaurant', 'dining', 'cuisine', 'culinary', 'gastronomy', 'food', 'taste'],
                    'weight': 1.2
                }
            }

            total_score = 0.0
            for rec in recommendations:
                rec_text = f"{rec.get('type', '')} {rec.get('description', '')} {rec.get('recommended_for', '')}".lower()
                score = 0.0

                # Direct interest matching
                for interest in interests:
                    # Direct match
                    if interest in rec_text:
                        score += 1.0

                    # Category matching with weights
                    for category, info in interest_categories.items():
                        if (interest in category or
                            any(kw in interest for kw in info['keywords']) or
                            any(kw in rec_text for kw in info['keywords'])):
                            category_score = 0.8 * info['weight']
                            
                            # Context bonuses
                            if 'guided' in rec_text or 'tour' in rec_text:
                                category_score *= 1.1
                            if 'private' in rec_text:
                                category_score *= 1.1
                            if 'authentic' in rec_text or 'traditional' in rec_text:
                                category_score *= 1.2

                            score += category_score

                # Rating bonus
                try:
                    rating = float(rec.get('rating', 0))
                    if rating >= 4.5:
                        score *= 1.2
                    elif rating >= 4.0:
                        score *= 1.1
                except (ValueError, TypeError):
                    pass

                # Normalize and add to total
                score = min(score, 1.0)
                total_score += score

            # Calculate final score
            final_score = total_score / len(recommendations) if recommendations else 0.0
            self.validation_metrics['interest_match'] = min(final_score, 1.0)

        except Exception as e:
            logger.error(f"Interest validation error: {str(e)}", exc_info=True)
            self.validation_metrics['interest_match'] = 0.0

    def _calculate_coverage(self, recommendations: List[Dict], preferences: Dict[str, Any]) -> None:
        """Calculate comprehensive preference coverage"""
        try:
            all_prefs = set()
            for pref_type in ['interests', 'locations', 'activity_types', 'cuisine_preferences']:
                all_prefs.update(preferences.get(pref_type, []))

            if not all_prefs:
                self.validation_metrics['preference_coverage'] = 0.0
                return

            pref_scores = {}
            for pref in all_prefs:
                pref_lower = pref.lower()
                max_score = 0.0

                for rec in recommendations:
                    # Check all relevant fields
                    rec_text = (f"{rec.get('type', '')} {rec.get('description', '')} "
                              f"{rec.get('location', '')} {rec.get('recommended_for', '')} "
                              f"{rec.get('cuisine_type', '')}").lower()

                    if pref_lower in rec_text:
                        score = 1.0
                    elif any(word in rec_text for word in pref_lower.split()):
                        score = 0.5
                    else:
                        continue

                    max_score = max(max_score, score)

                if max_score > 0:
                    pref_scores[pref] = max_score

            total_score = sum(pref_scores.values())
            self.validation_metrics['preference_coverage'] = total_score / len(all_prefs)

        except Exception as e:
            logger.error(f"Coverage calculation error: {str(e)}")
            self.validation_metrics['preference_coverage'] = 0.0

    def _calculate_diversity(self, recommendations: List[Dict]) -> None:
        """Calculate diversity across multiple dimensions"""
        try:
            if not recommendations:
                self.validation_metrics['diversity_score'] = 0.0
                return

            # Calculate diversity across dimensions
            type_diversity = len(set(rec.get('type', '').lower() for rec in recommendations))
            location_diversity = len(set(rec.get('location', '').lower() for rec in recommendations))
            price_diversity = len(set(self._get_price_category(rec) for rec in recommendations))

            # Weighted average of diversity scores
            weights = {'type': 0.4, 'location': 0.4, 'price': 0.2}

            type_score = type_diversity / len(recommendations)
            location_score = location_diversity / len(recommendations)
            price_score = price_diversity / min(len(recommendations), 3)

            diversity_score = (
                type_score * weights['type'] +
                location_score * weights['location'] +
                price_score * weights['price']
            )

            self.validation_metrics['diversity_score'] = min(diversity_score, 1.0)

        except Exception as e:
            logger.error(f"Diversity calculation error: {str(e)}")
            self.validation_metrics['diversity_score'] = 0.0

    def _normalize_price(self, price_value: Any) -> float:
        """Normalize price values from different formats"""
        if not price_value:
            return 0.0

        try:
            if isinstance(price_value, (int, float)):
                return float(price_value)

            if isinstance(price_value, str):
                price_value = price_value.lower().strip()

                # Handle categorical prices
                if price_value in self.price_categories:
                    return self.price_categories[price_value]

                # Extract numeric value
                numeric_value = re.findall(r'\d+(?:\.\d+)?', price_value)
                if numeric_value:
                    return float(numeric_value[0])

            return 0.0

        except Exception:
            return 0.0

    def _get_price_category(self, recommendation: Dict) -> str:
        """Determine price category for a recommendation"""
        try:
            cost = self._normalize_price(recommendation.get('cost', 0))
            if cost <= 50:
                return 'budget'
            elif cost <= 150:
                return 'moderate'
            return 'premium'
        except Exception:
            return 'unknown'

    def _are_locations_nearby(self, loc1: str, loc2: str) -> bool:
        """Determine if two locations are nearby"""
        nearby_locations = {
            'willemstad': ['punda', 'otrobanda', 'pietermaai'],
            'punda': ['otrobanda', 'pietermaai', 'willemstad'],
            'otrobanda': ['punda', 'willemstad'],
            'westpunt': ['christoffel park', 'shete boka'],
            'jan thiel': ['caracasbaai', 'spanish water']
        }

        loc1 = loc1.lower()
        loc2 = loc2.lower()

        return (loc1 in nearby_locations and loc2 in nearby_locations[loc1]) or \
               (loc2 in nearby_locations and loc1 in nearby_locations[loc2])