from typing import Dict, List, Any, Optional
from datetime import datetime
import statistics
from collections import defaultdict
import logging

from core.analyzer.nlp_processor import ImprovedNLPProcessor
from core.recommender.validator import RecommendationValidator
from core.data.database import CSVDatabaseManager
from core.utils.validators import DataValidator

logger = logging.getLogger(__name__)

class RecommendationEngine:
    def __init__(self):
        self.nlp_processor = ImprovedNLPProcessor()
        self.validator = RecommendationValidator()
        self.db_manager = CSVDatabaseManager()
        
        # Scoring weights
        self.weights = {
            'interest_match': 0.35,
            'location_match': 0.25,
            'budget_match': 0.20,
            'rating': 0.10,
            'diversity': 0.10
        }
        
        # Category requirements
        self.category_balance = {
            'cultural': {'min': 2, 'max': 4},
            'nature': {'min': 1, 'max': 3},
            'food': {'min': 1, 'max': 2},
            'activity': {'min': 1, 'max': 3}
        }

    def get_recommendations(self, query: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Obtener recomendaciones personalizadas de turismo
        
        Args:
            query (str): Consulta en lenguaje natural
            preferences (Dict[str, Any]): Preferencias explícitas del usuario
            
        Returns:
            Dict[str, Any]: Recomendaciones y metadata
        """
        try:
            start_time = datetime.now()
            
            # Procesar consulta y preferencias
            extracted_prefs = self.nlp_processor.extract_preferences(query)
            merged_prefs = self._merge_preferences(extracted_prefs, preferences)
            
            # Obtener recomendaciones iniciales
            recommendations = self._fetch_recommendations(merged_prefs)
            if not recommendations:
                return self._create_empty_response("No recommendations found")
            
            # Puntuar y ordenar recomendaciones 
            scored_recs = self._score_recommendations(recommendations, merged_prefs)
            
            # Balancear y diversificar resultados
            final_recs = self._balance_recommendations(scored_recs, merged_prefs)
            
            # Validar resultados
            validation_results = self.validator.validate(final_recs, merged_prefs)
            
            return {
                "status": "success",
                "recommendations": final_recs,
                "metadata": {
                    "query_time": datetime.now().isoformat(),
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "total_results": len(final_recs),
                    "query_understanding": self._calculate_query_understanding(query, merged_prefs)
                },
                "validation": validation_results
            }
            
        except Exception as e:
            logger.error(f"Error in recommendation engine: {str(e)}")
            return self._create_error_response(str(e))

    def _get_relevant_categories(self, preferences: Dict[str, Any]) -> List[str]:
        """Determinar las categorías relevantes basadas en las preferencias del usuario"""
        categories = set()
        
        # Mapear intereses a categorías
        interest_to_category = {
            'cultural': ['realistic_curacao_tourist_spots', 'realistic_curacao_activities'],
            'history': ['realistic_curacao_tourist_spots'],
            'food': ['realistic_curacao_restaurants'],
            'nightlife': ['realistic_curacao_nightclubs'],
            'music': ['realistic_curacao_nightclubs'],
            'adventure': ['realistic_curacao_activities'],
            'nature': ['realistic_curacao_activities']
        }
        
        # Agregar categorías basadas en intereses
        if interests := preferences.get('interests', []):
            for interest in interests:
                for category in interest_to_category.get(interest, []):
                    categories.add(category)
        
        # Agregar categorías basadas en tipos de actividad
        activity_to_category = {
            'walking_tour': ['realistic_curacao_activities'],
            'museum_visits': ['realistic_curacao_tourist_spots'],
            'food_tasting': ['realistic_curacao_restaurants'],
            'dancing': ['realistic_curacao_nightclubs'],
            'live_music': ['realistic_curacao_nightclubs'] 
        }
        
        if activities := preferences.get('activity_types', []):
            for activity in activities:
                for category in activity_to_category.get(activity, []):
                    categories.add(category)
        
        # Si no hay categorías específicas, incluir todas
        if not categories:
            categories = {
                'realistic_curacao_tourist_spots',
                'realistic_curacao_activities',
                'realistic_curacao_restaurants',
                'realistic_curacao_nightclubs'
            }
        
        return list(categories)
    

    def _build_query(self, category: str, preferences: Dict[str, Any]) -> str:
        """
        Construir consulta SQL basada en categoría y preferencias para consultar dataframes
        
        Args:
            category (str): Categoría a consultar 
            preferences (Dict[str, Any]): Preferencias del usuario
            
        Returns:
            str: Consulta SQL construida
        """
        conditions = []
        
        # Filtrado por ubicación usando LOWER para case-insensitive
        if locations := preferences.get('locations'):
            location_clauses = []
            for loc in locations:
                location_clauses.append(f"location LIKE '%{loc.lower()}%'") 
            if location_clauses:
                conditions.append("(" + " OR ".join(location_clauses) + ")")
        
        # Filtrado por presupuesto con campo dinámico
        if budget := preferences.get('budget_per_day'):
            cost_field = {
                'realistic_curacao_activities': 'cost',
                'realistic_curacao_tourist_spots': 'entry_fee',
                'realistic_curacao_restaurants': 'average_person_expense',
                'realistic_curacao_nightclubs': 'average_person_expense'
            }.get(category)
            if cost_field:
                conditions.append(f"{cost_field} <= {float(budget)}")
        
        # Construir cláusula WHERE
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # Construir consulta completa
        query = f"""
            SELECT *
            FROM {category}
            WHERE {where_clause}
            ORDER BY rating DESC
            LIMIT 10
        """
        
        return query.strip()

    def _fetch_recommendations(self, preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch recommendations from database"""
        try:
            recommendations = []
            
            # Query each relevant table based on preferences
            for category in self._get_relevant_categories(preferences):
                query = self._build_query(category, preferences)
                results = self.db_manager.execute_query(query)
                recommendations.extend(results)
                
                # Validar resultados obtenidos
                for rec in results:
                    if 'type' not in rec or not rec['type']:
                        rec['type'] = 'unknown'
                
                recommendations.extend(results)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error fetching recommendations: {str(e)}")
            return []

    def _score_recommendations(self, recommendations: List[Dict], preferences: Dict) -> List[Dict]:
        """Score recommendations based on preferences"""
        scored_recs = []
        
        for rec in recommendations:
            score = 0.0
            
            # Interest match
            interest_score = self._calculate_interest_match(rec, preferences)
            score += interest_score * self.weights['interest_match']
            
            # Location match
            location_score = self._calculate_location_match(rec, preferences)
            score += location_score * self.weights['location_match']
            
            # Budget match
            budget_score = self._calculate_budget_match(rec, preferences)
            score += budget_score * self.weights['budget_match']
            
            # Rating contribution
            rating_score = float(rec.get('rating', 0)) / 5.0
            score += rating_score * self.weights['rating']
            
            # Store scores for transparency
            rec['_scores'] = {
                'interest': interest_score,
                'location': location_score,
                'budget': budget_score,
                'rating': rating_score,
                'total': score
            }
            
            scored_recs.append(rec)
            
        return sorted(scored_recs, key=lambda x: x['_scores']['total'], reverse=True)

    def _balance_recommendations(self, recommendations: List[Dict], preferences: Dict) -> List[Dict]:
        """Balance and diversify recommendations"""
        balanced_recs = []
        category_counts = defaultdict(int)
        
        # Sort by score but respect category balance
        for rec in sorted(recommendations, key=lambda x: x['_scores']['total'], reverse=True):
            category = self._get_category(rec)
            
            if (category_counts[category] < self.category_balance.get(category, {}).get('max', float('inf')) and
                len(balanced_recs) < preferences.get('limit', 10)):
                
                # Add diversity bonus for underrepresented categories
                if category_counts[category] < self.category_balance.get(category, {}).get('min', 0):
                    rec['_scores']['total'] *= 1.2
                
                balanced_recs.append(rec)
                category_counts[category] += 1
        
        return balanced_recs

    def _calculate_interest_match(self, recommendation: Dict, preferences: Dict) -> float:
        """Calculate how well a recommendation matches user interests"""
        if not preferences.get('interests'):
            return 0.0
            
        rec_text = f"{recommendation.get('type', '')} {recommendation.get('description', '')}".lower()
        matches = sum(1 for interest in preferences['interests'] 
                     if interest.lower() in rec_text)
        
        return matches / len(preferences['interests'])

    def _calculate_location_match(self, recommendation: Dict, preferences: Dict) -> float:
        """Calculate location match score"""
        if not preferences.get('locations'):
            return 0.0
            
        rec_location = recommendation.get('location', '').lower()
        for location in preferences['locations']:
            if location.lower() == rec_location:
                return 1.0
            elif location.lower() in rec_location or rec_location in location.lower():
                return 0.8
                
        return 0.0

    def _calculate_budget_match(self, recommendation: Dict, preferences: Dict) -> float:
        """Calculate budget match score"""
        if not preferences.get('budget_per_day'):
            return 0.0
            
        try:
            budget = float(preferences['budget_per_day'])
            cost = float(recommendation.get('cost', 0))
            
            if cost <= budget:
                return 1.0
            elif cost <= budget * 1.2:
                return 0.7
            elif cost <= budget * 1.5:
                return 0.3
                
            return 0.0
            
        except (ValueError, TypeError):
            return 0.0

    def _merge_preferences(self, extracted: Dict, explicit: Dict) -> Dict:
        """Merge and validate extracted and explicit preferences"""
        merged = explicit.copy()
        
        # Add extracted preferences if not explicitly specified
        for key in ['interests', 'locations', 'budget_per_day']:
            if key not in merged and key in extracted:
                merged[key] = extracted[key]
                
        # Validate merged preferences
        return DataValidator.validate_preferences(merged)

    def _create_empty_response(self, message: str) -> Dict:
        """Create response for no recommendations"""
        return {
            "status": "no_results",
            "message": message,
            "recommendations": [],
            "metadata": {
                "query_time": datetime.now().isoformat(),
                "total_results": 0
            }
        }

    def _create_error_response(self, error: str) -> Dict:
        """Create error response"""
        return {
            "status": "error",
            "error": error,
            "recommendations": [],
            "metadata": {
                "query_time": datetime.now().isoformat(),
                "error_type": "recommendation_error"
            }
        }

    def _get_category(self, recommendation: Dict) -> str:
        """
        Determine category based on establishment type and details
        """
        try:
            rec_type = recommendation.get('type', '').lower()
            
            # Nightlife categories based on music_type
            nightlife_types = {
                # Dance Music
                'salsa', 'bachata', 'merengue', 'reggaeton', 'latin', 'cumbia',
                'electronic', 'techno', 'house', 'edm', 'dance', 'disco', 'trance',
                
                # Live Music
                'rock', 'jazz', 'blues', 'live music', 'live band', 'acoustic',
                'pop', 'r&b', 'soul', 'funk', 'hip hop', 'rap',
                
                # Caribbean/Local
                'zouk', 'soca', 'calypso', 'caribbean', 'steel pan', 'tumba',
                
                # Venue Types  
                'lounge', 'club', 'nightclub', 'bar', 'pub', 'dance club',
                'cocktail bar', 'music bar', 'karaoke', 'disco bar',
                
                # Event Types
                'dj', 'live performance', 'concert', 'dance party', 'social dancing',
                'dance floor', 'live entertainment', 'music show', 'party'
            }

            # Restaurant categories based on cuisine_type 
            food_types = {
                # Local & Regional
                'local', 'krioyo', 'caribbean', 'creole', 'antillean', 'curaçaoan',
                'dutch', 'surinamese', 'venezuelan', 'island cuisine',
                
                # Seafood & Fish
                'seafood', 'fish', 'lobster', 'crab', 'shellfish', 'fresh catch',
                'ocean grill', 'sea food', 'maritime cuisine', 'fisherman\'s',
                
                # International
                'international', 'fusion', 'mediterranean', 'european', 'american',
                'asian', 'chinese', 'japanese', 'thai', 'indian', 'indonesian',
                'italian', 'french', 'spanish', 'greek', 'mexican', 'latin american',
                
                # Dietary Specific
                'vegan', 'vegetarian', 'plant-based', 'gluten-free', 'dairy-free',
                'halal', 'kosher', 'organic', 'health food', 'raw food',
                
                # Restaurant Types
                'fine dining', 'casual dining', 'bistro', 'cafe', 'steakhouse',
                'grill', 'buffet', 'food court', 'beach bar', 'tapas', 'wine bar',
                'gastro pub', 'trattoria', 'pizzeria', 'sushi bar', 'noodle house',
                
                # Meal Types
                'breakfast', 'brunch', 'lunch', 'dinner', 'all-day dining',
                'street food', 'snack bar', 'quick bites', 'takeaway', 'delivery',
                
                # Specialties
                'barbecue', 'grill', 'rotisserie', 'smokehouse', 'raw bar',
                'seafood market', 'farm to table', 'home cooking', 'traditional',
                'contemporary', 'modern', 'innovative', 'fusion'
            }

            # Activity and attraction types
            cultural_types = {
                # Museums & Galleries
                'museum', 'art gallery', 'exhibition hall', 'heritage museum',
                'maritime museum', 'history museum', 'science museum', 'art studio',
                
                # Historic Sites
                'historic', 'heritage', 'monument', 'historic site', 'landmark',
                'colonial building', 'fort', 'castle', 'palace', 'plantation house',
                'historic district', 'old town', 'heritage site', 'unesco site',
                
                # Cultural Venues
                'theater', 'cultural center', 'concert hall', 'opera house',
                'performing arts center', 'amphitheater', 'library', 'archive',
                
                # Religious Sites
                'church', 'synagogue', 'temple', 'cathedral', 'chapel',
                'religious site', 'sacred place', 'monastery', 'shrine',
                
                # Cultural Activities
                'cultural tour', 'guided tour', 'heritage walk', 'historical tour',
                'art workshop', 'cultural workshop', 'cultural festival',
                'traditional ceremony', 'cultural demonstration', 'artisan market',
                
                # Architecture
                'architecture', 'colonial architecture', 'dutch architecture',
                'historic architecture', 'architectural site', 'traditional building',
                
                # Education & Research
                'research center', 'educational center', 'cultural institute',
                'heritage center', 'interpretation center', 'visitor center'
            }

            nature_types = {
                # Beaches & Coastal
                'beach', 'cove', 'bay', 'lagoon', 'coast', 'shoreline', 'reef',
                'marine park', 'seaside', 'oceanfront', 'beach park', 'tide pools',
                
                # Parks & Reserves
                'park', 'nature reserve', 'national park', 'marine reserve', 
                'protected area', 'conservation area', 'wildlife sanctuary',
                'botanical garden', 'eco park', 'nature park', 'forest reserve',
                
                # Natural Features
                'cave', 'cliff', 'rock formation', 'coral reef', 'mangrove',
                'wetland', 'salt flat', 'lagoon', 'natural pool', 'sinkhole',
                'natural bridge', 'limestone formation',
                
                # Activities & Trails
                'hiking', 'hiking trail', 'nature trail', 'walking path',
                'bird watching', 'wildlife viewing', 'nature walk', 'eco tour',
                'trekking', 'outdoor adventure', 'nature exploration',
                
                # Eco Tourism
                'eco tourism', 'eco friendly', 'sustainable tourism',
                'environmental education', 'nature education', 'conservation site',
                'eco lodge', 'eco resort', 'nature center', 'visitor center',
                
                # Outdoor Recreation
                'outdoor', 'nature activity', 'outdoor recreation',
                'nature photography', 'camping', 'picnic area', 'viewpoint',
                'observation point', 'scenic spot', 'lookout point'
            }

            activity_types = {
                # Tours & Guided Experiences
                'tour', 'guided tour', 'walking tour', 'sightseeing', 'city tour',
                'excursion', 'day trip', 'guided visit', 'island tour', 'boat tour',
                'food tour', 'cultural tour', 'photography tour', 'sunset cruise',
                'private tour', 'group tour',
                
                # Water Activities  
                'diving', 'snorkeling', 'swimming', 'kayaking', 'paddleboarding',
                'sailing', 'fishing', 'boat ride', 'jet skiing', 'windsurfing',
                'kite surfing', 'surfing', 'water sports',
                
                # Educational & Cultural
                'workshop', 'class', 'cooking class', 'art class', 'dance lesson',
                'craft workshop', 'tasting', 'seminar', 'demonstration', 
                'cultural workshop', 'language lesson',
                
                # Adventure & Sport
                'hiking', 'biking', 'cycling', 'climbing', 'horseback riding',
                'atv tour', 'jeep safari', 'zip lining', 'paragliding',
                'cliff jumping', 'cave exploration',
                
                # Entertainment & Shows
                'performance', 'show', 'concert', 'festival', 'carnival',
                'live entertainment', 'dance show', 'music performance',
                'cultural show', 'dinner show',
                
                # Wellness & Recreation
                'yoga', 'spa treatment', 'massage', 'meditation', 'fitness class',
                'beach activity', 'pool access', 'golf', 'tennis'
            }
            
            logger.debug("Recommendation object: %s", recommendation)
            logger.debug("Type fields present: %s", {k:v for k,v in recommendation.items() if 'type' in k})

            if recommendation.get('music_type'):
                    return 'nightlife'
                
            if recommendation.get('cuisine_type'):
                return 'food'

            if recommendation.get('type'):
                rec_type = recommendation['type'].lower()
                
                for type_set, category in [
                    (cultural_types, 'cultural'),
                    (food_types, 'food'),
                    (nature_types, 'nature'),
                    (nightlife_types, 'nightlife'),
                    (activity_types, 'activity')
                ]:
                    if any(t in rec_type for t in type_set):
                        logger.debug("Category object: %s", category)
                        return category
                
            logger.debug("No category match found, returning 'other'")
            return 'other'
            
        except Exception as e:
            logger.error(f"Error determining category: {str(e)}", exc_info=True)
            return 'other'

    def _calculate_query_understanding(self, query: str, preferences: Dict) -> float:
        """Calculate how well the system understood the query"""
        key_preferences = ['interests', 'locations', 'budget_per_day']
        matches = sum(1 for pref in key_preferences if preferences.get(pref))
        return matches / len(key_preferences)

# Initialize singleton instance
recommender = RecommendationEngine()