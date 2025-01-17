# -*- coding: utf-8 -*-
"""Tourism Recommendation System

Enhanced system with comprehensive recommendation features and error handling.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import time
import statistics
from collections import Counter, defaultdict

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedTourismSystem:
    from app.core.analyzer import ImprovedNLPProcessor, DataValidator
    from app.core.data import CSVDatabaseManager
    from app.core.recommender import RecommendationValidator
    """Enhanced tourism system with improved recommendations and error handling"""

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the system and its components"""
        # Database and components initialization
        self.db_manager = CSVDatabaseManager()
        self.nlp_processor = ImprovedNLPProcessor()
        self.data_validator = DataValidator()
        self.recommendation_validator = RecommendationValidator()

        # Define category mappings
        self.category_mappings = {
            'cultural': {
                'keywords': ['museum', 'heritage', 'history', 'art', 'architecture', 'colonial'],
                'locations': ['punda', 'otrobanda', 'pietermaai'],
                'activities': ['walking_tour', 'guided_tour', 'museum_visit']
            },
            'nature': {
                'keywords': ['park', 'trail', 'hiking', 'wildlife', 'bird', 'cave'],
                'locations': ['christoffel park', 'shete boka', 'westpunt'],
                'activities': ['hiking', 'birdwatching', 'nature_walk']
            },
            'water_activities': {
                'keywords': ['beach', 'diving', 'snorkel', 'swim', 'kayak', 'boat'],
                'locations': ['westpunt', 'spanish water', 'piscadera bay'],
                'activities': ['diving', 'snorkeling', 'kayaking']
            },
            'food': {
                'keywords': ['restaurant', 'dining', 'cuisine', 'local food', 'gastronomy'],
                'locations': ['willemstad', 'punda', 'otrobanda'],
                'activities': ['food_tour', 'dining', 'cooking_class']
            },
            'nightlife': {
                'keywords': ['bar', 'club', 'music', 'dance', 'entertainment'],
                'locations': ['pietermaai', 'punda', 'willemstad'],
                'activities': ['live_music', 'dancing', 'nightclub']
            },
            'shopping': {
                'keywords': ['market', 'shop', 'mall', 'boutique', 'craft'],
                'locations': ['willemstad', 'punda', 'otrobanda'],
                'activities': ['shopping_tour', 'market_visit', 'souvenir']
            }
        }

        # Table mappings for recommendations
        self.category_table_mappings = {
            'cultural': {
                'table': 'realistic_curacao_tourist_spots',
                'type_field': 'type',
                'cost_field': 'entry_fee',
                'id_field': 'id_spot',
                'recommendations_field': 'ideal_for',
                'conditions': ['historic site', 'museum', 'cultural']
            },
            'nature': {
                'table': 'realistic_curacao_activities',
                'type_field': 'type',
                'cost_field': 'cost',
                'id_field': 'id_activity',
                'recommendations_field': 'recommended_for',
                'conditions': ['hiking', 'nature', 'park']
            },
            'water_activities': {
                'table': 'realistic_curacao_activities',
                'type_field': 'type',
                'cost_field': 'cost',
                'id_field': 'id_activity',
                'recommendations_field': 'recommended_for',
                'conditions': ['diving', 'snorkeling', 'kayaking']
            },
            'food': {
                'table': 'realistic_curacao_restaurants',
                'type_field': 'cuisine_type',
                'cost_field': 'price_range',
                'id_field': 'id_restaurant',
                'recommendations_field': 'recommended_for',
                'conditions': ['local', 'international', 'seafood']
            },
            'nightlife': {
                'table': 'realistic_curacao_nightclubs',
                'type_field': 'music_type',
                'cost_field': 'price_range',
                'id_field': 'id_nightclub',
                'recommendations_field': 'recommended_for',
                'conditions': ['live music', 'dance', 'bar']
            }
        }
        
                # Scoring weights
        self.scoring_weights = {
            'interest_match': 2.0,
            'location_match': 1.5,
            'budget_match': 1.3,
            'rating_bonus': 1.2,
            'diversity_bonus': 1.1
        }

        # Category requirements for diversity balancing
        self.category_requirements = {
            'cultural': {'min': 2, 'max': 4, 'weight': 1.4},
            'nature': {'min': 1, 'max': 3, 'weight': 1.3},
            'food': {'min': 1, 'max': 2, 'weight': 1.2},
            'adventure': {'min': 1, 'max': 2, 'weight': 1.3},
            'specialty': {'min': 1, 'max': 2, 'weight': 1.5}
        }

        # Initialize LangChain components if API key is provided
        if openai_api_key:
            self.setup_langchain(openai_api_key)
          
    def _get_category_recommendations(self, category: str, preferences: Dict) -> List[Dict]:
        """
        Get recommendations for a specific category with enhanced error handling
        
        Args:
            category: Category to search for recommendations
            preferences: User preferences dictionary
            
        Returns:
            List of recommendations matching criteria
        """
        try:
            if category not in self.category_table_mappings:
                logger.warning(f"Unknown category: {category}")
                return []

            mapping = self.category_table_mappings[category]

            # Build query conditions
            conditions = []

            # Location filtering with fuzzy matching
            if preferences.get('locations'):
                locations = [loc.lower().strip() for loc in preferences['locations'] if loc]
                if locations:
                    location_clauses = []
                    for location in locations:
                        location_clauses.append(f"LOWER(location) LIKE '%{location}%'")
                    if location_clauses:
                        conditions.append(f"({' OR '.join(location_clauses)})")

            # Type conditions
            type_clauses = []
            for condition in mapping['conditions']:
                type_clauses.append(f"LOWER({mapping['type_field']}) LIKE '%{condition}%'")
            if type_clauses:
                conditions.append(f"({' OR '.join(type_clauses)})")

            # Enhanced budget handling with string cost support
            if preferences.get('budget_per_day'):
                try:
                    budget = float(preferences['budget_per_day'])
                    if mapping['cost_field'] == 'price_range':
                        # Handle categorical price ranges
                        conditions.append(f"""
                            CASE
                                WHEN LOWER({mapping['cost_field']}) = 'low' THEN 50
                                WHEN LOWER({mapping['cost_field']}) = 'medium' THEN 100
                                WHEN LOWER({mapping['cost_field']}) = 'high' THEN 200
                                WHEN CAST({mapping['cost_field']} AS FLOAT) > 0 THEN CAST({mapping['cost_field']} AS FLOAT)
                                ELSE 0
                            END <= {budget}
                        """)
                    else:
                        # Direct numeric comparison with CAST
                        conditions.append(f"CAST(COALESCE({mapping['cost_field']}, '0') AS FLOAT) <= {budget}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Budget conversion error: {str(e)}")

            # Build WHERE clause
            where_clause = " AND ".join(conditions) if conditions else "1=1"

            # Construct final query with cost handling
            query = f"""
                SELECT DISTINCT
                    '{category}' as source,
                    {mapping['id_field']} as id,
                    name,
                    {mapping['type_field']} as type,
                    location,
                    CASE
                        WHEN LOWER({mapping['cost_field']}) = 'low' THEN '50'
                        WHEN LOWER({mapping['cost_field']}) = 'medium' THEN '100'
                        WHEN LOWER({mapping['cost_field']}) = 'high' THEN '200'
                        ELSE COALESCE({mapping['cost_field']}, '0')
                    END as cost,
                    rating,
                    CASE
                        WHEN rating >= 4.5 THEN rating * 1.3
                        WHEN rating >= 4.0 THEN rating * 1.2
                        WHEN rating >= 3.5 THEN rating * 1.1
                        ELSE rating
                    END as adjusted_rating,
                    description,
                    COALESCE({mapping['recommendations_field']}, '') as recommended_for
                FROM {mapping['table']}
                WHERE {where_clause}
                ORDER BY rating DESC, adjusted_rating DESC
                LIMIT 5
            """

            try:
                results = self.db_manager.execute_query(query)

                # If no results, try with relaxed conditions
                if not results:
                    logger.info(f"No results for {category} with strict conditions. Trying relaxed search...")
                    # Remove budget condition if present
                    conditions = [c for c in conditions if 'cost' not in c.lower() and 'price' not in c.lower()]
                    where_clause = " AND ".join(conditions) if conditions else "1=1"

                    # Retry query with relaxed conditions
                    query = query.replace(f"WHERE {where_clause}", f"WHERE {where_clause}")
                    results = self.db_manager.execute_query(query)

                # Post-process results to ensure valid cost values
                processed_results = []
                for result in results:
                    try:
                        # Convert cost to float if possible
                        cost_str = str(result.get('cost', '0')).lower()
                        if cost_str in ['low', 'medium', 'high']:
                            cost_mapping = {'low': 50, 'medium': 100, 'high': 200}
                            result['cost'] = cost_mapping[cost_str]
                        else:
                            result['cost'] = float(cost_str) if cost_str.replace('.','',1).isdigit() else 0
                        processed_results.append(result)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error processing result cost: {e}")
                        result['cost'] = 0
                        processed_results.append(result)

                return processed_results

            except Exception as e:
                logger.error(f"Database query failed for category {category}: {str(e)}")
                return []

        except Exception as e:
            logger.error(f"Error in category recommendations for {category}: {str(e)}")
            return []

    def _handle_missing_data(self, recommendations: List[Dict], preferences: Dict) -> List[Dict]:
        """Enhanced recovery for missing recommendations"""
        if recommendations:
            return recommendations

        logger.info("No recommendations found with initial criteria. Attempting recovery...")

        try:
            # Create relaxed preferences
            relaxed_prefs = preferences.copy()

            # Remove strict constraints
            relaxed_prefs.pop('budget_per_day', None)
            relaxed_prefs.pop('specific_sites', None)
            relaxed_prefs.pop('special_interests', None)

            # Keep only primary location if multiple exist
            if relaxed_prefs.get('locations'):
                relaxed_prefs['locations'] = [relaxed_prefs['locations'][0]]

            # Keep only primary interests
            if relaxed_prefs.get('interests'):
                relaxed_prefs['interests'] = relaxed_prefs['interests'][:2]

            # Try getting recommendations with relaxed preferences
            recovery_recommendations = []
            for category in self.category_table_mappings:
                try:
                    category_recs = self._get_category_recommendations(category, relaxed_prefs)
                    if category_recs:
                        recovery_recommendations.extend(category_recs)
                except Exception as e:
                    logger.error(f"Error in recovery recommendations for {category}: {str(e)}")
                    continue

            if recovery_recommendations:
                logger.info(f"Retrieved {len(recovery_recommendations)} recommendations with relaxed criteria")
                return recovery_recommendations[:10]  # Limit to top 10

            # If still no results, try with minimal criteria
            minimal_prefs = {
                'locations': preferences.get('locations', [])[:1],
                'interests': preferences.get('interests', [])[:1]
            }

            minimal_recommendations = []
            for category in self.category_table_mappings:
                try:
                    category_recs = self._get_category_recommendations(category, minimal_prefs)
                    if category_recs:
                        minimal_recommendations.extend(category_recs)
                except Exception as e:
                    logger.error(f"Error in minimal recommendations for {category}: {str(e)}")
                    continue

            return minimal_recommendations[:10] if minimal_recommendations else []

        except Exception as e:
            logger.error(f"Error in recommendation recovery: {str(e)}")
            return []

    def _analyze_preference_coverage(self, recommendations: List[Dict], preferences: Dict) -> Dict[str, float]:
        """Analyze how well recommendations cover all user preferences"""
        try:
            coverage = {
                'interests_coverage': 0.0,
                'locations_coverage': 0.0,
                'price_range_coverage': 0.0,
                'overall_coverage': 0.0
            }

            # Calculate coverage scores
            if preferences.get('interests'):
                matched_interests = set()
                for rec in recommendations:
                    rec_text = f"{rec.get('type', '')} {rec.get('description', '')}".lower()
                    for interest in preferences['interests']:
                        if interest.lower() in rec_text:
                            matched_interests.add(interest.lower())
                coverage['interests_coverage'] = len(matched_interests) / len(preferences['interests'])

            # Calculate overall coverage
            coverage['overall_coverage'] = statistics.mean(
                [v for v in coverage.values() if v > 0]
            ) if any(v > 0 for v in coverage.values()) else 0.0

            return coverage
        except Exception as e:
            logger.error(f"Error analyzing preference coverage: {str(e)}")
            return {'overall_coverage': 0.0}

    def _merge_and_validate_preferences(self, extracted: Dict, explicit: Dict) -> Dict:
        """Merge and validate preferences from multiple sources"""
        try:
            merged = {}

            # Validate and merge interests
            interests = set(extracted.get('interests', []))
            interests.update(explicit.get('interests', []))
            merged['interests'] = [i for i in interests if self.data_validator.normalize_interest(i)]

            # Validate and merge locations
            locations = set(extracted.get('locations', []))
            locations.update(explicit.get('locations', []))
            merged['locations'] = [l for l in locations if self.data_validator.normalize_location(l)]

            # Validate budget
            budget = explicit.get('budget_per_day', extracted.get('budget'))
            if budget:
                merged['budget_per_day'] = float(budget)

            # Add other preferences
            for key, value in explicit.items():
                if key not in merged and value is not None:
                    merged[key] = value

            return merged
        except Exception as e:
            logger.error(f"Error merging preferences: {str(e)}")
            return {}  # Return empty dict on error
        
    def _add_debug_logs(self, stage: str, data: Any) -> None:
        """Add detailed debug logs for troubleshooting"""
        try:
            if stage == 'query':
                logger.debug(f"Generated SQL Query:\n{data}")
            elif stage == 'original_preferences':
                logger.debug(f"Original preferences:\n{json.dumps(data, indent=2)}")
            elif stage == 'extracted_preferences':
                logger.debug(f"Extracted preferences:\n{json.dumps(data, indent=2)}")
            elif stage == 'merged_preferences':
                logger.debug(f"Merged preferences:\n{json.dumps(data, indent=2)}")
            elif stage == 'final_recommendations':
                logger.debug(f"Found {len(data)} recommendations")
                for idx, rec in enumerate(data[:3], 1):
                    logger.debug(f"Top {idx} recommendation: {rec.get('name')} ({rec.get('type')})")
            else:
                logger.debug(f"{stage}: {data}")
        except Exception as e:
            logger.error(f"Error in debug logging: {str(e)}")

    def _calculate_query_understanding(self, query: str, extracted_prefs: Dict, merged_prefs: Dict) -> float:
        """Calculate how well the system understood the query"""
        try:
            matches = 0
            total_points = 0

            # Check key preference extractions
            key_preferences = ['interests', 'locations', 'budget_per_day', 'trip_duration']
            for pref in key_preferences:
                if pref in merged_prefs and merged_prefs[pref]:
                    matches += 1
                total_points += 1

            # Calculate final score
            return matches / total_points if total_points > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating query understanding: {str(e)}")
            return 0.0

    def _handle_error(self, error_type: str, message: str) -> Dict[str, Any]:
        """Handle errors consistently and return detailed error information."""
        error_message = ERROR_TYPES.get(error_type, 'Unknown error')
        logger.error(f"{error_message}: {message}")

        return {
            "status": "error",
            "error_type": error_type,
            "error_message": f"{error_message}: {message}",
            "recommendations": [],
            "validation": {
                'location_match': 0.0,
                'budget_match': 0.0,
                'interest_match': 0.0,
                'diversity_score': 0.0,
                'preference_coverage': 0.0
            },
            "query_analysis": {
                "intent_scores": {},
                "extracted_preferences": {},
                "merged_preferences": {}
            },
            "metadata": {
                "query_time": datetime.now().isoformat(),
                "preference_count": 0,
                "recommendation_count": 0
            }
        }




    def setup_langchain(self, api_key: str):
        """Setup LangChain components with enhanced OpenAI configuration"""
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo",
            api_key=api_key,
            max_tokens=2000,
            presence_penalty=0.2,
            frequency_penalty=0.3
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )

        self.tools = [
            Tool(
                name="get_recommendations",
                func=self.get_recommendations,
                description="Get personalized tourism recommendations based on preferences"
            ),
            Tool(
                name="analyze_preferences",
                func=lambda x: self.nlp_processor.extract_preferences(x),
                description="Analyze and extract user preferences from natural language"
            )
        ]

        self.agent = self._setup_agent()

    def _setup_agent(self) -> AgentExecutor:
        """Setup LangChain agent with enhanced prompts"""
        system_prompt = """
        You are an expert tourism assistant for Curacao specializing in:
        - Personalized travel recommendations
        - Activity and accommodation suggestions
        - Budget planning
        - Accessible tourism
        - Cultural and historical insights
        - Local cuisine and dining experiences
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", "Agent Scratchpad: {agent_scratchpad}")
        ])

        agent = create_openai_functions_agent(
            llm=self.llm,
            prompt=prompt,
            tools=self.tools
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )
        
    def _score_recommendations(
    self, 
    recommendations: List[Dict], 
    preferences: Dict
) -> List[Dict]:
        
        """Score recommendations based on preferences"""
        scored_recommendations = []
        for rec in recommendations:
            try:
                score = self._calculate_recommendation_score(rec, preferences)
                scored_recommendations.append({**rec, '_score': score})
            except Exception as e:
                logger.error(f"Error scoring recommendation: {str(e)}")
                continue
        return sorted(
            scored_recommendations, 
            key=lambda x: x.get('_score', 0), 
            reverse=True
        )

    def get_recommendations(
        self, 
        preferences: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict]:
        """Get recommendations with enhanced error handling"""
        try:
            recommendations = []
            errors = []

            # Get initial recommendations for each category
            for category in self.category_table_mappings:
                try:
                    category_recs = self._get_category_recommendations(
                        category, 
                        preferences
                    )
                    if category_recs:
                        recommendations.extend(category_recs)
                except Exception as e:
                    errors.append(
                        f"Error getting {category} recommendations: {str(e)}"
                    )
                    continue

            if not recommendations and errors:
                logger.error(
                    "Errors occurred while getting recommendations:\n" + 
                    "\n".join(errors)
                )
                recommendations = self._handle_missing_data(
                    recommendations, 
                    preferences
                )

            # Score and sort recommendations
            scored_recommendations = self._score_recommendations(
                recommendations, 
                preferences
            )

            # Balance and diversify recommendations
            balanced_recommendations = self._ensure_recommendation_diversity(
                scored_recommendations,
                limit=limit
            )

            return balanced_recommendations

        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}", exc_info=True)
            return []

    def process_query(self, query: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Process user query and generate recommendations with enhanced metrics"""
        start_time = time.time()
        processing_times = {}

        try:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Invalid query format")

            # Initialize performance metrics
            metrics = {
                'processing_stages': {},
                'recommendation_stats': {},
                'validation_scores': {},
                'performance_metrics': {}
            }

            # Extract and merge preferences with timing
            stage_start = time.time()
            extracted_prefs = self.nlp_processor.extract_preferences(query)
            processing_times['preference_extraction'] = time.time() - stage_start
            
            stage_start = time.time()
            merged_preferences = self._merge_and_validate_preferences(extracted_prefs, preferences)
            processing_times['preference_merging'] = time.time() - stage_start

            # Get recommendations with timing
            stage_start = time.time()
            recommendations = self.get_recommendations(merged_preferences)
            processing_times['recommendation_generation'] = time.time() - stage_start

            # Enhanced recommendation statistics
            recommendation_stats = {
                'total_count': len(recommendations),
                'type_distribution': self._get_type_distribution(recommendations),
                'price_range_distribution': self._get_price_distribution(recommendations),
                'location_distribution': self._get_location_distribution(recommendations),
                'rating_distribution': self._get_rating_distribution(recommendations)
            }

            # Validate recommendations with timing
            stage_start = time.time()
            validation_results = self.recommendation_validator.validate(
                recommendations,
                merged_preferences
            )
            processing_times['validation'] = time.time() - stage_start

            # Get intent analysis with timing
            stage_start = time.time()
            intent_analysis = self.nlp_processor.classify_intent(query)
            processing_times['intent_analysis'] = time.time() - stage_start

            # Calculate overall performance metrics
            total_time = time.time() - start_time
            performance_metrics = {
                'total_processing_time': total_time,
                'processing_times': processing_times,
                'average_time_per_recommendation': total_time / len(recommendations) if recommendations else 0,
                'preference_processing_ratio': processing_times['preference_extraction'] / total_time,
                'recommendation_generation_ratio': processing_times['recommendation_generation'] / total_time
            }

            # Construct enhanced response
            response = {
                "status": "success",
                "recommendations": recommendations,
                "validation": validation_results,
                "query_analysis": {
                    "intent_scores": intent_analysis,
                    "extracted_preferences": extracted_prefs,
                    "merged_preferences": merged_preferences,
                    "query_understanding_score": self._calculate_query_understanding(
                        query, extracted_prefs, merged_preferences
                    )
                },
                "metadata": {
                    "query_time": datetime.now().isoformat(),
                    "preference_count": len(merged_preferences),
                    "recommendation_count": len(recommendations),
                    "processing_times": processing_times,
                    "performance_metrics": performance_metrics
                },
                "statistics": {
                    "recommendation_stats": recommendation_stats,
                    "validation_metrics": validation_results,
                    "coverage_analysis": self._analyze_preference_coverage(
                        recommendations, merged_preferences
                    )
                }
            }

            return response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            error_response = self._handle_error("query_error", str(e))
            error_response['metadata'] = {
                'error_time': datetime.now().isoformat(),
                'processing_time': time.time() - start_time,
                'processing_stage_times': processing_times
            }
            return error_response

    def _calculate_recommendation_score(self, item: Dict, preferences: Dict) -> float:
        """Enhanced scoring algorithm with weighted criteria"""
        base_score = 0.0
        try:
            # Base rating score (0-5 scale)
            rating = float(item.get('rating', 0))
            base_score = rating / 5.0

            # Interest matching with enhanced scoring
            if preferences.get('interests'):
                interest_score = 0
                item_text = f"{item.get('type', '')} {item.get('description', '')} {item.get('recommended_for', '')}".lower()

                for interest in preferences['interests']:
                    # Direct match
                    if interest.lower() in item_text:
                        interest_score += 1.0

                    # Category matching
                    for category, keywords in self.INTEREST_MAPPINGS.items():
                        if interest in category or any(kw in interest for kw in keywords):
                            if any(kw in item_text for kw in keywords):
                                interest_score += 0.8

                interest_score = min(interest_score / len(preferences['interests']), 1.0)
                base_score += interest_score * self.scoring_weights['interest_match']

            # Location matching with proximity bonus
            if preferences.get('locations'):
                location_score = 0
                item_location = item.get('location', '').lower()

                for location in preferences['locations']:
                    if location.lower() in item_location:
                        location_score = 1.0
                        # Proximity bonus for exact match
                        if location.lower() == item_location:
                            location_score *= 1.2
                        break

                base_score += location_score * self.scoring_weights['location_match']

            # Budget matching with range consideration
            if preferences.get('budget_per_day'):
                budget = float(preferences['budget_per_day'])
                item_cost = float(item.get('cost', 0)) or float(item.get('price', 0))

                if item_cost <= budget:
                    budget_score = 1.0
                    # Bonus for being well within budget
                    if item_cost <= budget * 0.7:
                        budget_score *= 1.2
                    base_score += budget_score * self.scoring_weights['budget_match']

            # Rating bonus for highly-rated items
            if rating >= 4.5:
                base_score *= self.scoring_weights['rating_bonus']

            # Diversity bonus based on unique attributes
            if (item.get('type') not in self._seen_types and
                item.get('location') not in self._seen_locations):
                base_score *= self.scoring_weights['diversity_bonus']

            return base_score

        except Exception as e:
            logger.error(f"Error calculating recommendation score: {str(e)}")
            return 0.0

    def _ensure_recommendation_diversity(self, recommendations: List[Dict]) -> List[Dict]:
        """Enhanced diversity handling with comprehensive balancing"""
        try:
            if not recommendations:
                return []

            # Deduplicate recommendations
            unique_recs = {}
            for rec in recommendations:
                key = f"{rec.get('name', '')}-{rec.get('location', '')}-{rec.get('type', '')}"
                if key not in unique_recs:
                    unique_recs[key] = rec

            # Initialize tracking dictionaries with category balancing
            diverse_recs = []
            type_counts = defaultdict(int)
            location_counts = defaultdict(int)
            price_ranges = defaultdict(int)
            category_counts = defaultdict(int)

            # Define diversity thresholds and category limits
            MAX_PER_TYPE = 3
            MAX_PER_LOCATION = 2
            MAX_PRICE_RANGE = 3
            TARGET_SIZE = min(10, len(unique_recs))

            # Sort by composite score (adjusted rating + diversity bonus)
            sorted_recs = sorted(
                unique_recs.values(),
                key=lambda x: (
                    float(x.get('adjusted_rating', 0)) +
                    float(x.get('rating', 0)) * 0.5 +
                    (1.0 if x.get('type', '').lower() not in type_counts else 0.0) +
                    (0.5 if x.get('location', '').lower() not in location_counts else 0.0)
                ),
                reverse=True
            )

            # First pass: ensure minimum category requirements
            for rec in sorted_recs[:]:
                category = self._get_recommendation_category(rec)
                if category in self.category_requirements:
                    if (category_counts[category] < self.category_requirements[category]['min'] and
                        len(diverse_recs) < TARGET_SIZE):
                        diverse_recs.append(rec)
                        category_counts[category] += 1
                        type_counts[str(rec.get('type', '')).lower()] += 1
                        location_counts[str(rec.get('location', '')).lower()] += 1
                        price_ranges[self._get_price_category(rec)] += 1
                        sorted_recs.remove(rec)

            # Second pass: fill remaining slots while maintaining diversity
            for rec in sorted_recs:
                rec_type = str(rec.get('type', '')).lower()
                location = str(rec.get('location', '')).lower()
                price_category = self._get_price_category(rec)
                category = self._get_recommendation_category(rec)

                # Check all diversity conditions
                if (len(diverse_recs) < TARGET_SIZE and
                    type_counts[rec_type] < MAX_PER_TYPE and
                    location_counts[location] < MAX_PER_LOCATION and
                    price_ranges[price_category] < MAX_PRICE_RANGE and
                    (category not in self.category_requirements or
                     category_counts[category] < self.category_requirements[category]['max'])):

                    # Calculate diversity bonus
                    diversity_bonus = (
                        (1.0 if type_counts[rec_type] == 0 else 0.5) +
                        (1.0 if location_counts[location] == 0 else 0.3) +
                        (0.5 if price_ranges[price_category] == 0 else 0.2)
                    )

                    # Add recommendation with diversity score
                    rec['diversity_score'] = diversity_bonus
                    diverse_recs.append(rec)

                    # Update counters
                    type_counts[rec_type] += 1
                    location_counts[location] += 1
                    price_ranges[price_category] += 1
                    if category in self.category_requirements:
                        category_counts[category] += 1

            return diverse_recs

        except Exception as e:
            logger.error(f"Error ensuring diversity: {str(e)}", exc_info=True)
            return recommendations[:10] if recommendations else []

    def _get_recommendation_category(self, recommendation: Dict) -> str:
        """Determine the primary category of a recommendation"""
        rec_type = str(recommendation.get('type', '')).lower()

        # Category mapping
        for category, info in self.category_mappings.items():
            if any(keyword in rec_type for keyword in info['keywords']):
                return category

        # Check description if type is inconclusive
        description = str(recommendation.get('description', '')).lower()
        for category, info in self.category_mappings.items():
            if any(keyword in description for keyword in info['keywords']):
                return category

        return 'other'

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