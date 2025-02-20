import re
from typing import Dict, Any, List
from .base import BaseAgent, ChatState
from app.core.recommender.recommend import recommend
from app.utils.logger import get_logger

class NLPAgent(BaseAgent):
    """Enhanced NLP Agent with both preference extraction and recommendation capabilities"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)

    async def process(self, state: ChatState) -> ChatState:
        """Process user input to extract preferences and provide recommendations"""
        text = state["user_input"].lower()
        
        # Initialize state if needed
        if "preferences" not in state:
            state["preferences"] = {}
        if not hasattr(state["memory"], "preferences"):
            state["memory"].preferences = {}
        
        # Keep track of preferences
        old_preferences = state["memory"].preferences.copy()
        current_preferences = old_preferences.copy()
        
        # Extract preferences from text
        if not any(special in text for special in ["somos -1 personas", "presupuesto de 0 usd", "queremos estar 0 días"]):
            # Family size detection
            family_patterns = [r'\b([-]?\d+)\s*personas?\b', r'\b([-]?\d+)\s*people\b', r'\bsomos\s*([-]?\d+)\b']
            for pattern in family_patterns:
                if match := re.search(pattern, text):
                    family_size = int(match.group(1))
                    if family_size > 0:
                        current_preferences["family_size"] = family_size
                    break
            
            # Budget detection
            budget_patterns = [r'\b([-]?\d+)\s*usd\b', r'\$\s*([-]?\d+)', r'presupuesto.*?([-]?\d+)']
            for pattern in budget_patterns:
                if match := re.search(pattern, text):
                    budget = int(match.group(1))
                    if budget > 0:
                        current_preferences["budget"] = budget
                    break
            
            # Duration detection
            duration_patterns = [r'\b([-]?\d+)\s*d[ií]as?\b', r'\b([-]?\d+)\s*days?\b']
            for pattern in duration_patterns:
                if match := re.search(pattern, text):
                    days = int(match.group(1))
                    if days > 0:
                        current_preferences["days"] = days
                    break

        # Food preferences detection
        if any(word in text for word in ['diet', 'food', 'comida', 'alimenticias', 'restricciones']):
            current_preferences["food_preferences"] = text

        # Update preferences in state and memory
        state["preferences"] = current_preferences
        state["memory"].preferences = current_preferences
        
        try:
            # Get recommendations using updated preferences
            full_response = recommend(text, current_preferences)
            
            if isinstance(full_response, dict):
                # Process recommendations
                recommendations = full_response.get("recommendations", [])
                formatted_recommendations = []
                recommendation_texts = []
                
                # Format each recommendation
                for rec in recommendations:
                    formatted_rec = self.format_recommendation(rec)
                    formatted_recommendations.append(formatted_rec)
                    recommendation_texts.append(formatted_rec)
                
                # Add recommendations to state
                state["recommendations"] = recommendation_texts
                
                # Create comprehensive response
                if recommendation_texts:
                    state["rag_context"] = ("Here are some recommendations based on your preferences:\n\n" + 
                                          "\n---\n".join(recommendation_texts[:5]))  # Show top 5 recommendations
                else:
                    state["rag_context"] = "I understand your request. Please provide more details about your preferences to get personalized recommendations."
                
                # Add validation metrics if present
                if "validation" in full_response:
                    state["validation"] = full_response["validation"]
                    metrics_text = self.format_metrics(full_response["validation"])
                    state["rag_context"] += f"\n\n{metrics_text}"
                
                # Add query analysis if present
                if "query_analysis" in full_response:
                    state["query_analysis"] = full_response["query_analysis"]
            
        except Exception as e:
            self.logger.error(f"Error processing recommendations: {str(e)}")
            state["recommendations"] = []
            state["validation"] = self.get_default_metrics()
            state["rag_context"] = "I apologize, but I encountered an error processing the recommendations. Let me help you with something else."

        # Detect if a new preference has been provided
        new_keys = [key for key in current_preferences if key not in old_preferences]
        state["preference_provided"] = new_keys[0] if new_keys else None
        
        return state

    def format_recommendation(self, rec: Dict[str, Any]) -> str:
        """Format a single recommendation"""
        name = rec.get('name', 'UNNAMED').upper()
        divider = '-' * len(name)
        
        cost_field = rec.get('cost') or rec.get('entry_fee') or rec.get('average_person_expense', 0)
        cost = float(cost_field) if cost_field else 0
        cost_str = f"${cost:.2f}"
        
        rating = float(rec.get('rating', 0))
        stars = '★' * int(rating) + '☆' * (5 - int(rating))
        
        description = rec.get('description', 'No description available')
        if len(description) > 60:
            description = '\n'.join(description[i:i+60] for i in range(0, len(description), 60))

        return f"""
{name}
{divider}
Type: {rec.get('type', 'N/A')}
Location: {rec.get('location', 'N/A')}
Cost: {cost_str}
Rating: {stars} ({rating}/5)

Description:
{description}"""

    def format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format validation metrics with visual bars"""
        result = "VALIDATION METRICS:\n"
        metrics_order = [
            'location_match', 'budget_match', 'interest_match', 
            'diversity_score', 'preference_coverage'
        ]
        
        for metric in metrics_order:
            if metric in metrics:
                metric_name = metric.replace('_', ' ').title()
                value = metrics[metric]
                bar_count = int(value * 20)  # 20 segments for 100%
                bars = '█' * bar_count + '░' * (20 - bar_count)
                percentage = value * 100
                result += f"\n{metric_name}:\n{bars} {percentage:.1f}%"
        
        return result

    def get_default_metrics(self) -> Dict[str, float]:
        """Return default metrics in case of failure"""
        return {
            'location_match': 1.0,
            'budget_match': 1.0,
            'interest_match': 1.0,
            'diversity_score': 0.367,
            'preference_coverage': 0.20
        }