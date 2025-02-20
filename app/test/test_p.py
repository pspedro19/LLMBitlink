import pytest
from typing import Dict, List, Any
from app.core.recommender.recommend import recommend  

def format_recommendation(rec: Dict[str, Any]) -> str:
    """Format a single recommendation in a readable way"""
    name = rec.get('name', 'UNNAMED').upper()
    divider = '-' * len(name)
    
    # Format various fields
    cost_field = rec.get('cost') or rec.get('entry_fee') or rec.get('average_person_expense', 0)
    cost = float(cost_field) if cost_field else 0
    cost_str = f"${cost:.2f}"
    
    # Format rating with stars
    rating = float(rec.get('rating', 0))
    stars = '★' * int(rating) + '☆' * (5 - int(rating))
    
    # Format description
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

def format_metrics(metrics: Dict[str, float]) -> str:
    """Format validation metrics with visual bars"""
    result = "\n==================================================\nVALIDATION METRICS\n==================================================\n\n"
    
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
            result += f"{metric_name}:\n{bars} {percentage:.1f}%\n\n"
    
    return result

def format_query_analysis(analysis: Dict[str, Any]) -> str:
    """Format query analysis information"""
    result = "\n==================================================\nQUERY ANALYSIS\n==================================================\n\n"
    
    if 'intent_scores' in analysis:
        result += "Intent Analysis:\n"
        for intent, score in sorted(analysis['intent_scores'].items()):
            if score > 0:  # Only show intents with value > 0
                intent_name = intent.replace('_', ' ').title()
                result += f"{intent_name}: {score * 100:.1f}%\n"
    
    return result

def test_recommendation_system():
    test_cases = [
        {
            "query": "I want to explore Punda and Otrobanda's cultural highlights",
            "preferences": {
                "interests": ["culture", "history", "architecture"],
                "budget_per_day": 100,
                "locations": ["Punda", "Otrobanda"],
                "language_preferences": ["English"],
                "travel_style": "cultural",
                "accessibility_needs": False
            }
        },
        # Add more test cases as needed
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test Case {i}: {case['preferences']['interests']}")
        print(f"{'=' * 80}\n")

        try:
            # Get recommendations and full response
            full_response = recommend(case["query"], case["preferences"])
            
            # Basic validations with detailed logging
            assert isinstance(full_response, dict), f"Test case {i} failed: Response is not a dictionary"
            assert "recommendations" in full_response, f"Test case {i} failed: No recommendations key in response"
            recommendations = full_response.get("recommendations", [])
            assert len(recommendations) > 0, f"Test case {i} failed: No recommendations found"
            
            # Show recommendations
            print("==================================================")
            print("RECOMMENDATIONS")
            print("==================================================")
            
            for rec in recommendations:
                print(format_recommendation(rec))
            
            # Show validation metrics if present
            if "validation" in full_response:
                print(format_metrics(full_response["validation"]))
            
            # Show query analysis if present
            if "query_analysis" in full_response:
                print(format_query_analysis(full_response["query_analysis"]))
            
            # Show additional metadata
            if "metadata" in full_response:
                print("\nMETADATA:")
                print(f"Processing time: {full_response['metadata'].get('total_processing_time', 'N/A')}s")
                print(f"Total recommendations: {full_response['metadata'].get('recommendation_count', 0)}")
            
            print('-' * 80)
            
        except Exception as e:
            print(f"Error in test case {i}:")
            print(f"Query: {case['query'][:100]}...")
            print(f"Preferences: {case['preferences']}")
            print(f"Error: {str(e)}")
            raise

if __name__ == "__main__":
    pytest.main([__file__])
    


