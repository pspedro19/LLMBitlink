"""
Helper functions for formatting and displaying results.
"""
import textwrap
from typing import Dict, Any
from utils.logger import get_logger

logger = get_logger(__name__)

def print_results(result: Dict[str, Any], validation: Dict[str, Any]) -> None:
    """
    Enhanced results printing with improved formatting and error handling
    
    Args:
        result (Dict[str, Any]): Results to print
        validation (Dict[str, Any]): Validation metrics to print
    """
    try:
        _print_recommendations(result)
        _print_validation_metrics(validation)
        _print_query_analysis(result)
    except Exception as e:
        logger.error(f"Error printing results: {str(e)}")

def _print_recommendations(result: Dict[str, Any]) -> None:
    """Print formatted recommendations section"""
    print("\n" + "="*50)
    print("RECOMMENDATIONS")
    print("="*50)

    for i, rec in enumerate(result.get("recommendations", []), 1):
        try:
            print(f"\n{i}. {rec.get('name', 'N/A').upper()}")
            print("-" * (len(str(i)) + 2 + len(rec.get('name', 'N/A'))))

            # Main details
            print(f"Type: {rec.get('type', 'N/A')}")
            print(f"Location: {rec.get('location', 'N/A')}")

            # Cost information with formatting
            cost = rec.get('cost', 0)
            if cost:
                try:
                    print(f"Cost: ${float(cost):.2f}")
                except (ValueError, TypeError):
                    print(f"Cost: {cost}")

            # Rating with stars
            try:
                rating = float(rec.get('rating', 0))
                stars = "★" * int(rating) + "☆" * (5 - int(rating))
                print(f"Rating: {stars} ({rating:.1f}/5)")
            except (ValueError, TypeError):
                print(f"Rating: N/A")

            # Description with proper wrapping
            if 'description' in rec and rec['description']:
                print("\nDescription:")
                try:
                    desc = str(rec['description'])
                    wrapped_desc = textwrap.fill(desc[:200] + "..." if len(desc) > 200 else desc, 70)
                    print(wrapped_desc)
                except Exception as e:
                    logger.error(f"Error formatting description: {str(e)}")
                    print("Description unavailable")

        except Exception as e:
            logger.error(f"Error printing recommendation {i}: {str(e)}")
            continue

def _print_validation_metrics(validation: Dict[str, Any]) -> None:
    """Print formatted validation metrics section"""
    print("\n" + "="*50)
    print("VALIDATION METRICS")
    print("="*50)

    try:
        for metric, score in validation.items():
            try:
                percentage = float(score) * 100
                bar_length = int(percentage / 5)  # 20 characters for 100%
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"\n{metric.replace('_', ' ').title()}:")
                print(f"{bar} {percentage:.1f}%")
            except (ValueError, TypeError):
                print(f"{metric.replace('_', ' ').title()}: N/A")
    except Exception as e:
        logger.error(f"Error printing validation metrics: {str(e)}")

def _print_query_analysis(result: Dict[str, Any]) -> None:
    """Print formatted query analysis section"""
    print("\n" + "="*50)
    print("QUERY ANALYSIS")
    print("="*50)

    try:
        print("\nIntent Analysis:")
        for intent, score in result.get("query_analysis", {}).get("intent_scores", {}).items():
            try:
                percentage = float(score) * 100
                print(f"{intent.replace('_', ' ').title()}: {percentage:.1f}%")
            except (ValueError, TypeError):
                print(f"{intent.replace('_', ' ').title()}: N/A")
    except Exception as e:
        logger.error(f"Error printing intent analysis: {str(e)}")

def format_price(amount: float) -> str:
    """
    Format price with proper currency symbol and decimals
    
    Args:
        amount (float): Price amount to format
        
    Returns:
        str: Formatted price string
    """
    try:
        return f"${amount:.2f}" if amount else "N/A"
    except (ValueError, TypeError):
        return "N/A"

def format_rating(rating: float) -> str:
    """
    Format rating with stars
    
    Args:
        rating (float): Rating value to format
        
    Returns:
        str: Formatted rating string with stars
    """
    try:
        rating = float(rating)
        stars = "★" * int(rating) + "☆" * (5 - int(rating))
        return f"{stars} ({rating:.1f}/5)"
    except (ValueError, TypeError):
        return "N/A"

def wrap_text(text: str, width: int = 70, max_length: int = 200) -> str:
    """
    Wrap text with proper width and length limits
    
    Args:
        text (str): Text to wrap
        width (int): Maximum line width
        max_length (int): Maximum total text length
        
    Returns:
        str: Wrapped text string
    """
    try:
        if not text:
            return "N/A"
        text = str(text)
        if len(text) > max_length:
            text = text[:max_length] + "..."
        return textwrap.fill(text, width)
    except Exception:
        return "N/A"

def format_percentage(value: float) -> str:
    """
    Format percentage value with proper decimals
    
    Args:
        value (float): Percentage value to format
        
    Returns:
        str: Formatted percentage string
    """
    try:
        return f"{float(value):.1f}%" if value is not None else "N/A"
    except (ValueError, TypeError):
        return "N/A"

def create_progress_bar(percentage: float, width: int = 20) -> str:
    """
    Create a visual progress bar
    
    Args:
        percentage (float): Percentage value (0-100)
        width (int): Width of the progress bar
        
    Returns:
        str: Progress bar string
    """
    try:
        percentage = min(max(float(percentage), 0), 100)
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"{bar} {percentage:.1f}%"
    except (ValueError, TypeError):
        return "░" * width + " N/A"

ERROR_MESSAGES = {
    'validation_error': 'Data validation failed',
    'recommendation_error': 'Unable to generate recommendations',
    'query_error': 'Error processing query',
    'database_error': 'Database operation failed',
    'preference_error': 'Error processing preferences'
}"""
Helper functions for formatting and displaying results.
"""
import textwrap
from typing import Dict, Any
from utils.logger import get_logger

logger = get_logger(__name__)

def print_results(result: Dict[str, Any], validation: Dict[str, Any]) -> None:
    """
    Enhanced results printing with improved formatting and error handling
    
    Args:
        result (Dict[str, Any]): Results to print
        validation (Dict[str, Any]): Validation metrics to print
    """
    try:
        _print_recommendations(result)
        _print_validation_metrics(validation)
        _print_query_analysis(result)
    except Exception as e:
        logger.error(f"Error printing results: {str(e)}")

def _print_recommendations(result: Dict[str, Any]) -> None:
    """Print formatted recommendations section"""
    print("\n" + "="*50)
    print("RECOMMENDATIONS")
    print("="*50