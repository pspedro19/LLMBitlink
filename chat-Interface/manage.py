#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()

    def _calculate_recommendation_score(self, item: Dict, preferences: Dict) -> float:
        """Enhanced scoring algorithm with weighted criteria"""
        base_score = 0.0
        weights = {
            'interest_match': 2.0,
            'location_match': 1.5,
            'budget_match': 1.3,
            'rating_bonus': 1.2,
            'diversity_bonus': 1.1
        }

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
                base_score += interest_score * weights['interest_match']

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

                base_score += location_score * weights['location_match']

            # Budget matching with range consideration
            if preferences.get('budget_per_day'):
                budget = float(preferences['budget_per_day'])
                item_cost = float(item.get('cost', 0)) or float(item.get('price', 0))

                if item_cost <= budget:
                    budget_score = 1.0
                    # Bonus for being well within budget
                    if item_cost <= budget * 0.7:
                        budget_score *= 1.2
                    base_score += budget_score * weights['budget_match']

            # Rating bonus for highly-rated items
            if rating >= 4.5:
                base_score *= weights['rating_bonus']

            # Diversity bonus based on unique attributes
            if (item.get('type') not in self._seen_types and
                item.get('location') not in self._seen_locations):
                base_score *= weights['diversity_bonus']

            return base_score

        except Exception as e:
            logger.error(f"Error calculating recommendation score: {str(e)}")
            return 0.0

