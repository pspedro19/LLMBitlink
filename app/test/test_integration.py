import pytest

# Assuming you have a function `recommend` in your recommendation system
from app.core.recommender import recommend

def test_recommendation_system():
    test_cases = [
        # 1. Punda & Otrobanda Cultural Experience
        {
            "query": """Want to explore the UNESCO World Heritage sites in Punda and
                    Otrobanda, focusing on the colorful Dutch architecture and Queen
                    Emma Bridge. 4-day trip with local food tasting at Plasa Bieu.
                    Interested in the Mikvé Israel-Emanuel Synagogue and Maritime
                    Museum. Budget $200/day for 2 people.""",
            "preferences": {
                "interests": ["cultural", "history", "architecture", "food"],
                "locations": ["Punda", "Otrobanda"],
                "budget_per_day": 200.0,
                "trip_duration": 4,
                "group_size": 2,
                "activity_types": ["walking_tour", "museum_visits", "food_tasting"],
                "specific_sites": ["Queen Emma Bridge", "Mikvé Israel-Emanuel Synagogue"],
                "cuisine_preferences": ["local"]
            }
        },

        # 2. Westpunt Beach & Diving Adventure
        {
            "query": """Planning diving trip to Westpunt, especially Playa Kalki and
                    Grote Knip. Want to stay near Blue Room Cave and do shore diving.
                    5 days, certified divers, need equipment rental. Also interested in
                    visiting Shete Boka National Park. Need beachfront accommodation.""",
            "preferences": {
                "interests": ["diving", "nature", "beaches"],
                "locations": ["Westpunt", "Grote Knip", "Playa Kalki"],
                "trip_duration": 5,
                "group_size": 2,
                "activity_types": ["diving", "snorkeling", "hiking"],
                "equipment_needs": ["diving_gear"],
                "specific_sites": ["Blue Room Cave", "Shete Boka"],
                "accommodation_type": "beachfront"
            }
        },

        # 3. Willemstad Local Food Tour
        {
            "query": """Looking for authentic Curaçaoan food experience in Willemstad.
                    Want to try local dishes at Plasa Bieu, visit Floating Market, and
                    take cooking classes to learn about krioyo cuisine. 3 days focused
                    on food, especially interested in fresh fish at Fisherman's Wharf.""",
            "preferences": {
                "interests": ["food", "local_cuisine", "cooking"],
                "locations": ["Willemstad"],
                "trip_duration": 3,
                "group_size": 2,
                "activity_types": ["food_tour", "cooking_class", "market_visit"],
                "specific_sites": ["Plasa Bieu", "Floating Market", "Fisherman's Wharf"],
                "cuisine_preferences": ["krioyo", "seafood"]
            }
        },

        # 4. Christoffel Park Nature Experience
        {
            "query": """Want to hike Christoffel Mountain early morning and explore
                    the park's nature trails. Interested in bird watching, especially
                    for the native barn owl and yellow oriole. 2-day visit including
                    Savonet Museum. Need transportation from Willemstad.""",
            "preferences": {
                "interests": ["nature", "hiking", "birdwatching"],
                "locations": ["Christoffel Park"],
                "trip_duration": 2,
                "group_size": 1,
                "activity_types": ["hiking", "birdwatching", "museum"],
                "specific_sites": ["Christoffel Mountain", "Savonet Museum"],
                "time_preferences": ["early_morning"],
                "transportation_needs": True
            }
        },

        # 5. Pietermaai District Cultural Stay
        {
            "query": """Seeking boutique hotel in Pietermaai District for 6 days.
                    Want to experience the nightlife, street art, and restored colonial
                    buildings. Interested in live music at local bars and trying craft
                    cocktails. Walking distance to restaurants and cafes.""",
            "preferences": {
                "interests": ["nightlife", "culture", "art"],
                "locations": ["Pietermaai"],
                "trip_duration": 6,
                "group_size": 2,
                "activity_types": ["nightlife", "walking", "dining"],
                "accommodation_type": "boutique",
                "entertainment": ["live_music", "bars"],
                "walking_distance": True
            }
        },

        # 6. Spanish Water Watersports
        {
            "query": """Want to do watersports at Spanish Water - especially
                    windsurfing and kayaking through the mangroves. 4-day active
                    trip, need equipment rental and lessons. Interested in visiting
                    Jan Thiel Beach and evening BBQ at Caracasbaai.""",
            "preferences": {
                "interests": ["water_sports", "nature", "adventure"],
                "locations": ["Spanish Water", "Jan Thiel", "Caracasbaai"],
                "trip_duration": 4,
                "group_size": 2,
                "activity_types": ["windsurfing", "kayaking", "swimming"],
                "equipment_needs": ["water_sports_gear"],
                "skill_level": "beginner",
                "specific_sites": ["mangroves", "Jan Thiel Beach"]
            }
        },

        # 7. Banda Abou Historical Tour
        {
            "query": """Planning to explore the historical plantation houses in Banda
                    Abou region. Want to visit Landhuis Ascension, Landhuis Knip, and
                    other restored mansions. Interested in learning about colonial
                    history and visiting small villages like Soto and Barber.""",
            "preferences": {
                "interests": ["history", "architecture", "culture"],
                "locations": ["Banda Abou"],
                "trip_duration": 3,
                "group_size": 4,
                "activity_types": ["historical_tours", "photography"],
                "specific_sites": ["Landhuis Ascension", "Landhuis Knip", "Soto", "Barber"],
                "guide_requirements": "historical_knowledge",
                "transportation_needs": True
            }
        },

        # 8. Otrobanda Market & Craft Experience
        {
            "query": """Want to explore local crafts in Otrobanda, visit the Old Market,
                    and take workshops in Chichi doll painting. Interested in shopping
                    for local art and visiting galleries. Also want to see the Kura
                    Hulanda Museum complex.""",
            "preferences": {
                "interests": ["crafts", "art", "culture", "shopping"],
                "locations": ["Otrobanda"],
                "trip_duration": 2,
                "group_size": 1,
                "activity_types": ["shopping", "workshops", "museum_visits"],
                "specific_sites": ["Old Market", "Kura Hulanda Museum"],
                "special_interests": ["local_crafts", "art_galleries"]
            }
        },

        # 9. Shete Boka & North Coast Nature
        {
            "query": """Want to explore Shete Boka National Park and the north coast.
                    Interested in seeing Boka Tabla, hiking the coastal trails, and
                    watching waves crash into caves. Also want to visit nearby beaches
                    like Playa Gipy and see desert landscape.""",
            "preferences": {
                "interests": ["nature", "hiking", "photography"],
                "locations": ["Shete Boka", "North Coast"],
                "trip_duration": 2,
                "group_size": 2,
                "activity_types": ["hiking", "sightseeing", "photography"],
                "specific_sites": ["Boka Tabla", "Playa Gipy"],
                "transportation_needs": True,
                "time_preferences": ["morning"]
            }
        },

        # 10. Piscadera Bay Snorkeling
        {
            "query": """Looking for snorkeling spots around Piscadera Bay. Want to
                    stay near the Curaçao Sea Aquarium and do both shore and boat
                    snorkeling. Interested in night snorkeling and seeing the coral
                    restoration project. 5-day trip focused on marine life.""",
            "preferences": {
                "interests": ["snorkeling", "marine_life", "nature"],
                "locations": ["Piscadera Bay"],
                "trip_duration": 5,
                "group_size": 2,
                "activity_types": ["snorkeling", "boat_trips", "aquarium_visit"],
                "specific_sites": ["Sea Aquarium", "coral_restoration"],
                "equipment_needs": ["snorkel_gear"],
                "special_interests": ["marine_conservation"]
            }
        }
    ]


    for i, case in enumerate(test_cases):
        query = case["query"]
        preferences = case["preferences"]
        recommendations = recommend(query, preferences)

        # Add your expected conditions here
        assert recommendations is not None, f"Test case {i} failed: No recommendations"
        assert isinstance(recommendations, list), f"Test case {i} failed: Recommendations are not a list"

        # Add specific validation based on your system's behavior
        print(f"Test case {i} passed with recommendations: {recommendations}")
