# app/tests/integration/test_components.py

from pathlib import Path
from core.analyzer import ImprovedNLPProcessor
from core.data import CSVDatabaseManager
from core.recommender import IntegratedTourismSystem

# Get path to actual data files
DATA_PATH = Path(__file__).parents[2] / "data" / "database"

def test_nlp_processor():
    nlp = ImprovedNLPProcessor()
    
    test_query = """Want to explore the UNESCO World Heritage sites in Punda and
                    Otrobanda, focusing on the Dutch architecture."""
    
    extracted_prefs = nlp.extract_preferences(test_query)
    print("Extracted preferences:", extracted_prefs)

def test_database():
    db = CSVDatabaseManager()
    # Using actual table name from your data
    query = "SELECT * FROM activities1 LIMIT 5"
    results = db.execute_query(query)
    print("Database query results:", results)

def test_tourism_system():
    system = IntegratedTourismSystem()
    preferences = {
        "interests": ["cultural", "history"],
        "locations": ["Punda"],
        "budget_per_day": 200
    }
    
    recommendations = system.get_recommendations(preferences)
    print("Tourism system recommendations:", recommendations)

def main():
    print("Testing NLP Processor...")
    test_nlp_processor()
    
    print("\nTesting Database...")
    test_database()
    
    print("\nTesting Tourism System...")
    test_tourism_system()

if __name__ == "__main__":
    main()