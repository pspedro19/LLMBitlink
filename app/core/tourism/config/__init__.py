import os
import yaml

# Get the current directory path
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

def load_yaml_config(filename):
    """Load YAML configuration file."""
    filepath = os.path.join(CONFIG_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Load all configuration files
CONVERSATION_CONFIG = load_yaml_config('conversation.yaml')
NEURO_CONFIG = load_yaml_config('neuro.yaml')
PREFERENCES_CONFIG = load_yaml_config('preferences.yaml')

# Export configurations
__all__ = ['CONVERSATION_CONFIG', 'NEURO_CONFIG', 'PREFERENCES_CONFIG']