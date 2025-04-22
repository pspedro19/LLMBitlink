"""Session persistence module for MCP servers."""

import json
import os
import time
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger("persistence")

class SessionStore:
    """Store and retrieve user sessions."""
    
    def __init__(self, storage_dir=None):
        """
        Initialize the session store.
        
        Args:
            storage_dir: Directory to store session data (default: ~/.curacao_assistant)
        """
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            self.storage_dir = Path.home() / ".curacao_assistant"
        
        # Create directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using session storage directory: {self.storage_dir}")
    
    def save_session(self, session_id, session_data):
        """
        Save a session to disk.
        
        Args:
            session_id: Unique session identifier
            session_data: Session data to save
        
        Returns:
            bool: True if successful
        """
        try:
            # Make sure it's serializable
            json_data = {
                "id": session_id,
                "last_saved": time.time(),
                "data": session_data
            }
            
            file_path = self.storage_dir / f"{session_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            return False
    
    def load_session(self, session_id):
        """
        Load a session from disk.
        
        Args:
            session_id: Unique session identifier
        
        Returns:
            dict: Session data or None if not found
        """
        try:
            file_path = self.storage_dir / f"{session_id}.json"
            if not file_path.exists():
                logger.info(f"Session {session_id} not found")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            logger.info(f"Loaded session {session_id}")
            return json_data.get("data")
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def list_sessions(self):
        """
        List all saved sessions.
        
        Returns:
            list: List of session IDs
        """
        try:
            session_files = list(self.storage_dir.glob("*.json"))
            return [f.stem for f in session_files]
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
    
    def delete_session(self, session_id):
        """
        Delete a session.
        
        Args:
            session_id: Unique session identifier
        
        Returns:
            bool: True if successful
        """
        try:
            file_path = self.storage_dir / f"{session_id}.json"
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted session {session_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False