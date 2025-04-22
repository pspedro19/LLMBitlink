"""Communication module for coordinating MCP servers."""

import asyncio
import logging
import os
import sys
from typing import Dict, Any

# Configure logging
logger = logging.getLogger("communication")

class ServerCoordinator:
    """Coordinates communication between MCP servers."""
    
    def __init__(self):
        """Initialize the coordinator."""
        self.sessions = {}
        self.tools_cache = {}  # Cache of available tools per server
    
    async def initialize(self, server_sessions):
        """Initialize with existing server sessions."""
        self.sessions = server_sessions
        
        # Cache available tools
        for server_name, session in self.sessions.items():
            try:
                tools = await session.list_tools()
                self.tools_cache[server_name] = [tool.name for tool in tools]
                logger.info(f"Cached {len(tools)} tools for {server_name}")
            except Exception as e:
                logger.error(f"Failed to cache tools for {server_name}: {e}")
        
        return True
    
    async def call_server_tool(self, server_name, tool_name, params=None, max_retries=3):
        """Call a tool on a server with retry logic."""
        if server_name not in self.sessions:
            raise ValueError(f"Server {server_name} not found")
        
        session = self.sessions[server_name]
        
        # Implement retry logic
        retries = 0
        while retries <= max_retries:
            try:
                result = await session.call_tool(tool_name, params or {})
                return result
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    logger.error(f"Failed to call {tool_name} on {server_name} after {max_retries} retries: {e}")
                    raise
                
                # Exponential backoff
                wait_time = 0.5 * (2 ** retries)
                logger.warning(f"Retry {retries}/{max_retries} for {tool_name} on {server_name}. Waiting {wait_time}s")
                await asyncio.sleep(wait_time)
    
    async def process_user_message(self, message, session_id):
        """
        Process a user message through the appropriate servers.
        
        Args:
            message: User's message text
            session_id: User's session ID
            
        Returns:
            Generated response
        """
        try:
            # 1. Get current user context
            user_context = await self.call_server_tool(
                "orchestrator", "get_user_context", {"session_id": session_id}
            )
            
            # 2. Extract preferences
            previous_preferences = user_context.get("preferences", {})
            preferences = await self.call_server_tool(
                "orchestrator", "extract_preferences_from_message", {
                    "message": message,
                    "previous_preferences": previous_preferences
                }
            )
            
            # 3. Update user preferences
            if preferences:
                await self.call_server_tool(
                    "orchestrator", "update_user_preferences", 
                    {"session_id": session_id, "preferences": preferences}
                )
            
            # 4. Get updated user context
            user_context = await self.call_server_tool(
                "orchestrator", "get_user_context", {"session_id": session_id}
            )
            
            # 5. Check for special requests (like itineraries)
            special_request = None
            for keyword in ["itinerario", "itinerary", "plan", "agenda"]:
                if keyword in message.lower():
                    special_request = "itinerary"
                    break
            
            # 6. Check missing information
            missing_info = await self.call_server_tool(
                "orchestrator", "determine_missing_information", {"session_id": session_id}
            )
            
            # 7. Generate recommendations if we have all required info
            recommendations = []
            rag_info = {}
            
            if not missing_info:
                # Have enough info to generate recommendations
                recommendations = await self.call_server_tool(
                    "excel", "get_recommendations", {
                        "interests": user_context.get("preferences", {}).get("interests", []),
                        "budget": user_context.get("preferences", {}).get("budget"),
                        "duration": user_context.get("preferences", {}).get("duration"),
                        "locations": user_context.get("preferences", {}).get("locations", ["Curaçao"]),
                        "limit": 5
                    }
                )
                
                # 8. Enrich with RAG if we have recommendations
                if recommendations:
                    place_names = [rec.get("name", "") for rec in recommendations[:2]]
                    if place_names:
                        query = f"Información sobre {', '.join(place_names)} en Curaçao"
                        rag_results = await self.call_server_tool(
                            "rag", "search_knowledge", {"query": query, "top_k": 3}
                        )
                        
                        enriched_recommendations = await self.call_server_tool(
                            "orchestrator", "enrich_recommendations", {
                                "recommendations": recommendations,
                                "rag_results": rag_results
                            }
                        )
                        
                        recommendations = enriched_recommendations
                        rag_info = rag_results
            
            # 9. Add message to history
            await self.call_server_tool(
                "orchestrator", "add_message_to_history", {
                    "session_id": session_id,
                    "sender": "user",
                    "message": message
                }
            )
            
            # 10. Generate response
            if special_request == "itinerary":
                # Handle special itinerary request
                itinerary_data = await self.call_server_tool(
                    "orchestrator", "handle_special_request", {
                        "session_id": session_id,
                        "request_type": "itinerary"
                    }
                )
                
                response_params = {
                    "query": message,
                    "user_context": {
                        **user_context,
                        "special_request": "itinerary",
                        "itinerary_data": itinerary_data
                    },
                    "recommendations": recommendations,
                    "rag_info": rag_info
                }
            else:
                # Normal response
                response_params = {
                    "query": message,
                    "user_context": user_context,
                    "recommendations": recommendations,
                    "rag_info": rag_info
                }
            
            response_data = await self.call_server_tool(
                "response", "generate_response", response_params
            )
            
            # 11. Add response to history
            await self.call_server_tool(
                "orchestrator", "add_message_to_history", {
                    "session_id": session_id,
                    "sender": "assistant",
                    "message": response_data.get("response", "")
                }
            )
            
            # 12. Store recommendations if we have them
            if recommendations:
                await self.call_server_tool(
                    "orchestrator", "store_recommendations", {
                        "session_id": session_id,
                        "recommendations": recommendations
                    }
                )
            
            return response_data.get("response", "")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"Lo siento, tuve un problema procesando tu mensaje. Error: {str(e)}"