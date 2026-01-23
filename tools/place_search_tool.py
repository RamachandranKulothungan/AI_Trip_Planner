"""Place search tool provider with improved error handling."""
import os
from typing import List
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.tools import BaseTool

from utils.place_info_search import GooglePlaceSearchTool, TavilyPlaceSearchTool
from tools.base_tool_provider import BaseToolProvider
from logger.logging import get_logger
from exceptions.exception_handling import ToolInitializationError, ToolExecutionError

logger = get_logger(__name__)
load_dotenv()


class PlaceSearchTool(BaseToolProvider):
    """Tool provider for place-related searches."""
    
    def _initialize_tools(self) -> None:
        """Initialize place search tools."""
        try:
            self.google_api_key = os.environ.get("GPLACES_API_KEY")
            self.google_places_search = (
                GooglePlaceSearchTool(self.google_api_key) 
                if self.google_api_key else None
            )
            self.tavily_search = TavilyPlaceSearchTool()
            self._tools = self._setup_tools()
        except Exception as e:
            logger.error(f"Failed to initialize PlaceSearchTool: {e}", exc_info=True)
            raise ToolInitializationError(f"Place search tool initialization failed: {e}") from e
    
    def _search_with_fallback(self, primary_search, fallback_search, place: str, search_type: str) -> str:
        """Helper method for search with fallback."""
        try:
            if primary_search:
                result = primary_search(place)
                if result:
                    return f"Following are the {search_type} of {place}:\n{result}"
        except Exception as e:
            logger.warning(f"Primary search failed for {search_type}: {e}")
        
        try:
            result = fallback_search(place)
            return f"Following are the {search_type} of {place} (from alternative source):\n{result}"
        except Exception as e:
            logger.error(f"Fallback search also failed for {search_type}: {e}", exc_info=True)
            raise ToolExecutionError(f"Both search methods failed for {search_type}: {e}") from e
    
    def _setup_tools(self) -> List[BaseTool]:
        """Setup all place search tools."""
        @tool
        def search_attractions(place: str) -> str:
            """Search attractions of a place."""
            if not place:
                return "Place name is required"
            
            try:
                primary = lambda p: self.google_places_search.google_search_attractions(p) if self.google_places_search else None
                fallback = lambda p: self.tavily_search.tavily_search_attractions(p)
                return self._search_with_fallback(primary, fallback, place, "attractions")
            except Exception as e:
                return f"Error searching attractions: {str(e)}"
        
        @tool
        def search_restaurants(place: str) -> str:
            """Search restaurants of a place."""
            if not place:
                return "Place name is required"
            
            try:
                primary = lambda p: self.google_places_search.google_search_restaurants(p) if self.google_places_search else None
                fallback = lambda p: self.tavily_search.tavily_search_attractions(p)
                return self._search_with_fallback(primary, fallback, place, "restaurants")
            except Exception as e:
                return f"Error searching restaurants: {str(e)}"
        
        @tool
        def search_activities(place: str) -> str:
            """Search activities of a place."""
            if not place:
                return "Place name is required"
            
            try:
                primary = lambda p: self.google_places_search.google_search_activity(p) if self.google_places_search else None
                fallback = lambda p: self.tavily_search.tavily_search_activity(p)
                return self._search_with_fallback(primary, fallback, place, "activities")
            except Exception as e:
                return f"Error searching activities: {str(e)}"
        
        @tool
        def search_transportation(place: str) -> str:
            """Search transportation options of a place."""
            if not place:
                return "Place name is required"
            
            try:
                primary = lambda p: self.google_places_search.google_search_transportation(p) if self.google_places_search else None
                fallback = lambda p: self.tavily_search.tavily_search_transportation(p)
                return self._search_with_fallback(primary, fallback, place, "transportation")
            except Exception as e:
                return f"Error searching transportation: {str(e)}"
        
        return [search_attractions, search_restaurants, search_activities, search_transportation]
