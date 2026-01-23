"""Weather information tool provider."""
import os
from typing import List
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.tools import BaseTool

from utils.weather_info import WeatherForecastTool
from tools.base_tool_provider import BaseToolProvider
from logger.logging import get_logger
from exceptions.exception_handling import ToolInitializationError

logger = get_logger(__name__)
load_dotenv()


class WeatherInfoTool(BaseToolProvider):
    """
    Tool provider for weather information.
    
    Follows Single Responsibility Principle: Only handles weather-related tools.
    """
    
    def _initialize_tools(self) -> None:
        """Initialize weather tools."""
        try:
            api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
            if not api_key:
                logger.warning("OPENWEATHERMAP_API_KEY not found, weather tools may not work")
            
            self.weather_service = WeatherForecastTool(api_key) if api_key else None
            self._tools = self._setup_tools()
        except Exception as e:
            logger.error(f"Failed to initialize WeatherInfoTool: {e}", exc_info=True)
            raise ToolInitializationError(f"Weather tool initialization failed: {e}") from e
    
    def _setup_tools(self) -> List[BaseTool]:
        """Setup all weather-related tools."""
        @tool
        def get_current_weather(city: str) -> str:
            """
            Get current weather for a city.
            
            Args:
                city: Name of the city to get weather for
                
            Returns:
                Formatted string with current weather information
            """
            if not self.weather_service:
                return f"Weather service unavailable. API key not configured."
            
            try:
                weather_data = self.weather_service.get_current_weather(city)
                if weather_data and weather_data.get('main'):
                    temp = weather_data.get('main', {}).get('temp', 'N/A')
                    desc = weather_data.get('weather', [{}])[0].get('description', 'N/A')
                    humidity = weather_data.get('main', {}).get('humidity', 'N/A')
                    return (
                        f"Current weather in {city}:\n"
                        f"Temperature: {temp}°C\n"
                        f"Description: {desc}\n"
                        f"Humidity: {humidity}%"
                    )
                return f"Could not fetch weather data for {city}"
            except Exception as e:
                logger.error(f"Error fetching current weather for {city}: {e}", exc_info=True)
                return f"Error fetching weather for {city}: {str(e)}"
        
        @tool
        def get_weather_forecast(city: str, days: int = 5) -> str:
            """
            Get weather forecast for a city.
            
            Args:
                city: Name of the city
                days: Number of days to forecast (default: 5, max: 10)
                
            Returns:
                Formatted string with weather forecast
            """
            if not self.weather_service:
                return f"Weather service unavailable. API key not configured."
            
            try:
                days = min(max(days, 1), 10)  # Clamp between 1 and 10
                forecast_data = self.weather_service.get_forecast_weather(city)
                
                if forecast_data and 'list' in forecast_data:
                    forecast_summary = []
                    items_to_show = min(len(forecast_data['list']), days)
                    
                    for i in range(items_to_show):
                        item = forecast_data['list'][i]
                        date = item.get('dt_txt', '').split(' ')[0]
                        temp = item.get('main', {}).get('temp', 'N/A')
                        desc = item.get('weather', [{}])[0].get('description', 'N/A')
                        forecast_summary.append(f"{date}: {temp}°C, {desc}")
                    
                    return f"Weather forecast for {city} (next {items_to_show} periods):\n" + "\n".join(forecast_summary)
                return f"Could not fetch forecast for {city}"
            except Exception as e:
                logger.error(f"Error fetching forecast for {city}: {e}", exc_info=True)
                return f"Error fetching forecast for {city}: {str(e)}"
        
        return [get_current_weather, get_weather_forecast]
