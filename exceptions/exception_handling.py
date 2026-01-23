"""Custom exceptions following exception hierarchy best practices."""
from typing import Optional


class TravelPlannerBaseException(Exception):
    """Base exception for all travel planner exceptions."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ModelLoadError(TravelPlannerBaseException):
    """Raised when model loading fails."""
    pass


class GraphBuildError(TravelPlannerBaseException):
    """Raised when graph construction fails."""
    pass


class ToolInitializationError(TravelPlannerBaseException):
    """Raised when tool initialization fails."""
    pass


class ToolExecutionError(TravelPlannerBaseException):
    """Raised when tool execution fails."""
    pass


class ConfigurationError(TravelPlannerBaseException):
    """Raised when configuration is invalid or missing."""
    pass


class APIError(TravelPlannerBaseException):
    """Raised when external API calls fail."""
    pass
