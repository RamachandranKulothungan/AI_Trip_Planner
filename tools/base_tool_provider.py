"""Base class for tool providers following Template Method Pattern."""
from abc import ABC, abstractmethod
from typing import List
from langchain_core.tools import BaseTool

from agent.interfaces import IToolProvider
from logger.logging import get_logger

logger = get_logger(__name__)


class BaseToolProvider(ABC, IToolProvider):
    """
    Abstract base class for tool providers.
    
    Follows Template Method Pattern and Single Responsibility Principle.
    """
    
    def __init__(self):
        """Initialize tool provider and load tools."""
        self._tools: List[BaseTool] = []
        self._initialize_tools()
        logger.debug(f"{self.__class__.__name__} initialized with {len(self._tools)} tools")
    
    @property
    def tool_list(self) -> List[BaseTool]:
        """Return list of tools provided by this provider."""
        return self._tools
    
    @abstractmethod
    def _initialize_tools(self) -> None:
        """Initialize tools - to be implemented by subclasses."""
        pass
