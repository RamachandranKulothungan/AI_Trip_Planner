"""Interfaces and abstractions following Dependency Inversion Principle."""
from abc import ABC, abstractmethod
from typing import List, Protocol, Any, TYPE_CHECKING
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph as CompiledGraph
else:
    # Use Any for runtime to avoid import issues
    CompiledGraph = Any


class IToolProvider(Protocol):
    """Protocol for tool providers following Interface Segregation Principle."""
    
    @property
    @abstractmethod
    def tool_list(self) -> List[BaseTool]:
        """Return list of tools provided by this provider."""
        pass


class IModelLoader(ABC):
    """Abstract interface for model loading following Dependency Inversion Principle."""
    
    @abstractmethod
    def load_llm(self) -> BaseChatModel:
        """Load and return the LLM model."""
        pass
    
    @abstractmethod
    def get_model_provider(self) -> str:
        """Return the model provider name."""
        pass


class IGraphBuilder(ABC):
    """Abstract interface for graph building."""
    
    @abstractmethod
    def build_graph(self) -> CompiledGraph:
        """Build and compile the LangGraph graph."""
        pass
    
    @abstractmethod
    def get_graph(self) -> CompiledGraph:
        """Get the compiled graph instance."""
        pass
