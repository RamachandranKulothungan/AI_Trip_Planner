"""Refactored GraphBuilder following SOLID principles."""
from typing import List, Optional
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from agent.interfaces import IGraphBuilder, IModelLoader, IToolProvider
from agent.graph_nodes import AgentNode
from logger.logging import get_logger
from exceptions.exception_handling import GraphBuildError

logger = get_logger(__name__)


class TravelPlannerGraphBuilder(IGraphBuilder):
    """
    Builds and manages the LangGraph graph for travel planning.
    
    Follows Single Responsibility Principle: Only responsible for graph construction.
    Follows Dependency Inversion Principle: Depends on abstractions (IModelLoader, IToolProvider).
    """
    
    def __init__(
        self,
        model_loader: IModelLoader,
        tool_providers: List[IToolProvider],
        system_prompt: SystemMessage,
        max_iterations: int = 15
    ):
        """
        Initialize the graph builder.
        
        Args:
            model_loader: Model loader instance implementing IModelLoader
            tool_providers: List of tool providers implementing IToolProvider
            system_prompt: System prompt for the agent
            max_iterations: Maximum graph execution iterations to prevent infinite loops
        """
        self.model_loader = model_loader
        self.tool_providers = tool_providers
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        
        self._llm: Optional[BaseChatModel] = None
        self._tools: List[BaseTool] = []
        self._agent_node: Optional[AgentNode] = None
        self._graph = None
        
        logger.info(f"GraphBuilder initialized with {len(tool_providers)} tool providers")
    
    def _load_llm(self) -> BaseChatModel:
        """Load LLM model using the model loader."""
        if self._llm is None:
            logger.info("Loading LLM model")
            self._llm = self.model_loader.load_llm()
        return self._llm
    
    def _load_tools(self) -> List[BaseTool]:
        """Aggregate tools from all tool providers."""
        if not self._tools:
            logger.info("Loading tools from providers")
            for provider in self.tool_providers:
                tools = provider.tool_list
                self._tools.extend(tools)
                logger.debug(f"Loaded {len(tools)} tools from {provider.__class__.__name__}")
            logger.info(f"Total tools loaded: {len(self._tools)}")
        return self._tools
    
    def _create_agent_node(self, tracer=None) -> AgentNode:
        """Create the agent node with LLM and tools."""
        if self._agent_node is None or (tracer and self._agent_node.tracer != tracer):
            llm = self._load_llm()
            tools = self._load_tools()
            llm_with_tools = llm.bind_tools(tools=tools)
            self._agent_node = AgentNode(llm_with_tools, self.system_prompt, tracer=tracer)
        elif tracer:
            # Update tracer if it changed
            self._agent_node.tracer = tracer
        return self._agent_node
    
    def build_graph(self, tracer=None):
        """
        Build and compile the LangGraph graph.
        
        Graph Structure:
        - START -> agent
        - agent -> (conditional) -> tools or END
        - tools -> agent
        
        Args:
            tracer: Optional execution tracer for tracking
        
        Returns:
            Compiled LangGraph instance
            
        Raises:
            GraphBuildError: If graph construction fails
        """
        try:
            logger.info("Building LangGraph structure" + (" with tracing" if tracer else ""))
            
            # Create nodes
            agent_node = self._create_agent_node(tracer=tracer)
            tools = self._load_tools()
            
            # Build graph
            graph_builder = StateGraph(MessagesState)
            
            # Add nodes - use TrackedToolNode if tracer is provided
            graph_builder.add_node("agent", agent_node.execute)
            
            if tracer:
                from agent.tool_node_wrapper import TrackedToolNode
                tool_node = TrackedToolNode(tools=tools, tracer=tracer)
                graph_builder.add_node("tools", tool_node.execute)
            else:
                graph_builder.add_node("tools", ToolNode(tools=tools))
            
            # Add edges
            graph_builder.add_edge(START, "agent")
            graph_builder.add_conditional_edges(
                "agent",
                tools_condition,
                {
                    "tools": "tools",
                    "__end__": END
                }
            )
            graph_builder.add_edge("tools", "agent")
            
            # Compile with safety checks
            self._graph = graph_builder.compile()
            
            logger.info("Graph built and compiled successfully")
            return self._graph
            
        except Exception as e:
            logger.error(f"Graph build failed: {e}", exc_info=True)
            raise GraphBuildError(f"Failed to build graph: {e}") from e
    
    def get_graph(self):
        """
        Get the compiled graph, building it if necessary.
        
        Returns:
            Compiled graph instance
        """
        if self._graph is None:
            return self.build_graph()
        return self._graph
