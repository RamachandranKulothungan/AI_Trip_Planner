"""Agent module for LangGraph workflows."""
from agent.graph_builder import TravelPlannerGraphBuilder

# For backward compatibility, keep the old name
GraphBuilder = TravelPlannerGraphBuilder

__all__ = ['TravelPlannerGraphBuilder', 'GraphBuilder']
