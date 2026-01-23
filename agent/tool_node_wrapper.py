"""Wrapped tool node with execution tracking."""
from typing import Dict, Any, List
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from logger.logging import get_logger
from utils.execution_tracer import ExecutionTracer

logger = get_logger(__name__)


class TrackedToolNode:
    """Wraps ToolNode to track tool executions."""
    
    def __init__(self, tools: List[BaseTool], tracer: ExecutionTracer):
        """
        Initialize tracked tool node.
        
        Args:
            tools: List of tools to execute
            tracer: Execution tracer instance
        """
        self.tool_node = ToolNode(tools=tools)
        self.tracer = tracer
        self.tools_map = {tool.name: tool for tool in tools}
    
    def execute(self, state: MessagesState) -> Dict[str, Any]:
        """
        Execute tools with tracking.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with tool results
        """
        try:
            messages = state.get("messages", [])
            last_message = messages[-1]
            
            if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
                logger.warning("TrackedToolNode: No tool calls found")
                return {}
            
            logger.info(f"TrackedToolNode: Executing {len(last_message.tool_calls)} tool calls")
            
            # Execute tools through standard ToolNode
            result = self.tool_node.invoke(state)
            
            # Track tool executions
            if "messages" in result:
                for message in result["messages"]:
                    if isinstance(message, ToolMessage):
                        self.tracer.record_tool_execution(message)
                        # Find tool name for logging
                        tool_name = "unknown"
                        for tc in last_message.tool_calls:
                            if tc.get("id") == message.tool_call_id:
                                tool_name = tc.get("name", "unknown")
                                break
                        logger.info(f"Tool execution completed: {tool_name} ({message.tool_call_id}) -> "
                                  f"{str(message.content)[:100]}...")
            
            return result
        except Exception as e:
            logger.error(f"TrackedToolNode error: {e}", exc_info=True)
            # Record error for each tool call that failed
            if hasattr(last_message, "tool_calls"):
                for tool_call in last_message.tool_calls:
                    self.tracer.record_tool_error(tool_call.get("id", "unknown"), str(e))
            raise
