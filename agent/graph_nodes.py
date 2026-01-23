"""Graph nodes following Single Responsibility Principle."""
from typing import Dict, Any, Optional
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, AIMessage
from logger.logging import get_logger

logger = get_logger(__name__)


class AgentNode:
    """Handles agent reasoning and tool selection."""
    
    def __init__(self, llm_with_tools, system_prompt: SystemMessage, tracer=None):
        self.llm_with_tools = llm_with_tools
        self.system_prompt = system_prompt
        self.tracer = tracer  # Optional execution tracer
    
    def execute(self, state: MessagesState) -> Dict[str, Any]:
        """
        Execute agent node: process user query and generate response with tool calls.
        
        Args:
            state: Current graph state containing messages
            
        Returns:
            Updated state with agent response
        """
        try:
            logger.info("Agent node: Processing user query")
            
            # Record agent step in tracer
            if self.tracer:
                self.tracer.start_agent_step()
            
            messages = state.get("messages", [])
            
            # Prepend system prompt only if not already present
            input_messages = [self.system_prompt] + messages
            response = self.llm_with_tools.invoke(input_messages)
            
            # Record agent response in tracer
            if self.tracer and isinstance(response, AIMessage):
                self.tracer.record_agent_response(response)
            
            tool_calls_count = len(getattr(response, 'tool_calls', []))
            logger.info(f"Agent node: Generated response with {tool_calls_count} tool calls")
            
            # Log tool calls details
            if tool_calls_count > 0 and self.tracer:
                for tool_call in response.tool_calls:
                    logger.info(f"  - Tool: {tool_call.get('name')} with args: {tool_call.get('args', {})}")
            
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Agent node error: {e}", exc_info=True)
            if self.tracer:
                self.tracer.record_tool_error("agent", str(e))
            raise RuntimeError(f"Agent execution failed: {e}") from e
