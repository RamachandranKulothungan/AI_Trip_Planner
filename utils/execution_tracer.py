"""Execution tracer for tracking tool calls and agent flow."""
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage

from logger.logging import get_logger

logger = get_logger(__name__)


class NodeType(str, Enum):
    """Types of graph nodes."""
    AGENT = "agent"
    TOOLS = "tools"
    START = "start"
    END = "end"


@dataclass
class ToolCall:
    """Represents a single tool call."""
    tool_name: str
    tool_args: Dict[str, Any]
    tool_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    execution_time_ms: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AgentResponse:
    """Represents an agent's response."""
    step_number: int
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    node_type: NodeType = NodeType.AGENT

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        result["node_type"] = self.node_type.value
        return result


@dataclass
class ExecutionTrace:
    """Complete execution trace for a query."""
    query: str
    execution_id: str
    start_time: str
    end_time: Optional[str] = None
    total_steps: int = 0
    agent_responses: List[AgentResponse] = field(default_factory=list)
    tool_executions: List[ToolCall] = field(default_factory=list)
    execution_flow: List[Dict[str, Any]] = field(default_factory=list)
    total_execution_time_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None

    def add_step(self, node_type: NodeType, details: Dict[str, Any]):
        """Add a step to execution flow."""
        self.execution_flow.append({
            "step": len(self.execution_flow) + 1,
            "node": node_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            **details
        })
        self.total_steps += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["agent_responses"] = [ar.to_dict() for ar in self.agent_responses]
        result["tool_executions"] = [te.to_dict() for te in self.tool_executions]
        result["execution_flow"] = self.execution_flow
        return result


class ExecutionTracer:
    """Tracks execution flow of graph execution."""
    
    def __init__(self, query: str, execution_id: Optional[str] = None):
        """Initialize tracer."""
        from uuid import uuid4
        self.trace = ExecutionTrace(
            query=query,
            execution_id=execution_id or str(uuid4()),
            start_time=datetime.utcnow().isoformat()
        )
        self._step_counter = 0
        self._current_agent_response: Optional[AgentResponse] = None
    
    def start_agent_step(self) -> None:
        """Mark start of agent step."""
        self._step_counter += 1
        self._current_agent_response = AgentResponse(
            step_number=self._step_counter,
            content="",
            tool_calls=[]
        )
        self.trace.add_step(NodeType.AGENT, {"step_number": self._step_counter})
    
    def record_agent_response(self, message: AIMessage) -> None:
        """Record agent's response."""
        if self._current_agent_response is None:
            self.start_agent_step()
        
        self._current_agent_response.content = message.content or ""
        
        # Extract tool calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_call_obj = ToolCall(
                    tool_name=tool_call.get("name", "unknown"),
                    tool_args=tool_call.get("args", {}),
                    tool_id=tool_call.get("id"),
                )
                self._current_agent_response.tool_calls.append(tool_call_obj)
                self.trace.tool_executions.append(tool_call_obj)
        
        self.trace.agent_responses.append(self._current_agent_response)
        logger.info(f"Recorded agent response with {len(self._current_agent_response.tool_calls)} tool calls")
    
    def record_tool_execution(self, tool_message: ToolMessage) -> None:
        """Record tool execution result."""
        self.trace.add_step(NodeType.TOOLS, {
            "tool_id": tool_message.tool_call_id,
            "content_preview": str(tool_message.content)[:100] if tool_message.content else ""
        })
        
        # Update corresponding tool call with result
        for tool_execution in self.trace.tool_executions:
            if tool_execution.tool_id == tool_message.tool_call_id:
                tool_execution.result = str(tool_message.content)[:500] if tool_message.content else None  # Limit result size
                tool_execution.success = True
                break
    
    def record_tool_error(self, tool_id: str, error: str) -> None:
        """Record tool execution error."""
        for tool_execution in self.trace.tool_executions:
            if tool_execution.tool_id == tool_id:
                tool_execution.error = str(error)
                tool_execution.success = False
                break
    
    def finalize(self, success: bool = True, error: Optional[str] = None) -> Dict[str, Any]:
        """Finalize trace and return summary."""
        self.trace.end_time = datetime.utcnow().isoformat()
        self.trace.success = success
        self.trace.error = error
        
        if self.trace.start_time and self.trace.end_time:
            start = datetime.fromisoformat(self.trace.start_time)
            end = datetime.fromisoformat(self.trace.end_time)
            self.trace.total_execution_time_ms = (end - start).total_seconds() * 1000
        
        logger.info(f"Execution trace finalized: {self.trace.total_steps} steps, "
                   f"{len(self.trace.tool_executions)} tool calls")
        
        return self.trace.to_dict()
