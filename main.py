"""Refactored FastAPI application with proper error handling and dependency injection."""
import os
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from agent.graph_builder import TravelPlannerGraphBuilder
from agent.interfaces import IModelLoader
from utils.model_loaders import ModelLoader
from tools.weather_info_tool import WeatherInfoTool
from tools.place_search_tool import PlaceSearchTool
from tools.expense_calculator_tool import ExpenseCalculatorTool
from tools.currency_converter_tool import CurrencyConverterTool
from prompt_library.prompt import SYSTEM_PROMPT
from logger.logging import get_logger, LoggerConfig
from exceptions.exception_handling import TravelPlannerBaseException, GraphBuildError

# Initialize logging
logger = get_logger(__name__)

# Load environment variables
load_dotenv(dotenv_path=f"{os.getcwd()}/.env")

# Global graph instance (singleton pattern for performance)
_graph_instance: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    LoggerConfig().configure(
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE")
    )
    logger.info("Application starting up")
    yield
    # Shutdown
    logger.info("Application shutting down")
    _graph_instance.clear()


app = FastAPI(
    title="AI Trip Planner API",
    description="Agentic travel planning application using LangGraph",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration - should be from config in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Request model for travel queries."""
    question: str = Field(..., min_length=1, max_length=2000, description="User travel query")


class QueryResponse(BaseModel):
    """Response model for travel queries."""
    answer: str = Field(..., description="AI-generated travel plan")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")


def get_model_provider() -> str:
    """Get model provider from environment or default."""
    return os.getenv("MODEL_PROVIDER", "openrouter")


def create_graph_builder(model_provider: str = Depends(get_model_provider)) -> TravelPlannerGraphBuilder:
    """
    Create or retrieve graph builder instance (singleton pattern).
    
    This prevents rebuilding the graph on every request, improving performance.
    """
    cache_key = f"{model_provider}"
    
    if cache_key not in _graph_instance:
        try:
            logger.info(f"Creating new graph builder for provider: {model_provider}")
            
            # Create dependencies
            model_loader: IModelLoader = ModelLoader(model_provider=model_provider)
            
            # Create tool providers
            tool_providers = [
                WeatherInfoTool(),
                PlaceSearchTool(),
                ExpenseCalculatorTool(),
                CurrencyConverterTool()
            ]
            
            # Build graph
            graph_builder = TravelPlannerGraphBuilder(
                model_loader=model_loader,
                tool_providers=tool_providers,
                system_prompt=SYSTEM_PROMPT
            )
            
            _graph_instance[cache_key] = graph_builder
            logger.info("Graph builder created and cached")
        except Exception as e:
            logger.error(f"Failed to create graph builder: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize graph: {str(e)}"
            )
    
    return _graph_instance[cache_key]


@app.exception_handler(TravelPlannerBaseException)
async def travel_planner_exception_handler(request, exc: TravelPlannerBaseException):
    """Handle custom travel planner exceptions."""
    logger.error(f"Travel planner error: {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.message,
            "details": exc.details,
            "type": exc.__class__.__name__
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "An unexpected error occurred",
            "type": exc.__class__.__name__
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ai-trip-planner"}


@app.post("/query", response_model=QueryResponse)
async def query_travel_agent(
    query: QueryRequest,
    graph_builder: TravelPlannerGraphBuilder = Depends(create_graph_builder)
):
    """
    Process travel planning query with execution tracing.
    
    Args:
        query: User query request
        graph_builder: Graph builder instance (injected)
        
    Returns:
        Travel plan response with execution trace
    """
    from utils.execution_tracer import ExecutionTracer
    from uuid import uuid4
    from langchain_core.messages import AIMessage
    
    execution_id = str(uuid4())
    tracer = ExecutionTracer(query=query.question, execution_id=execution_id)
    
    try:
        logger.info(f"Processing query: {query.question[:100]}... [ID: {execution_id}]")
        
        # Build graph with tracer
        react_app = graph_builder.build_graph(tracer=tracer)
        
        # Prepare input messages
        from langchain_core.messages import HumanMessage
        messages = {"messages": [HumanMessage(content=query.question)]}
        
        # Invoke graph (tracer automatically captures everything)
        output = react_app.invoke(messages)
        
        # Extract final response
        if isinstance(output, dict) and "messages" in output:
            final_message = output["messages"][-1]
            final_output = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            # Record final agent response if it's an AIMessage and hasn't been recorded
            if isinstance(final_message, AIMessage) and not hasattr(final_message, 'tool_calls'):
                # This is the final response without tool calls
                if tracer and not tracer._current_agent_response:
                    tracer.start_agent_step()
                    tracer._current_agent_response.content = final_output
                    tracer.trace.agent_responses.append(tracer._current_agent_response)
        else:
            final_output = str(output)
        
        # Finalize trace
        execution_trace = tracer.finalize(success=True)
        
        logger.info(f"Query processed successfully [ID: {execution_id}]: "
                   f"{execution_trace['total_steps']} steps, "
                   f"{len(execution_trace['tool_executions'])} tool calls")
        
        return QueryResponse(
            answer=final_output,
            metadata={
                "model_provider": graph_builder.model_loader.get_model_provider(),
                "tools_count": len(graph_builder._tools) if hasattr(graph_builder, '_tools') else 0,
                "execution_trace": execution_trace,
                "execution_id": execution_id
            }
        )
        
    except GraphBuildError as e:
        execution_trace = tracer.finalize(success=False, error=str(e))
        logger.error(f"Graph build error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")
    except Exception as e:
        execution_trace = tracer.finalize(success=False, error=str(e))
        logger.error(f"Query processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@app.get("/graph/visualize")
async def visualize_graph(
    graph_builder: TravelPlannerGraphBuilder = Depends(create_graph_builder)
):
    """
    Generate and return graph visualization.
    
    Args:
        graph_builder: Graph builder instance (injected)
        
    Returns:
        Information about graph visualization
    """
    try:
        graph = graph_builder.get_graph()
        png_graph = graph.get_graph().draw_mermaid_png()
        
        output_path = "my_graph.png"
        with open(output_path, "wb") as f:
            f.write(png_graph)
        
        logger.info(f"Graph visualization saved to {output_path}")
        
        return {
            "message": "Graph visualization generated",
            "path": output_path,
            "nodes": len(graph.get_graph().nodes),
            "edges": len(graph.get_graph().edges)
        }
    except Exception as e:
        logger.error(f"Graph visualization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate visualization: {str(e)}")
