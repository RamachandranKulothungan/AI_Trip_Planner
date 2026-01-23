"""Refactored ModelLoader following SOLID principles."""
import os
from typing import Literal, Optional, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from utils.config_loader import ConfigLoader
from agent.interfaces import IModelLoader
from logger.logging import get_logger
from exceptions.exception_handling import ModelLoadError

logger = get_logger(__name__)
load_dotenv()

# OpenRouter API endpoint
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


class ModelLoader(BaseModel, IModelLoader):
    """
    Loads LLM models from various providers.
    
    Follows Single Responsibility Principle: Only responsible for model loading.
    Follows Open/Closed Principle: Can be extended for new providers without modification.
    """
    
    model_provider: Literal["groq", "openai", "openrouter"] = Field(
        default="openrouter", 
        description="Model provider name"
    )
    config: Optional[ConfigLoader] = Field(default=None, exclude=True)
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize config loader after model initialization."""
        if self.config is None:
            self.config = ConfigLoader()
    
    class Config:
        arbitrary_types_allowed = True
    
    def get_model_provider(self) -> str:
        """Return the model provider name."""
        return self.model_provider
    
    def load_llm(self) -> BaseChatModel:
        """
        Load and return the LLM model based on provider.
        
        Returns:
            Initialized LLM model instance
            
        Raises:
            ModelLoadError: If model loading fails or API key is missing
        """
        logger.info(f"Loading LLM from provider: {self.model_provider}")
        
        try:
            if self.model_provider == "groq":
                return self._load_groq_model()
            elif self.model_provider == "openai":
                return self._load_openai_model()
            elif self.model_provider == "openrouter":
                return self._load_openrouter_model()
            else:
                raise ValueError(f"Unsupported model provider: {self.model_provider}")
        except Exception as e:
            logger.error(f"Model loading failed: {e}", exc_info=True)
            raise ModelLoadError(f"Failed to load model from {self.model_provider}: {e}") from e
    
    def _load_groq_model(self) -> ChatGroq:
        """Load Groq model."""
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ModelLoadError("GROQ_API_KEY environment variable is not set")
        
        model_name = self.config["llm"]["groq"]["model_name"]
        logger.debug(f"Loading Groq model: {model_name}")
        
        return ChatGroq(model=model_name, api_key=groq_api_key)
    
    def _load_openai_model(self) -> ChatOpenAI:
        """Load OpenAI model."""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ModelLoadError("OPENAI_API_KEY environment variable is not set")
        
        # Fixed: Use config instead of hard-coded model name
        model_name = self.config["llm"]["openai"]["model_name"]
        logger.debug(f"Loading OpenAI model: {model_name}")
        
        return ChatOpenAI(model=model_name, api_key=openai_api_key)
    
    def _load_openrouter_model(self) -> ChatOpenAI:
        """
        Load model from OpenRouter.
        
        OpenRouter provides access to multiple LLM models through a unified API.
        Uses OpenAI-compatible API with custom base URL.
        """
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ModelLoadError("OPENROUTER_API_KEY environment variable is not set")
        
        # Get model name from config
        model_name = self.config["llm"]["openrouter"]["model_name"]
        logger.debug(f"Loading OpenRouter model: {model_name}")
        
        # OpenRouter uses OpenAI-compatible API
        # We need to set the base_url and add the HTTP Referer header
        model_config = self.config["llm"]["openrouter"]
        
        # Create ChatOpenAI instance with OpenRouter configuration
        chat_model = ChatOpenAI(
            model=model_name,
            api_key=openrouter_api_key,
            base_url=OPENROUTER_API_BASE,
            default_headers={
                "HTTP-Referer": model_config.get("http_referer", ""),  # Optional: Your site URL
                "X-Title": model_config.get("app_name", "AI Trip Planner"),  # Optional: App name
            },
            temperature=model_config.get("temperature", 0.7),
            max_tokens=model_config.get("max_tokens", None),
        )
        
        logger.info(f"Successfully loaded OpenRouter model: {model_name}")
        return chat_model