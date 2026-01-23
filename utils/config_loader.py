"""Refactored configuration loader with validation."""
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

from logger.logging import get_logger
from exceptions.exception_handling import ConfigurationError

logger = get_logger(__name__)


class ConfigLoader:
    """
    Loads and validates application configuration.
    
    Follows Single Responsibility Principle: Only handles configuration loading.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to config file. Defaults to config/config.yaml
            
        Raises:
            ConfigurationError: If config file is missing or invalid
        """
        if config_path is None:
            # Use absolute path based on project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {self.config_path}")
        
        try:
            logger.info(f"Loading configuration from {self.config_path}")
            with open(self.config_path, "r", encoding="utf-8") as file:
                self.config = yaml.safe_load(file) or {}
            logger.debug("Configuration loaded successfully")
            self._validate_config()
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config file: {e}") from e
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}") from e
    
    def _validate_config(self) -> None:
        """Validate required configuration sections."""
        required_sections = ["llm"]
        for section in required_sections:
            if section not in self.config:
                raise ConfigurationError(f"Missing required configuration section: {section}")
        
        # Validate LLM providers
        if "llm" in self.config:
            providers = ["groq", "openai", "openrouter"]
            configured_providers = [p for p in providers if p in self.config["llm"]]
            if not configured_providers:
                logger.warning(f"No LLM provider configured. Supported providers: {', '.join(providers)}")
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access to config."""
        if key not in self.config:
            raise ConfigurationError(f"Configuration key not found: {key}")
        return self.config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self.config.get(key, default)
