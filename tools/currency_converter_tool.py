"""Currency converter tool provider."""
import os
from typing import List
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.tools import BaseTool

from utils.currency_convertor import CurrencyConverter
from tools.base_tool_provider import BaseToolProvider
from logger.logging import get_logger
from exceptions.exception_handling import ToolInitializationError

logger = get_logger(__name__)
load_dotenv()


class CurrencyConverterTool(BaseToolProvider):
    """Tool provider for currency conversion."""
    
    def _initialize_tools(self) -> None:
        """Initialize currency converter tools."""
        try:
            api_key = os.environ.get("EXCHANGE_RATE_API_KEY")
            if not api_key:
                logger.warning("EXCHANGE_RATE_API_KEY not found, currency converter may not work")
            
            self.currency_service = CurrencyConverter(api_key) if api_key else None
            self._tools = self._setup_tools()
        except Exception as e:
            logger.error(f"Failed to initialize CurrencyConverterTool: {e}", exc_info=True)
            raise ToolInitializationError(f"Currency converter initialization failed: {e}") from e
    
    def _setup_tools(self) -> List[BaseTool]:
        """Setup currency converter tools."""
        @tool
        def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
            """
            Convert amount from one currency to another.
            
            Args:
                amount: Amount to convert
                from_currency: Source currency code (e.g., USD)
                to_currency: Target currency code (e.g., EUR)
                
            Returns:
                Converted amount
            """
            if not self.currency_service:
                raise ValueError("Currency service unavailable. API key not configured.")
            
            try:
                return self.currency_service.convert(amount, from_currency, to_currency)
            except Exception as e:
                logger.error(f"Error converting currency: {e}", exc_info=True)
                raise ValueError(f"Currency conversion failed: {e}") from e
        
        return [convert_currency]
