"""Expense calculator tool provider."""
from typing import List
from langchain.tools import tool
from langchain_core.tools import BaseTool

from utils.expense_calculator import Calculator
from tools.base_tool_provider import BaseToolProvider
from logger.logging import get_logger

logger = get_logger(__name__)


class ExpenseCalculatorTool(BaseToolProvider):
    """Tool provider for expense calculations."""
    
    def _initialize_tools(self) -> None:
        """Initialize calculator tools."""
        self.calculator = Calculator()
        self._tools = self._setup_tools()
    
    def _setup_tools(self) -> List[BaseTool]:
        """Setup all calculator tools."""
        @tool
        def estimate_total_hotel_cost(price_per_night: str, total_days: float) -> float:
            """
            Calculate total hotel cost.
            
            Args:
                price_per_night: Price per night (can be string or number)
                total_days: Total number of days
                
            Returns:
                Total hotel cost
            """
            try:
                return self.calculator.multiply(price_per_night, total_days)
            except Exception as e:
                logger.error(f"Error calculating hotel cost: {e}", exc_info=True)
                raise ValueError(f"Invalid input for hotel cost calculation: {e}") from e
        
        @tool
        def calculate_total_expense(*costs: float) -> float:
            """
            Calculate total expense of the trip.
            
            Args:
                *costs: Variable number of cost values
                
            Returns:
                Total expense
            """
            try:
                return self.calculator.calculate_total(*costs)
            except Exception as e:
                logger.error(f"Error calculating total expense: {e}", exc_info=True)
                raise ValueError(f"Invalid input for total expense calculation: {e}") from e
        
        @tool
        def calculate_daily_expense_budget(total_cost: float, days: int) -> float:
            """
            Calculate daily expense budget.
            
            Args:
                total_cost: Total cost
                days: Number of days
                
            Returns:
                Daily expense budget
            """
            try:
                return self.calculator.calculate_daily_budget(total_cost, days)
            except Exception as e:
                logger.error(f"Error calculating daily budget: {e}", exc_info=True)
                raise ValueError(f"Invalid input for daily budget calculation: {e}") from e
        
        return [estimate_total_hotel_cost, calculate_total_expense, calculate_daily_expense_budget]
