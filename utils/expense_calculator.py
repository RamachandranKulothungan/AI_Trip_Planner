"""Expense calculator utility with improved type safety."""
from typing import Union


class Calculator:
    """
    Calculator utility for expense calculations.
    
    Follows Single Responsibility Principle: Only handles mathematical calculations.
    """
    
    @staticmethod
    def multiply(a: Union[int, float, str], b: Union[int, float, str]) -> float:
        """
        Multiply two numbers.
        
        Args:
            a: First number (can be string representation)
            b: Second number (can be string representation)
            
        Returns:
            Product of a and b as float
            
        Raises:
            ValueError: If conversion fails
        """
        try:
            return float(a) * float(b)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid input for multiplication: {a}, {b}") from e
    
    @staticmethod
    def calculate_total(*costs: Union[float, int, str]) -> float:
        """
        Calculate sum of the given list of numbers.
        
        Args:
            *costs: Variable number of cost values
            
        Returns:
            Sum of all costs as float
            
        Raises:
            ValueError: If any conversion fails
        """
        try:
            return sum(float(cost) for cost in costs)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid input for total calculation: {costs}") from e
    
    @staticmethod
    def calculate_daily_budget(total: Union[float, int, str], days: Union[int, str]) -> float:
        """
        Calculate daily budget.
        
        Args:
            total: Total cost
            days: Total number of days
            
        Returns:
            Expense for a single day
            
        Raises:
            ValueError: If days is zero or negative, or conversion fails
        """
        try:
            days_int = int(days)
            if days_int <= 0:
                raise ValueError("Number of days must be positive")
            return float(total) / days_int
        except (ValueError, TypeError, ZeroDivisionError) as e:
            raise ValueError(f"Invalid input for daily budget: total={total}, days={days}") from e
