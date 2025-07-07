"""
Advanced Mathematical Tools for the Math Agent
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats, optimize, integrate
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import json
import io
import base64
from langchain.tools import tool
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


@tool
def calculus_tool(expression: str, variable: str = "x", operation: str = "derivative", 
                 additional_params: Optional[Dict] = None) -> str:
    """
    Performs advanced calculus operations including:
    - Derivatives (partial, higher-order)
    - Integrals (definite, indefinite, multiple)  
    - Limits
    - Taylor/Maclaurin series
    - Differential equations
    
    Args:
        expression: Mathematical expression to process
        variable: Variable to use for calculation (default: "x")
        operation: Operation type (derivative, integral, limit, series)
        additional_params: Additional parameters as dict
    """
    try:
        # Parse the expression
        expr = sp.sympify(expression)
        var = sp.Symbol(variable)
        
        if additional_params is None:
            additional_params = {}
        
        result = None
        
        if operation.lower() == "derivative":
            order = additional_params.get("order", 1)
            result = sp.diff(expr, var, order)
            
        elif operation.lower() == "integral":
            if "limits" in additional_params:
                # Definite integral
                limits = additional_params["limits"]
                result = sp.integrate(expr, (var, limits[0], limits[1]))
            else:
                # Indefinite integral
                result = sp.integrate(expr, var)
                
        elif operation.lower() == "limit":
            point = additional_params.get("point", 0)
            direction = additional_params.get("direction", "+-")
            result = sp.limit(expr, var, point, direction)
            
        elif operation.lower() == "series":
            point = additional_params.get("point", 0)
            order = additional_params.get("order", 6)
            result = sp.series(expr, var, point, order)
            
        elif operation.lower() == "solve_ode":
            # Simple ODE solver
            func = sp.Function(additional_params.get("function", "y"))
            ode_expr = expr.subs(sp.Symbol(additional_params.get("function", "y")), func(var))
            result = sp.dsolve(ode_expr, func(var))
        
        if result is not None:
            # Format the result
            latex_result = sp.latex(result)
            simplified = sp.simplify(result)
            
            return json.dumps({
                "operation": operation,
                "expression": expression,
                "result": str(result),
                "simplified": str(simplified),
                "latex": latex_result,
                "numerical_value": _try_numerical_evaluation(result, var)
            })
        else:
            return f"Unknown operation: {operation}"
            
    except Exception as e:
        logger.error(f"Calculus tool error: {e}")
        return f"Error in calculus operation: {str(e)}"


def _try_numerical_evaluation(expr, var) -> Optional[float]:
    """Try to evaluate expression numerically at a sample point."""
    try:
        if var in expr.free_symbols:
            # Evaluate at x=1 as sample
            return float(expr.subs(var, 1))
        else:
            return float(expr)
    except:
        return None


@tool  
def algebra_tool(expression: str, operation: str, variables: Optional[List[str]] = None) -> str:
    """
    Performs algebraic operations including:
    - Equation solving (linear, quadratic, polynomial, systems)
    - Factorization
    - Expansion
    - Simplification
    - Matrix operations
    
    Args:
        expression: Mathematical expression or equation
        operation: Operation type (solve, factor, expand, simplify)
        variables: Variables to solve for (optional)
    """
    try:
        if operation.lower() == "solve":
            return _solve_equation(expression, variables)
        elif operation.lower() == "factor":
            expr = sp.sympify(expression)
            result = sp.factor(expr)
            return json.dumps({
                "operation": "factor",
                "original": expression,
                "result": str(result),
                "latex": sp.latex(result)
            })
        elif operation.lower() == "expand":
            expr = sp.sympify(expression)
            result = sp.expand(expr)
            return json.dumps({
                "operation": "expand",
                "original": expression,
                "result": str(result),
                "latex": sp.latex(result)
            })
        elif operation.lower() == "simplify":
            expr = sp.sympify(expression)
            result = sp.simplify(expr)
            return json.dumps({
                "operation": "simplify",
                "original": expression,
                "result": str(result),
                "latex": sp.latex(result)
            })
        else:
            return f"Unknown operation: {operation}"
            
    except Exception as e:
        logger.error(f"Algebra tool error: {e}")
        return f"Error in algebra operation: {str(e)}"


def _solve_equation(expression: str, variables: Optional[List[str]] = None) -> str:
    """Solve equations or systems of equations."""
    try:
        # Handle system of equations (comma-separated)
        if ',' in expression:
            equations = [sp.sympify(eq.strip()) for eq in expression.split(',')]
            if variables:
                vars_symbols = [sp.Symbol(var) for var in variables]
            else:
                # Auto-detect variables
                all_symbols = set()
                for eq in equations:
                    all_symbols.update(eq.free_symbols)
                vars_symbols = list(all_symbols)
            
            solutions = sp.solve(equations, vars_symbols)
        else:
            # Single equation
            expr = sp.sympify(expression)
            if variables:
                var_symbol = sp.Symbol(variables[0])
            else:
                # Auto-detect variable
                free_symbols = expr.free_symbols
                var_symbol = list(free_symbols)[0] if free_symbols else sp.Symbol('x')
            
            solutions = sp.solve(expr, var_symbol)
        
        return json.dumps({
            "operation": "solve",
            "expression": expression,
            "solutions": [str(sol) for sol in solutions] if isinstance(solutions, list) else str(solutions),
            "latex_solutions": [sp.latex(sol) for sol in solutions] if isinstance(solutions, list) else sp.latex(solutions)
        })
        
    except Exception as e:
        return f"Error solving equation: {str(e)}"


@tool
def statistics_tool(data: Optional[List[float]] = None, operation: str = "descriptive", 
                   parameters: Optional[Dict] = None) -> str:
    """
    Performs statistical analysis including:
    - Descriptive statistics
    - Probability distributions
    - Hypothesis testing
    - Regression analysis
    - Correlation analysis
    
    Args:
        data: Numerical data (optional)
        operation: Operation type (descriptive, probability, hypothesis_test, distribution)
        parameters: Additional parameters as dict
    """
    try:
        if parameters is None:
            parameters = {}
        
        if operation.lower() == "descriptive" and data:
            return _descriptive_stats(data)
        elif operation.lower() == "probability":
            return _probability_calculation(parameters)
        elif operation.lower() == "hypothesis_test" and data:
            return _hypothesis_test(data, parameters)
        elif operation.lower() == "distribution":
            return _distribution_analysis(parameters)
        else:
            return f"Unknown operation or missing data: {operation}"
            
    except Exception as e:
        logger.error(f"Statistics tool error: {e}")
        return f"Error in statistics operation: {str(e)}"


def _descriptive_stats(data: List[float]) -> str:
    """Calculate descriptive statistics."""
    arr = np.array(data)
    
    stats_dict = {
        "count": len(data),
        "mean": np.mean(arr),
        "median": np.median(arr),
        "mode": stats.mode(arr).mode[0] if len(stats.mode(arr).mode) > 0 else None,
        "std_dev": np.std(arr, ddof=1),
        "variance": np.var(arr, ddof=1),
        "min": np.min(arr),
        "max": np.max(arr),
        "range": np.max(arr) - np.min(arr),
        "q1": np.percentile(arr, 25),
        "q3": np.percentile(arr, 75),
        "iqr": np.percentile(arr, 75) - np.percentile(arr, 25),
        "skewness": stats.skew(arr),
        "kurtosis": stats.kurtosis(arr)
    }
    
    return json.dumps(stats_dict)


def _probability_calculation(parameters: Dict) -> str:
    """Calculate probability distributions."""
    dist_type = parameters.get("distribution", "normal")
    
    if dist_type == "normal":
        mean = parameters.get("mean", 0)
        std = parameters.get("std", 1)
        x = parameters.get("x", 0)
        
        pdf = stats.norm.pdf(x, mean, std)
        cdf = stats.norm.cdf(x, mean, std)
        
        return json.dumps({
            "distribution": "normal",
            "parameters": {"mean": mean, "std": std},
            "x": x,
            "pdf": pdf,
            "cdf": cdf
        })
    
    # Add more distributions as needed
    return json.dumps({"error": f"Distribution {dist_type} not implemented"})


def _hypothesis_test(data: List[float], parameters: Dict) -> str:
    """Perform hypothesis tests."""
    test_type = parameters.get("test", "ttest")
    
    if test_type == "ttest":
        mu0 = parameters.get("mu0", 0)
        statistic, p_value = stats.ttest_1samp(data, mu0)
        
        return json.dumps({
            "test": "one_sample_ttest",
            "statistic": statistic,
            "p_value": p_value,
            "alpha": parameters.get("alpha", 0.05),
            "reject_null": p_value < parameters.get("alpha", 0.05)
        })
    
    return json.dumps({"error": f"Test {test_type} not implemented"})


def _distribution_analysis(parameters: Dict) -> str:
    """Analyze probability distributions."""
    # Implementation for distribution fitting, etc.
    return json.dumps({"message": "Distribution analysis not yet implemented"})


@tool
def visualization_tool(expression: Optional[str] = None, data: Optional[Dict] = None,
                      plot_type: str = "function", parameters: Optional[Dict] = None) -> str:
    """
    Creates mathematical visualizations including:
    - Function plots
    - Statistical plots
    - 3D visualizations
    - Interactive plots
    
    Args:
        expression: Mathematical expression to plot (optional)
        data: Data to visualize (optional)
        plot_type: Type of plot (function, scatter, histogram, etc.)
        parameters: Plot parameters as dict
    """
    try:
        if parameters is None:
            parameters = {}
        
        if plot_type == "function" and expression:
            return _plot_function(expression, parameters)
        elif plot_type == "scatter" and data:
            return _plot_scatter(data, parameters)
        elif plot_type == "histogram" and data:
            return _plot_histogram(data, parameters)
        else:
            return f"Unknown plot type or missing data: {plot_type}"
            
    except Exception as e:
        logger.error(f"Visualization tool error: {e}")
        return f"Error in visualization: {str(e)}"


def _plot_function(expression: str, parameters: Dict) -> str:
    """Plot mathematical functions."""
    try:
        # Parse expression
        x = sp.Symbol('x')
        expr = sp.sympify(expression)
        
        # Create numerical function
        func = sp.lambdify(x, expr, 'numpy')
        
        # Generate points
        x_min = parameters.get("x_min", -10)
        x_max = parameters.get("x_max", 10)
        num_points = parameters.get("num_points", 1000)
        
        x_vals = np.linspace(x_min, x_max, num_points)
        y_vals = func(x_vals)
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=f'f(x) = {expression}'))
        fig.update_layout(
            title=f'Plot of f(x) = {expression}',
            xaxis_title='x',
            yaxis_title='f(x)',
            template='plotly_white'
        )
        
        # Save as HTML string
        html_str = fig.to_html()
        
        return json.dumps({
            "plot_type": "function",
            "expression": expression,
            "html": html_str,
            "message": "Function plotted successfully"
        })
        
    except Exception as e:
        return f"Error plotting function: {str(e)}"


def _plot_scatter(data: Dict, parameters: Dict) -> str:
    """Create scatter plots."""
    # Implementation for scatter plots
    return json.dumps({"message": "Scatter plot not yet implemented"})


def _plot_histogram(data: Dict, parameters: Dict) -> str:
    """Create histograms."""
    # Implementation for histograms
    return json.dumps({"message": "Histogram not yet implemented"})


# Export all tools
def get_math_tools() -> List:
    """Get all mathematical tools."""
    return [
        calculus_tool,
        algebra_tool,
        statistics_tool,
        visualization_tool
    ]
