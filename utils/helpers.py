"""
Utility functions for the Math Agent
"""
import re
import json
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper())
    
    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=format_string,
        handlers=handlers
    )


def load_environment():
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("Environment variables loaded from .env file")
    except ImportError:
        logger.warning("python-dotenv not installed. Environment variables should be set manually.")
    except Exception as e:
        logger.warning(f"Error loading .env file: {e}")


def validate_api_keys() -> Dict[str, bool]:
    """Validate that required API keys are present."""
    required_keys = ["OPENROUTER_API_KEY"]
    optional_keys = ["WOLFRAM_ALPHA_APP_ID"]
    
    validation_results = {}
    
    for key in required_keys:
        validation_results[key] = bool(os.getenv(key))
        if not validation_results[key]:
            logger.error(f"Required API key missing: {key}")
    
    for key in optional_keys:
        validation_results[key] = bool(os.getenv(key))
        if not validation_results[key]:
            logger.warning(f"Optional API key missing: {key}")
    
    return validation_results


def format_mathematical_expression(expression: str) -> str:
    """Format mathematical expressions for better display."""
    # Replace common mathematical symbols
    replacements = {
        "**": "^",
        "*": "·",
        "sqrt": "√",
        "pi": "π",
        "infinity": "∞",
        "alpha": "α",
        "beta": "β",
        "gamma": "γ",
        "delta": "δ",
        "theta": "θ",
        "lambda": "λ",
        "mu": "μ",
        "sigma": "σ",
        "phi": "φ",
        "omega": "ω"
    }
    
    formatted = expression
    for old, new in replacements.items():
        formatted = formatted.replace(old, new)
    
    return formatted


def extract_mathematical_elements(text: str) -> Dict[str, List[str]]:
    """Extract mathematical elements from text."""
    # Regular expressions for different mathematical elements
    patterns = {
        "equations": r'[a-zA-Z]*\s*=\s*[^,\n]+',
        "expressions": r'[a-zA-Z0-9\+\-\*/\^\(\)\s]+(?:[=<>≤≥≠]+[a-zA-Z0-9\+\-\*/\^\(\)\s]+)*',
        "numbers": r'-?\d+\.?\d*',
        "variables": r'[a-zA-Z]+(?:\([a-zA-Z0-9,\s]*\))?',
        "operators": r'[\+\-\*/\^=<>≤≥≠∫∑∏]',
        "functions": r'(?:sin|cos|tan|log|ln|exp|sqrt|abs|max|min|sum|int)\s*\([^)]+\)'
    }
    
    extracted = {}
    for element_type, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        extracted[element_type] = list(set(matches))  # Remove duplicates
    
    return extracted


def parse_problem_type(problem: str) -> str:
    """Determine the type of mathematical problem."""
    problem_lower = problem.lower()
    
    # Define keywords for different problem types
    problem_types = {
        "calculus": ["derivative", "integral", "limit", "differentiate", "integrate", "series", "taylor"],
        "algebra": ["solve", "equation", "factor", "expand", "simplify", "polynomial", "quadratic"],
        "geometry": ["triangle", "circle", "area", "perimeter", "volume", "angle", "coordinate"],
        "statistics": ["probability", "mean", "median", "standard deviation", "variance", "distribution"],
        "linear_algebra": ["matrix", "vector", "eigenvalue", "determinant", "linear system"],
        "discrete_math": ["combinatorics", "permutation", "combination", "graph", "sequence"],
        "number_theory": ["prime", "gcd", "lcm", "modular", "congruence", "divisibility"],
        "trigonometry": ["sin", "cos", "tan", "sine", "cosine", "tangent", "trigonometric"]
    }
    
    # Count keyword matches for each type
    type_scores = {}
    for problem_type, keywords in problem_types.items():
        score = sum(1 for keyword in keywords if keyword in problem_lower)
        if score > 0:
            type_scores[problem_type] = score
    
    # Return the type with the highest score, or "general" if no clear match
    if type_scores:
        return max(type_scores.items(), key=lambda x: x[1])[0]
    else:
        return "general_math"


def format_solution_output(solution_data: Dict[str, Any]) -> str:
    """Format solution data for display."""
    if not solution_data.get("success", False):
        return f"❌ Error: {solution_data.get('error', 'Unknown error')}"
    
    output_lines = []
    
    # Header
    output_lines.append("🔢 Mathematical Solution")
    output_lines.append("=" * 50)
    
    # Problem
    if "problem" in solution_data:
        output_lines.append(f"📝 Problem: {solution_data['problem']}")
        output_lines.append("")
    
    # Solution
    if "solution" in solution_data:
        output_lines.append("✅ Solution:")
        output_lines.append(solution_data['solution'])
        output_lines.append("")
    
    # Steps (if available)
    if "steps" in solution_data and solution_data["steps"]:
        output_lines.append("📋 Solution Steps:")
        for step in solution_data["steps"]:
            step_num = step.get("step_number", "?")
            action = step.get("action", "")
            output_lines.append(f"  {step_num}. {action}")
        output_lines.append("")
    
    # Metadata
    output_lines.append("ℹ️ Metadata:")
    if "method" in solution_data:
        output_lines.append(f"  Method: {solution_data['method']}")
    if "tools_used" in solution_data and solution_data["tools_used"]:
        output_lines.append(f"  Tools: {', '.join(solution_data['tools_used'])}")
    if "confidence_score" in solution_data:
        confidence = solution_data['confidence_score']
        output_lines.append(f"  Confidence: {confidence:.1%}")
    if "processing_time" in solution_data:
        time_taken = solution_data['processing_time']
        output_lines.append(f"  Time: {time_taken:.2f}s")
    
    return "\\n".join(output_lines)


def create_directories():
    """Create necessary directories for the application."""
    directories = [
        "./data",
        "./logs",
        "./exports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")


def export_solution(solution_data: Dict[str, Any], format: str = "json", 
                   filename: Optional[str] = None) -> str:
    """Export solution to a file."""
    create_directories()
    
    if filename is None:
        timestamp = solution_data.get("timestamp", "unknown").replace(":", "-")
        filename = f"solution_{timestamp}.{format}"
    
    filepath = Path("./exports") / filename
    
    try:
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(solution_data, f, indent=2)
        
        elif format.lower() == "txt":
            with open(filepath, 'w') as f:
                f.write(format_solution_output(solution_data))
        
        elif format.lower() == "md":
            markdown_content = convert_solution_to_markdown(solution_data)
            with open(filepath, 'w') as f:
                f.write(markdown_content)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Solution exported to: {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Error exporting solution: {e}")
        raise


def convert_solution_to_markdown(solution_data: Dict[str, Any]) -> str:
    """Convert solution data to Markdown format."""
    lines = []
    
    # Header
    lines.append("# Mathematical Solution")
    lines.append("")
    
    # Problem
    if "problem" in solution_data:
        lines.append("## Problem")
        lines.append(f"```")
        lines.append(solution_data['problem'])
        lines.append("```")
        lines.append("")
    
    # Solution
    if "solution" in solution_data:
        lines.append("## Solution")
        lines.append(solution_data['solution'])
        lines.append("")
    
    # Steps
    if "steps" in solution_data and solution_data["steps"]:
        lines.append("## Solution Steps")
        for step in solution_data["steps"]:
            step_num = step.get("step_number", "?")
            action = step.get("action", "")
            lines.append(f"{step_num}. {action}")
        lines.append("")
    
    # Metadata
    lines.append("## Metadata")
    metadata_items = []
    if "method" in solution_data:
        metadata_items.append(f"- **Method**: {solution_data['method']}")
    if "tools_used" in solution_data and solution_data["tools_used"]:
        metadata_items.append(f"- **Tools**: {', '.join(solution_data['tools_used'])}")
    if "confidence_score" in solution_data:
        confidence = solution_data['confidence_score']
        metadata_items.append(f"- **Confidence**: {confidence:.1%}")
    if "processing_time" in solution_data:
        time_taken = solution_data['processing_time']
        metadata_items.append(f"- **Processing Time**: {time_taken:.2f} seconds")
    
    lines.extend(metadata_items)
    
    return "\\n".join(lines)


def validate_mathematical_expression(expression: str) -> Tuple[bool, str]:
    """Validate a mathematical expression."""
    try:
        # Basic validation - check for balanced parentheses
        if expression.count('(') != expression.count(')'):
            return False, "Unbalanced parentheses"
        
        # Check for invalid characters (basic check)
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-*/^().,= <>≤≥≠πθαβγδλμσφω∫∑∏√")
        if not set(expression).issubset(allowed_chars):
            return False, "Contains invalid characters"
        
        # Check for common syntax errors
        if any(pattern in expression for pattern in ["++", "--", "*/", "//"]):
            return False, "Invalid operator combination"
        
        return True, "Valid expression"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging."""
    import platform
    import sys
    
    info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "working_directory": os.getcwd()
    }
    
    # Check for optional dependencies
    optional_deps = ["numpy", "sympy", "matplotlib", "plotly", "scipy", "pandas", "langchain"]
    dependency_status = {}
    
    for dep in optional_deps:
        try:
            __import__(dep)
            dependency_status[dep] = "available"
        except ImportError:
            dependency_status[dep] = "not available"
    
    info["dependencies"] = dependency_status
    
    return info
