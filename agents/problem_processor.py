"""
Problem processing and enhancement utilities for MathAgent
"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ProblemProcessor:
    """Handles problem analysis and enhancement."""
    
    @staticmethod
    def enhance_problem_prompt(
        problem: str, 
        context: Optional[str], 
        similar_solutions: List[Dict]
    ) -> str:
        """Create an enhanced prompt with context and similar solutions."""
        prompt_parts = [
            "I need to solve this mathematical problem step by step:",
            f"Problem: {problem}"
        ]
        
        if context:
            prompt_parts.append(f"Additional Context: {context}")
        
        if similar_solutions:
            prompt_parts.append("\\nI've solved similar problems before:")
            for i, sol in enumerate(similar_solutions[:3], 1):  # Show top 3
                prompt_parts.append(f"{i}. Problem: {sol.get('problem', 'N/A')}")
                prompt_parts.append(f"   Method: {sol.get('method', 'N/A')}")
                prompt_parts.append(f"   Confidence: {sol.get('confidence', 0):.2f}")
        
        prompt_parts.extend([
            "\\nPlease provide:",
            "1. A clear step-by-step solution",
            "2. Explanation of the mathematical concepts involved", 
            "3. The tools and methods used",
            "4. Verification of the answer when possible",
            "5. Alternative approaches if applicable"
        ])
        
        return "\\n".join(prompt_parts)
    
    @staticmethod
    def extract_problem_metadata(problem: str) -> Dict:
        """Extract metadata from a mathematical problem."""
        metadata = {
            "length": len(problem),
            "has_equations": any(op in problem for op in ["=", "+", "-", "*", "/", "^"]),
            "has_variables": any(var in problem.lower() for var in ["x", "y", "z", "n"]),
            "keywords": ProblemProcessor._extract_math_keywords(problem),
            "difficulty_indicators": ProblemProcessor._assess_difficulty(problem)
        }
        return metadata
    
    @staticmethod
    def _extract_math_keywords(problem: str) -> List[str]:
        """Extract mathematical keywords from problem text."""
        keywords = []
        math_terms = [
            "derivative", "integral", "limit", "function", "equation",
            "solve", "find", "calculate", "factor", "simplify",
            "trigonometric", "logarithm", "exponential", "polynomial",
            "matrix", "vector", "probability", "statistics"
        ]
        
        problem_lower = problem.lower()
        for term in math_terms:
            if term in problem_lower:
                keywords.append(term)
        
        return keywords
    
    @staticmethod
    def _assess_difficulty(problem: str) -> Dict:
        """Assess problem difficulty based on text analysis."""
        indicators = {
            "advanced_terms": 0,
            "multiple_steps": False,
            "complex_notation": False
        }
        
        advanced_terms = [
            "differential", "partial", "multivariable", "series",
            "complex", "abstract", "proof", "theorem"
        ]
        
        problem_lower = problem.lower()
        for term in advanced_terms:
            if term in problem_lower:
                indicators["advanced_terms"] += 1
        
        # Check for multiple steps
        step_indicators = ["then", "next", "after", "finally", "also"]
        indicators["multiple_steps"] = any(word in problem_lower for word in step_indicators)
        
        # Check for complex notation
        complex_notation = ["∫", "∑", "∏", "∂", "∇", "∞"]
        indicators["complex_notation"] = any(symbol in problem for symbol in complex_notation)
        
        return indicators
