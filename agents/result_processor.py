"""
Result processing and analysis utilities for MathAgent
"""
import logging
from typing import Any, Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultProcessor:
    """Handles solution result processing and analysis."""
    
    @staticmethod
    def process_result(
        problem: str, 
        result: Dict[str, Any], 
        start_time: datetime
    ) -> Dict[str, Any]:
        """Process and enhance the raw result."""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        processed = {
            "problem": problem,
            "success": result.get("success", False),
            "solution": result.get("solution", ""),
            "method": result.get("method", "unknown"),
            "tools_used": ResultProcessor.extract_tools_used(result),
            "confidence_score": ResultProcessor.calculate_confidence(result),
            "processing_time": duration,
            "timestamp": end_time.isoformat(),
            "metadata": ResultProcessor.extract_solution_metadata(result)
        }
        
        # Add intermediate steps if available
        if "intermediate_steps" in result:
            processed["steps"] = ResultProcessor.format_steps(result["intermediate_steps"])
        
        # Add error info if failed
        if not result.get("success", True):
            processed["error"] = result.get("error", "Unknown error")
        
        return processed
    
    @staticmethod
    def extract_tools_used(result: Dict[str, Any]) -> List[str]:
        """Extract tools used from the result."""
        tools_used = []
        
        # Check intermediate steps for tool usage
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if isinstance(step, tuple) and len(step) >= 2:
                    action = step[0]
                    if hasattr(action, 'tool'):
                        tools_used.append(action.tool)
        
        return list(set(tools_used))  # Remove duplicates
    
    @staticmethod
    def calculate_confidence(result: Dict[str, Any]) -> float:
        """Calculate confidence score for the solution."""
        if not result.get("success", True):
            return 0.0
        
        confidence = 0.8  # Base confidence
        
        # Increase confidence if tools were used
        if "intermediate_steps" in result and result["intermediate_steps"]:
            confidence += 0.1
        
        # Increase confidence if verification was performed
        solution = result.get("solution", "").lower()
        if any(word in solution for word in ["verify", "check", "confirm"]):
            confidence += 0.1
        
        # Increase confidence based on solution length and detail
        solution_length = len(solution)
        if solution_length > 500:  # Detailed solution
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    @staticmethod
    def format_steps(intermediate_steps: List) -> List[Dict]:
        """Format intermediate steps for better readability."""
        formatted_steps = []
        
        for i, step in enumerate(intermediate_steps, 1):
            if isinstance(step, tuple) and len(step) >= 2:
                action, observation = step[0], step[1]
                
                step_dict = {
                    "step_number": i,
                    "action": str(action),
                    "observation": str(observation)
                }
                
                # Extract tool info if available
                if hasattr(action, 'tool'):
                    step_dict["tool"] = action.tool
                if hasattr(action, 'tool_input'):
                    step_dict["tool_input"] = action.tool_input
                
                formatted_steps.append(step_dict)
        
        return formatted_steps
    
    @staticmethod
    def extract_solution_metadata(result: Dict[str, Any]) -> Dict:
        """Extract metadata from the solution."""
        solution = result.get("solution", "")
        
        metadata = {
            "solution_length": len(solution),
            "has_verification": any(word in solution.lower() 
                                  for word in ["verify", "check", "confirm"]),
            "has_steps": "step" in solution.lower(),
            "has_equations": any(op in solution for op in ["=", "+", "-", "*", "/"]),
            "method_used": result.get("method", "unknown"),
            "model_used": result.get("model", "unknown")
        }
        
        return metadata
    
    @staticmethod
    def validate_solution(result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the solution result."""
        validation = {
            "is_valid": True,
            "issues": [],
            "quality_score": 0.0
        }
        
        solution = result.get("solution", "")
        
        # Check for empty solution
        if not solution.strip():
            validation["is_valid"] = False
            validation["issues"].append("Empty solution")
            return validation
        
        # Check for error indicators
        error_indicators = ["error", "failed", "cannot", "unable"]
        if any(indicator in solution.lower() for indicator in error_indicators):
            validation["issues"].append("Solution contains error indicators")
        
        # Calculate quality score
        quality_factors = {
            "has_steps": "step" in solution.lower(),
            "has_explanation": len(solution) > 100,
            "has_verification": any(word in solution.lower() 
                                  for word in ["verify", "check", "confirm"]),
            "proper_format": solution.strip().endswith((".", "!", "?"))
        }
        
        validation["quality_score"] = sum(quality_factors.values()) / len(quality_factors)
        
        return validation
