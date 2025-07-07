"""
Solver orchestration for MathAgent
Handles different solving strategies and fallback mechanisms
"""
import logging
from typing import Any, Dict, Optional
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class SolverOrchestrator:
    """Orchestrates different solving approaches for mathematical problems."""
    
    def __init__(self, model_manager, agent_executor=None):
        self.model_manager = model_manager
        self.agent_executor = agent_executor
    
    def solve_problem(self, enhanced_problem: str) -> Dict[str, Any]:
        """
        Solve a problem using the best available method.
        
        Args:
            enhanced_problem: The enhanced problem prompt
            
        Returns:
            Solution result dictionary
        """
        # Try LangChain agent first if available
        if self.agent_executor:
            result = self._solve_with_langchain(enhanced_problem)
            if result.get("success", False):
                return result
            
            logger.warning("LangChain solver failed, falling back to direct LLM")
        
        # Fallback to direct LLM
        return self._solve_with_fallback(enhanced_problem)
    
    def _solve_with_langchain(self, problem: str) -> Dict[str, Any]:
        """Solve using LangChain agent."""
        try:
            result = self.agent_executor.invoke({"input": problem})
            
            return {
                "success": True,
                "solution": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", []),
                "method": "langchain_react"
            }
            
        except Exception as e:
            logger.error(f"LangChain agent error: {e}")
            return {
                "success": False,
                "error": str(e),
                "method": "langchain_react"
            }
    
    def _solve_with_fallback(self, problem: str) -> Dict[str, Any]:
        """Fallback solution method using direct LLM calls."""
        try:
            llm = self.model_manager.get_model("primary")
            
            # Create system message for math solving
            system_msg = self._create_system_message()
            human_msg = HumanMessage(content=problem)
            
            # Get response
            response = llm.invoke([system_msg, human_msg])
            
            return {
                "success": True,
                "solution": response.content,
                "method": "direct_llm",
                "model": llm.model
            }
            
        except Exception as e:
            logger.error(f"Fallback solution error: {e}")
            return {
                "success": False,
                "error": f"All solution methods failed: {str(e)}",
                "method": "direct_llm"
            }
    
    def _create_system_message(self) -> SystemMessage:
        """Create the system message for mathematical problem solving."""
        content = """
You are an advanced mathematical problem solver. Your role is to:
1. Analyze mathematical problems thoroughly
2. Provide step-by-step solutions with clear explanations
3. Show all work and reasoning
4. Use mathematical notation when appropriate
5. Verify answers when possible

Always structure your response with clear sections:
- Problem Analysis: Break down what type of problem this is
- Solution Steps: Show each step clearly with explanations  
- Final Answer: State the final answer clearly
- Verification: Check the answer if possible

For mathematical calculations, show your work step by step.
Be precise, thorough, and educational in your explanations.
"""
        return SystemMessage(content=content.strip())
    
    def solve_with_retry(
        self, 
        enhanced_problem: str, 
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Solve with retry mechanism for better reliability.
        
        Args:
            enhanced_problem: The enhanced problem prompt
            max_retries: Maximum number of retry attempts
            
        Returns:
            Solution result dictionary
        """
        for attempt in range(max_retries + 1):
            try:
                result = self.solve_problem(enhanced_problem)
                
                # If successful, return immediately
                if result.get("success", False):
                    if attempt > 0:
                        result["retry_attempt"] = attempt
                    return result
                
                # If not successful and not the last attempt, try again
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                    continue
                
                # Last attempt failed
                return result
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed with exception: {e}")
                if attempt == max_retries:
                    return {
                        "success": False,
                        "error": f"All {max_retries + 1} attempts failed: {str(e)}",
                        "method": "retry_exhausted"
                    }
    
    def explain_concept(self, concept: str) -> Dict[str, Any]:
        """Explain a mathematical concept in detail."""
        prompt = f"""
Please provide a comprehensive explanation of the mathematical concept: {concept}

Include:
1. Definition and fundamental principles
2. Key properties and characteristics  
3. Common applications and examples
4. Related concepts and connections
5. Common misconceptions or difficulties
6. Practice problems or exercises (if applicable)

Make the explanation clear and educational.
"""
        
        return self.solve_problem(prompt)
    
    def verify_solution(self, problem: str, proposed_solution: str) -> Dict[str, Any]:
        """Verify a proposed solution to a mathematical problem."""
        prompt = f"""
Please verify this proposed solution:

Problem: {problem}
Proposed Solution: {proposed_solution}

Please:
1. Check if the solution is correct
2. Identify any errors or issues
3. Provide the correct solution if the proposed one is wrong
4. Explain the verification process

Be thorough and constructive in your analysis.
"""
        
        return self.solve_problem(prompt)
