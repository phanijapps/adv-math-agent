"""
Advanced Math Agent - Modular Implementation
A sophisticated mathematical problem-solving agent using LangChain and OpenRouter
"""
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modular components
from .config import MathAgentConfig
from .agent_initializer import AgentInitializer
from .problem_processor import ProblemProcessor
from .result_processor import ResultProcessor
from .solver_orchestrator import SolverOrchestrator

from llm.openrouter import ModelManager
from memory.memory_manager import MemoryManager


class MathAgent:
    """
    Advanced Mathematical Problem Solving Agent
    
    Features:
    - Multi-step reasoning for complex problems
    - Memory integration for learning patterns
    - Tool orchestration for calculations
    - Step-by-step solution explanations
    - Modular architecture for maintainability
    """
    
    def __init__(self, config: Optional[MathAgentConfig] = None):
        """
        Initialize the MathAgent with modular components.
        
        Args:
            config: Configuration object for the agent
        """
        self.config = config or MathAgentConfig.create_default()
        
        # Initialize core components
        self._initialize_components()
        
        logger.info(f"Initialized MathAgent with {len(self.tools)} tools")
    
    def _initialize_components(self):
        """Initialize all agent components."""
        # Core infrastructure
        self.model_manager = ModelManager()
        self.memory_manager = MemoryManager() if self.config.memory_enabled else None
        
        # Load tools
        self.tools = AgentInitializer.load_tools()
        
        # Initialize LangChain agent if available
        self.agent_executor = AgentInitializer.create_langchain_agent(
            self.model_manager, 
            self.tools, 
            self.config
        )
        
        # Initialize solver orchestrator
        self.solver = SolverOrchestrator(
            self.model_manager, 
            self.agent_executor
        )
        
        # Validate setup
        self.validation_status = AgentInitializer.validate_setup(
            self.model_manager, 
            self.memory_manager, 
            self.tools
        )
    
    def solve(self, problem: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Solve a mathematical problem with comprehensive analysis.
        
        Args:
            problem: The mathematical problem to solve
            context: Additional context or constraints
            
        Returns:
            Dictionary containing solution, steps, and metadata
        """
        start_time = datetime.now()
        
        try:
            # Find similar problems in memory
            similar_solutions = self._find_similar_solutions(problem)
            
            # Enhance the problem prompt with context and history
            enhanced_problem = ProblemProcessor.enhance_problem_prompt(
                problem, context, similar_solutions
            )
            
            # Solve using the orchestrator
            result = self.solver.solve_problem(enhanced_problem)
            
            # Process and enhance the result
            processed_result = ResultProcessor.process_result(
                problem, result, start_time
            )
            
            # Store the solution in memory
            if self.memory_manager:
                self._store_solution_memory(problem, processed_result)
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Error solving problem: {e}")
            return {
                "success": False,
                "error": str(e),
                "problem": problem,
                "timestamp": datetime.now().isoformat()
            }
    
    def solve_with_retry(
        self, 
        problem: str, 
        context: Optional[str] = None,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Solve a problem with retry mechanism for better reliability.
        
        Args:
            problem: The mathematical problem to solve
            context: Additional context or constraints
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary containing solution, steps, and metadata
        """
        start_time = datetime.now()
        
        try:
            # Find similar problems in memory
            similar_solutions = self._find_similar_solutions(problem)
            
            # Enhance the problem prompt
            enhanced_problem = ProblemProcessor.enhance_problem_prompt(
                problem, context, similar_solutions
            )
            
            # Solve with retry mechanism
            result = self.solver.solve_with_retry(enhanced_problem, max_retries)
            
            # Process the result
            processed_result = ResultProcessor.process_result(
                problem, result, start_time
            )
            
            # Store successful solutions in memory
            if self.memory_manager and processed_result.get("success", False):
                self._store_solution_memory(problem, processed_result)
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Error in solve_with_retry: {e}")
            return {
                "success": False,
                "error": str(e),
                "problem": problem,
                "timestamp": datetime.now().isoformat()
            }
    
    def explain_concept(self, concept: str) -> Dict[str, Any]:
        """
        Explain a mathematical concept in detail.
        
        Args:
            concept: The mathematical concept to explain
            
        Returns:
            Dictionary containing explanation and metadata
        """
        logger.info(f"Explaining concept: {concept}")
        return self.solver.explain_concept(concept)
    
    def verify_solution(self, problem: str, proposed_solution: str) -> Dict[str, Any]:
        """
        Verify a proposed solution to a mathematical problem.
        
        Args:
            problem: The original mathematical problem
            proposed_solution: The solution to verify
            
        Returns:
            Dictionary containing verification results
        """
        logger.info(f"Verifying solution for: {problem[:50]}...")
        return self.solver.verify_solution(problem, proposed_solution)
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get learning insights for the current user."""
        if not self.memory_manager:
            return {"error": "Memory not enabled"}
        
        try:
            return self.memory_manager.get_learning_insights(self.config.user_id)
        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return {"error": str(e)}
    
    def analyze_problem(self, problem: str) -> Dict[str, Any]:
        """
        Analyze a mathematical problem without solving it.
        
        Args:
            problem: The mathematical problem to analyze
            
        Returns:
            Dictionary containing problem analysis
        """
        try:
            metadata = ProblemProcessor.extract_problem_metadata(problem)
            similar_solutions = self._find_similar_solutions(problem)
            
            analysis = {
                "problem": problem,
                "metadata": metadata,
                "similar_problems_count": len(similar_solutions),
                "similar_problems": similar_solutions[:3],  # Top 3
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing problem: {e}")
            return {
                "error": str(e),
                "problem": problem
            }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get the current status of the agent and its components."""
        return {
            "config": {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "max_iterations": self.config.max_iterations,
                "verbose": self.config.verbose,
                "memory_enabled": self.config.memory_enabled,
                "user_id": self.config.user_id
            },
            "components": {
                "model_manager": self.model_manager is not None,
                "memory_manager": self.memory_manager is not None,
                "tools_count": len(self.tools),
                "agent_executor": self.agent_executor is not None,
                "solver": self.solver is not None
            },
            "validation": self.validation_status
        }
    
    def _find_similar_solutions(self, problem: str) -> List[Dict]:
        """Find similar problems solved before."""
        if not self.memory_manager:
            return []
        
        try:
            return self.memory_manager.find_similar_solutions(
                self.config.user_id, problem
            )
        except Exception as e:
            logger.warning(f"Error finding similar solutions: {e}")
            return []
    
    def _store_solution_memory(self, problem: str, result: Dict[str, Any]):
        """Store the solution in memory for future reference."""
        if not self.memory_manager or not result.get("success", False):
            return
        
        try:
            solution = result.get("solution", "")
            method = result.get("method", "unknown")
            tools_used = result.get("tools_used", [])
            confidence = result.get("confidence_score", 0.8)
            
            self.memory_manager.remember_solution(
                user_id=self.config.user_id,
                problem=problem,
                solution=solution,
                method=method,
                tools_used=tools_used,
                success=True,
                confidence=confidence
            )
            
            logger.info(f"Stored solution in memory for problem: {problem[:50]}...")
            
        except Exception as e:
            logger.warning(f"Error storing solution in memory: {e}")


# Export the main classes for backward compatibility
__all__ = ["MathAgent", "MathAgentConfig"]
