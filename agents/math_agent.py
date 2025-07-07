"""
Advanced Math Agent using LangChain and OpenRouter
Following the modern create_react_agent pattern from user's memories
"""
import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from langchain import hub
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain.schema import BaseMessage, HumanMessage, SystemMessage
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.callbacks import StreamingStdOutCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not available. Agent will use fallback implementation.")
    LANGCHAIN_AVAILABLE = False

from llm.openrouter import OpenRouterLLM, ModelManager
from tools import get_all_tools
from memory.memory_manager import MemoryManager


class MathAgentConfig:
    """Configuration for the Math Agent."""
    
    def __init__(self):
        self.model = os.getenv("DEFAULT_MODEL", "openai/gpt-4-turbo-preview")
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", "10"))
        self.verbose = os.getenv("VERBOSE", "true").lower() == "true"
        self.memory_enabled = os.getenv("MEMORY_ENABLED", "true").lower() == "true"
        self.user_id = os.getenv("USER_ID", "default_user")


class MathAgent:
    """
    Advanced Mathematical Problem Solving Agent
    
    Features:
    - Multi-step reasoning for complex problems
    - Memory integration for learning patterns
    - Tool orchestration for calculations
    - Step-by-step solution explanations
    """
    
    def __init__(self, config: Optional[MathAgentConfig] = None):
        self.config = config or MathAgentConfig()
        
        # Initialize components
        self.model_manager = ModelManager()
        self.memory_manager = MemoryManager() if self.config.memory_enabled else None
        self.tools = self._load_tools()
        
        # Initialize agent
        if LANGCHAIN_AVAILABLE:
            self.agent_executor = self._create_langchain_agent()
        else:
            self.agent_executor = None
        
        logger.info(f"Initialized MathAgent with {len(self.tools)} tools")
    
    def _load_tools(self) -> List:
        """Load all available mathematical tools."""
        try:
            tools = get_all_tools()
            logger.info(f"Loaded {len(tools)} mathematical tools")
            return tools
        except Exception as e:
            logger.error(f"Error loading tools: {e}")
            return []
    
    def _create_langchain_agent(self):
        """Create the LangChain ReAct agent following user's preferred pattern."""
        try:
            # Get the primary model
            llm = self.model_manager.get_model("primary")
            
            # Load the ReAct prompt from LangChain Hub
            prompt = hub.pull("hwchase17/react")
            
            # Create the agent
            agent = create_react_agent(
                llm=llm.llm,  # Use the underlying ChatOpenAI instance
                tools=self.tools,
                prompt=prompt
            )
            
            # Create memory
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                k=10,  # Remember last 10 interactions
                return_messages=True
            ) if self.config.memory_enabled else None
            
            # Create agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=memory,
                verbose=self.config.verbose,
                max_iterations=self.config.max_iterations,
                handle_parsing_errors=True,
                callbacks=[StreamingStdOutCallbackHandler()] if self.config.verbose else None
            )
            
            return agent_executor
            
        except Exception as e:
            logger.error(f"Error creating LangChain agent: {e}")
            return None
    
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
            # Check for similar problems in memory
            similar_solutions = self._find_similar_solutions(problem)
            
            # Prepare the enhanced prompt
            enhanced_problem = self._enhance_problem_prompt(problem, context, similar_solutions)
            
            # Solve using the agent
            if self.agent_executor:
                result = self._solve_with_langchain(enhanced_problem)
            else:
                result = self._solve_with_fallback(enhanced_problem)
            
            # Process and enhance the result
            processed_result = self._process_result(problem, result, start_time)
            
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
    
    def _find_similar_solutions(self, problem: str) -> List[Dict]:
        """Find similar problems solved before."""
        if not self.memory_manager:
            return []
        
        try:
            return self.memory_manager.find_similar_solutions(self.config.user_id, problem)
        except Exception as e:
            logger.warning(f"Error finding similar solutions: {e}")
            return []
    
    def _enhance_problem_prompt(self, problem: str, context: Optional[str], 
                              similar_solutions: List[Dict]) -> str:
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
            # Fallback to direct LLM call
            return self._solve_with_fallback(problem)
    
    def _solve_with_fallback(self, problem: str) -> Dict[str, Any]:
        """Fallback solution method using direct LLM calls."""
        try:
            llm = self.model_manager.get_model("primary")
            
            # Create system message for math solving
            system_msg = SystemMessage(content="""
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
""")
            
            # Create human message with the problem
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
                "error": f"All solution methods failed: {str(e)}"
            }
    
    def _process_result(self, problem: str, result: Dict[str, Any], start_time: datetime) -> Dict[str, Any]:
        """Process and enhance the raw result."""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        processed = {
            "problem": problem,
            "success": result.get("success", False),
            "solution": result.get("solution", ""),
            "method": result.get("method", "unknown"),
            "tools_used": self._extract_tools_used(result),
            "confidence_score": self._calculate_confidence(result),
            "processing_time": duration,
            "timestamp": end_time.isoformat()
        }
        
        # Add intermediate steps if available
        if "intermediate_steps" in result:
            processed["steps"] = self._format_steps(result["intermediate_steps"])
        
        # Add error info if failed
        if not result.get("success", True):
            processed["error"] = result.get("error", "Unknown error")
        
        return processed
    
    def _extract_tools_used(self, result: Dict[str, Any]) -> List[str]:
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
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
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
        
        return min(confidence, 1.0)
    
    def _format_steps(self, intermediate_steps: List) -> List[Dict]:
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
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get learning insights for the current user."""
        if not self.memory_manager:
            return {"error": "Memory not enabled"}
        
        try:
            return self.memory_manager.get_learning_insights(self.config.user_id)
        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return {"error": str(e)}
    
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
        
        return self.solve(prompt, context="This is a concept explanation request")
    
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
"""
        
        return self.solve(prompt, context="This is a solution verification request")
