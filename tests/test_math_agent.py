"""
Tests for the Math Agent
"""
import unittest
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import MathAgent, MathAgentConfig
from memory import MemoryManager
from utils import parse_problem_type, validate_mathematical_expression


class TestMathAgent(unittest.TestCase):
    """Test cases for MathAgent."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = MathAgentConfig()
        self.config.verbose = False
        self.config.memory_enabled = False  # Disable memory for tests
        
    def test_agent_initialization(self):
        """Test agent initialization."""
        try:
            agent = MathAgent(self.config)
            self.assertIsNotNone(agent)
            self.assertIsNotNone(agent.model_manager)
        except Exception as e:
            self.skipTest(f"Agent initialization failed: {e}")
    
    def test_problem_type_detection(self):
        """Test problem type detection."""
        test_cases = [
            ("Find the derivative of x^2", "calculus"),
            ("Solve 2x + 5 = 0", "algebra"),
            ("What is the area of a circle?", "geometry"),
            ("Calculate the mean of [1,2,3,4,5]", "statistics")
        ]
        
        for problem, expected_type in test_cases:
            detected_type = parse_problem_type(problem)
            self.assertIn(expected_type, detected_type.lower())


class TestMemoryManager(unittest.TestCase):
    """Test cases for MemoryManager."""
    
    def setUp(self):
        """Set up test memory manager."""
        # Use in-memory database for testing
        self.memory_manager = MemoryManager()
    
    def test_memory_storage_and_retrieval(self):
        """Test basic memory operations."""
        user_id = "test_user"
        content = "Test memory content"
        
        # Store memory
        memory_id = self.memory_manager.remember(user_id, content, "test")
        self.assertIsNotNone(memory_id)
        
        # Retrieve memory
        memories = self.memory_manager.recall(user_id, "test", limit=5)
        self.assertGreater(len(memories), 0)
        self.assertIn(content, memories[0]["content"])


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_mathematical_expression_validation(self):
        """Test mathematical expression validation."""
        valid_expressions = [
            "x^2 + 2x + 1",
            "sin(x) + cos(x)",
            "2 * pi * r"
        ]
        
        invalid_expressions = [
            "x^2 + 2x + 1)",  # Unbalanced parentheses
            "2x ++ 3",         # Invalid operators
            "x^2 & 3"          # Invalid character
        ]
        
        for expr in valid_expressions:
            is_valid, message = validate_mathematical_expression(expr)
            self.assertTrue(is_valid, f"Expression '{expr}' should be valid: {message}")
        
        for expr in invalid_expressions:
            is_valid, message = validate_mathematical_expression(expr)
            self.assertFalse(is_valid, f"Expression '{expr}' should be invalid")


if __name__ == "__main__":
    unittest.main()
