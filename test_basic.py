"""
Quick start example for testing the Math Agent without full dependencies
"""
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    
    print("🔢 Advanced Math Agent - Basic Test")
    print("=" * 50)
    
    # Test utility functions
    from utils import parse_problem_type, validate_mathematical_expression
    
    # Test problem type detection
    test_problems = [
        "Find the derivative of x^2 + 3x",
        "Solve the equation 2x + 5 = 13", 
        "Calculate the area of a triangle",
        "What is the probability of rolling a 6?"
    ]
    
    print("📝 Testing Problem Type Detection:")
    for problem in test_problems:
        problem_type = parse_problem_type(problem)
        print(f"  • '{problem}' → {problem_type}")
    
    # Test expression validation
    print("\\n✅ Testing Expression Validation:")
    test_expressions = [
        "x^2 + 2x + 1",
        "sin(x) + cos(x)", 
        "x^2 + 2x + 1)",  # Invalid
        "2x ++ 3"         # Invalid
    ]
    
    for expr in test_expressions:
        is_valid, message = validate_mathematical_expression(expr)
        status = "✅" if is_valid else "❌"
        print(f"  {status} '{expr}' - {message}")
    
    # Test memory manager initialization
    print("\\n🧠 Testing Memory Manager:")
    try:
        from memory import MemoryManager
        memory_manager = MemoryManager()
        print("  ✅ Memory manager initialized successfully")
        
        # Test storing and retrieving a memory
        memory_id = memory_manager.remember("test_user", "Test problem: 2+2=4", "algebra")
        print(f"  ✅ Memory stored with ID: {memory_id[:8]}...")
        
        memories = memory_manager.recall("test_user", "2+2", limit=1)
        print(f"  ✅ Retrieved {len(memories)} memory(ies)")
        
    except Exception as e:
        print(f"  ❌ Memory manager error: {e}")
    
    # Test agent initialization (will likely fail without API key)
    print("\\n🤖 Testing Agent Initialization:")
    try:
        from agents import MathAgent, MathAgentConfig
        
        config = MathAgentConfig()
        config.memory_enabled = False  # Disable for basic test
        
        # This will likely fail without API key, but we can test the structure
        agent = MathAgent(config)
        print("  ✅ Agent initialized successfully")
        
    except Exception as e:
        print(f"  ⚠️ Agent initialization failed (expected without API key): {e}")
        print("  💡 To fully test, set OPENROUTER_API_KEY in environment")
    
    print("\\n" + "=" * 50)
    print("🎉 Basic functionality test complete!")
    print("\\nTo use the full agent:")
    print("1. Set OPENROUTER_API_KEY environment variable")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run: python main.py")


if __name__ == "__main__":
    test_basic_functionality()
