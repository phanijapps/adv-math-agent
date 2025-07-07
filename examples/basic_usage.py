"""
Example usage of the Math Agent
"""
from agents import MathAgent
from utils import setup_logging, load_environment


def main():
    """Demonstrate the Math Agent capabilities."""
    # Setup
    setup_logging()
    load_environment()
    
    # Initialize agent
    agent = MathAgent()
    
    # Example problems
    problems = [
        "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
        "Solve the equation 2x^2 - 8x + 6 = 0",
        "Calculate the area of a circle with radius 5",
        "What is the probability of rolling a sum of 7 with two dice?",
        "Find the limit of (sin(x)/x) as x approaches 0"
    ]
    
    print("🔢 Math Agent Examples")
    print("=" * 50)
    
    for i, problem in enumerate(problems, 1):
        print(f"\\nExample {i}: {problem}")
        print("-" * 40)
        
        try:
            result = agent.solve(problem)
            
            if result.get("success", False):
                print("✅ Solution:")
                print(result.get("solution", "No solution found"))
                
                tools_used = result.get("tools_used", [])
                if tools_used:
                    print(f"🔧 Tools used: {', '.join(tools_used)}")
                
                confidence = result.get("confidence_score", 0)
                print(f"📊 Confidence: {confidence:.1%}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    # Show learning insights
    print("\\n" + "=" * 50)
    print("📊 Learning Insights")
    print("=" * 50)
    
    insights = agent.get_learning_insights()
    if "error" not in insights:
        total_problems = insights.get("total_problems_solved", 0)
        avg_confidence = insights.get("average_confidence", 0)
        print(f"Problems solved: {total_problems}")
        print(f"Average confidence: {avg_confidence:.1%}")
    else:
        print(f"Could not retrieve insights: {insights.get('error')}")


if __name__ == "__main__":
    main()
