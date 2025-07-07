"""
Advanced Math Agent - Main Application
A sophisticated mathematical problem-solving agent using LangChain and OpenRouter
"""
import sys
import os
import argparse
import json
from typing import Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import MathAgent, MathAgentConfig
from utils import (
    setup_logging, load_environment, validate_api_keys,
    format_solution_output, parse_problem_type, export_solution,
    get_system_info, create_directories
)


class MathAgentApp:
    """Main application class for the Math Agent."""
    
    def __init__(self):
        self.agent = None
        self.config = None
        
    def initialize(self, log_level: str = "INFO", config_file: Optional[str] = None):
        """Initialize the application."""
        # Setup logging
        setup_logging(level=log_level, log_file="./logs/mathagent.log")
        
        # Load environment
        load_environment()
        
        # Create necessary directories
        create_directories()
        
        # Validate API keys
        api_validation = validate_api_keys()
        if not api_validation.get("OPENROUTER_API_KEY", False):
            print("❌ Error: OPENROUTER_API_KEY is required but not found.")
            print("Please set your OpenRouter API key in the .env file or environment.")
            sys.exit(1)
        
        # Load configuration
        if config_file and os.path.exists(config_file):
            self.config = self._load_config_from_file(config_file)
        else:
            self.config = MathAgentConfig()
        
        # Initialize the agent
        try:
            self.agent = MathAgent(self.config)
            print("✅ Math Agent initialized successfully!")
            
            # Show system info in verbose mode
            if self.config.verbose:
                self._show_system_info()
                
        except Exception as e:
            print(f"❌ Error initializing Math Agent: {e}")
            sys.exit(1)
    
    def _load_config_from_file(self, config_file: str) -> MathAgentConfig:
        """Load configuration from a JSON file."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            config = MathAgentConfig()
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            return config
        except Exception as e:
            print(f"Warning: Error loading config file {config_file}: {e}")
            return MathAgentConfig()
    
    def _show_system_info(self):
        """Display system information."""
        print("\\n" + "="*60)
        print("🔍 System Information")
        print("="*60)
        
        info = get_system_info()
        print(f"Platform: {info['platform']}")
        print(f"Python: {info['python_version'].split()[0]}")
        
        print("\\nDependency Status:")
        for dep, status in info['dependencies'].items():
            status_icon = "✅" if status == "available" else "❌"
            print(f"  {status_icon} {dep}: {status}")
        
        print("="*60 + "\\n")
    
    def solve_problem(self, problem: str, context: Optional[str] = None, 
                     export_format: Optional[str] = None) -> dict:
        """Solve a mathematical problem."""
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        print(f"🔢 Solving: {problem}")
        print("⏳ Processing...")
        
        # Solve the problem
        result = self.agent.solve(problem, context)
        
        # Display the result
        print("\\n" + format_solution_output(result))
        
        # Export if requested
        if export_format and result.get("success", False):
            try:
                filepath = export_solution(result, export_format)
                print(f"\\n📄 Solution exported to: {filepath}")
            except Exception as e:
                print(f"⚠️ Warning: Export failed: {e}")
        
        return result
    
    def explain_concept(self, concept: str) -> dict:
        """Explain a mathematical concept."""
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        print(f"📚 Explaining concept: {concept}")
        print("⏳ Processing...")
        
        result = self.agent.explain_concept(concept)
        print("\\n" + format_solution_output(result))
        
        return result
    
    def verify_solution(self, problem: str, solution: str) -> dict:
        """Verify a proposed solution."""
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        print(f"✅ Verifying solution for: {problem}")
        print("⏳ Processing...")
        
        result = self.agent.verify_solution(problem, solution)
        print("\\n" + format_solution_output(result))
        
        return result
    
    def get_learning_insights(self) -> dict:
        """Get learning insights for the user."""
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        insights = self.agent.get_learning_insights()
        
        if "error" not in insights:
            print("\\n📊 Learning Insights")
            print("="*50)
            
            patterns = insights.get("learning_patterns", [])
            if patterns:
                print("📈 Learning Patterns:")
                for pattern in patterns:
                    success_rate = pattern.get("success_rate", 0) * 100
                    print(f"  • {pattern.get('topic', 'Unknown')}: {success_rate:.1f}% success rate")
            
            stats = insights.get("total_problems_solved", 0)
            if stats > 0:
                avg_conf = insights.get("average_confidence", 0) * 100
                methods = insights.get("methods_used", 0)
                print(f"\\n📊 Statistics:")
                print(f"  • Problems solved: {stats}")
                print(f"  • Average confidence: {avg_conf:.1f}%")
                print(f"  • Methods used: {methods}")
        else:
            print(f"⚠️ Could not get learning insights: {insights['error']}")
        
        return insights
    
    def interactive_mode(self):
        """Run the agent in interactive mode."""
        print("🤖 Advanced Math Agent - Interactive Mode")
        print("Type 'help' for commands, 'quit' to exit\\n")
        
        while True:
            try:
                user_input = input("Math> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                
                elif user_input.lower() == 'insights':
                    self.get_learning_insights()
                
                elif user_input.lower() == 'info':
                    self._show_system_info()
                
                elif user_input.startswith('explain '):
                    concept = user_input[8:].strip()
                    self.explain_concept(concept)
                
                elif user_input.startswith('verify '):
                    # Simple parsing for verification
                    parts = user_input[7:].split(' solution: ')
                    if len(parts) == 2:
                        problem, solution = parts
                        self.verify_solution(problem.strip(), solution.strip())
                    else:
                        print("❌ Format: verify <problem> solution: <solution>")
                
                elif user_input.startswith('export '):
                    print("ℹ️ Export will be applied to the next solution.")
                    format_type = user_input[7:].strip()
                    if format_type not in ['json', 'txt', 'md']:
                        print("❌ Supported formats: json, txt, md")
                
                else:
                    # Treat as a math problem
                    problem_type = parse_problem_type(user_input)
                    print(f"🔍 Detected problem type: {problem_type}")
                    self.solve_problem(user_input)
                
                print()  # Add spacing
                
            except KeyboardInterrupt:
                print("\\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_help(self):
        """Show help information."""
        help_text = """
📖 Math Agent Commands:

Basic Usage:
  • Type any mathematical problem to solve it
  • explain <concept>     - Get explanation of a mathematical concept
  • verify <problem> solution: <solution> - Verify a proposed solution
  
Utility Commands:
  • insights             - Show your learning insights and progress
  • info                 - Show system information
  • help                 - Show this help message
  • quit/exit/q          - Exit the application

Examples:
  • Find the derivative of x^2 + 3x - 5
  • Solve the equation 2x + 5 = 13
  • explain derivatives
  • verify 2x + 5 = 13 solution: x = 4

The agent supports:
  ✓ Calculus (derivatives, integrals, limits)
  ✓ Algebra (equations, factoring, simplification)  
  ✓ Statistics (probability, descriptive stats)
  ✓ Geometry (area, perimeter, volume)
  ✓ Number theory (primes, GCD, LCM)
  ✓ And much more!
"""
        print(help_text)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Advanced Math Agent - Solve mathematical problems with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "problem", 
        nargs="?", 
        help="Mathematical problem to solve (if not provided, enters interactive mode)"
    )
    
    parser.add_argument(
        "--context", 
        help="Additional context for the problem"
    )
    
    parser.add_argument(
        "--explain", 
        help="Explain a mathematical concept"
    )
    
    parser.add_argument(
        "--verify", 
        nargs=2, 
        metavar=("PROBLEM", "SOLUTION"),
        help="Verify a problem and proposed solution"
    )
    
    parser.add_argument(
        "--export", 
        choices=["json", "txt", "md"],
        help="Export solution to file (json, txt, or md)"
    )
    
    parser.add_argument(
        "--config", 
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level"
    )
    
    parser.add_argument(
        "--insights", 
        action="store_true",
        help="Show learning insights"
    )
    
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Force interactive mode"
    )
    
    args = parser.parse_args()
    
    # Initialize the application
    app = MathAgentApp()
    app.initialize(log_level=args.log_level, config_file=args.config)
    
    try:
        # Handle different modes
        if args.insights:
            app.get_learning_insights()
        
        elif args.explain:
            app.explain_concept(args.explain)
        
        elif args.verify:
            problem, solution = args.verify
            app.verify_solution(problem, solution)
        
        elif args.problem:
            app.solve_problem(args.problem, args.context, args.export)
        
        elif args.interactive or len(sys.argv) == 1:
            app.interactive_mode()
        
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
