#!/usr/bin/env python3
"""
🤖 MathBot - Flashy Command Line Math Agent
A beautiful, colorful command-line tool for solving mathematical problems
"""

import sys
import os
import argparse
import time
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Third-party imports for beautiful CLI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.live import Live
    from rich.align import Align
    from rich.layout import Layout
    from rich.padding import Padding
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  Installing rich for beautiful CLI...")
    os.system(f"{sys.executable} -m pip install rich")
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        from rich.prompt import Prompt, Confirm
        from rich.progress import Progress, SpinnerColumn, TextColumn
        from rich.syntax import Syntax
        from rich.table import Table
        from rich.live import Live
        from rich.align import Align
        from rich.layout import Layout
        from rich.padding import Padding
        from rich import print as rprint
        RICH_AVAILABLE = True
    except ImportError:
        RICH_AVAILABLE = False

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.math_agent import MathAgent, MathAgentConfig
from utils.helpers import load_environment, validate_api_keys, create_directories


@dataclass
class MathBotConfig:
    """Configuration for MathBot CLI"""
    theme: str = "cyan"
    show_thinking: bool = True
    show_step_by_step: bool = True
    export_results: bool = False
    verbose: bool = False


class MathBot:
    """🤖 MathBot - Beautiful command-line math agent"""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.agent = None
        self.config = MathBotConfig()
        self.session_stats = {
            "problems_solved": 0,
            "start_time": time.time(),
            "total_thinking_time": 0
        }
        
    def initialize(self):
        """Initialize MathBot and underlying math agent"""
        if not RICH_AVAILABLE:
            print("🤖 MathBot Starting... (Basic Mode)")
            print("For the full experience, install rich: pip install rich")
        else:
            self.show_startup_banner()
        
        # Load environment and validate
        load_environment()
        create_directories()
        
        api_validation = validate_api_keys()
        if not api_validation.get("OPENROUTER_API_KEY", False):
            self.error_panel("🔑 OpenRouter API Key Missing", 
                           "Please set OPENROUTER_API_KEY in your .env file")
            sys.exit(1)
        
        # Initialize the math agent
        try:
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console
                ) as progress:
                    task = progress.add_task("🧠 Initializing MathBot brain...", total=None)
                    
                    agent_config = MathAgentConfig()
                    agent_config.verbose = self.config.verbose
                    self.agent = MathAgent(agent_config)
                    
                    progress.update(task, description="✅ MathBot ready!")
                    time.sleep(0.5)
            else:
                print("🧠 Initializing MathBot...")
                agent_config = MathAgentConfig()
                self.agent = MathAgent(agent_config)
                print("✅ MathBot ready!")
                
        except Exception as e:
            self.error_panel("❌ Initialization Failed", str(e))
            sys.exit(1)
    
    def show_startup_banner(self):
        """Display beautiful startup banner"""
        banner_text = """
    ███╗   ███╗ █████╗ ████████╗██╗  ██╗██████╗  ██████╗ ████████╗
    ████╗ ████║██╔══██╗╚══██╔══╝██║  ██║██╔══██╗██╔═══██╗╚══██╔══╝
    ██╔████╔██║███████║   ██║   ███████║██████╔╝██║   ██║   ██║   
    ██║╚██╔╝██║██╔══██║   ██║   ██╔══██║██╔══██╗██║   ██║   ██║   
    ██║ ╚═╝ ██║██║  ██║   ██║   ██║  ██║██████╔╝╚██████╔╝   ██║   
    ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═════╝  ╚═════╝    ╚═╝   
        """
        
        banner = Panel(
            Align.center(Text(banner_text, style="bold cyan")),
            title="🤖 Welcome to MathBot",
            subtitle="Your AI-Powered Mathematics Assistant",
            border_style="cyan",
            padding=(1, 2)
        )
        
        self.console.print(banner)
        
        # Show quick stats
        info_table = Table(show_header=False, box=None, padding=(0, 2))
        info_table.add_row("🔢", "Solve complex mathematical problems")
        info_table.add_row("🧮", "Step-by-step explanations") 
        info_table.add_row("📚", "Concept explanations")
        info_table.add_row("✅", "Solution verification")
        info_table.add_row("💡", "Learning insights")
        
        self.console.print(Panel(
            info_table,
            title="✨ Capabilities",
            border_style="green",
            width=60
        ))
        
    def solve_problem(self, problem: str, show_work: bool = True) -> Dict[str, Any]:
        """Solve a mathematical problem with beautiful output"""
        start_time = time.time()
        
        # Show problem in a beautiful box
        problem_panel = Panel(
            Text(problem, style="bold white"),
            title="🔢 Problem to Solve",
            border_style="blue",
            padding=(1, 2)
        )
        
        if RICH_AVAILABLE:
            self.console.print(problem_panel)
        else:
            print(f"\n🔢 Problem: {problem}")
        
        # Show thinking process
        if show_work and RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                thinking_task = progress.add_task("🤔 Analyzing problem...", total=None)
                
                # Simulate thinking steps
                thinking_steps = [
                    "🤔 Analyzing problem structure...",
                    "🔍 Identifying mathematical concepts...",
                    "🧮 Planning solution strategy...",
                    "⚡ Computing solution...",
                    "✅ Verifying results..."
                ]
                
                for step in thinking_steps:
                    progress.update(thinking_task, description=step)
                    time.sleep(0.3)
                
                # Actually solve the problem
                result = self.agent.solve(problem)
                
                progress.update(thinking_task, description="🎉 Solution complete!")
                time.sleep(0.2)
        else:
            if not RICH_AVAILABLE:
                print("🤔 Thinking...")
            result = self.agent.solve(problem)
        
        solve_time = time.time() - start_time
        self.session_stats["total_thinking_time"] += solve_time
        self.session_stats["problems_solved"] += 1
        
        # Display beautiful result
        self.display_solution(result, solve_time)
        
        return result
    
    def display_solution(self, result: Dict[str, Any], solve_time: float):
        """Display the solution in a beautiful format"""
        if not result.get("success", False):
            self.error_panel("❌ Solution Failed", result.get("error", "Unknown error"))
            return
        
        solution = result.get("solution", "No solution provided")
        explanation = result.get("explanation", "")
        steps = result.get("steps", [])
        confidence = result.get("confidence", 0)
        
        if RICH_AVAILABLE:
            # Main answer panel with emoji and color
            answer_text = Text()
            answer_text.append("🎯 FINAL ANSWER\n\n", style="bold yellow")
            answer_text.append(solution, style="bold green")
            
            answer_panel = Panel(
                Align.center(answer_text),
                title="✨ Solution",
                border_style="green",
                padding=(1, 2)
            )
            self.console.print(answer_panel)
            
            # Show detailed explanation if available
            if explanation and self.config.show_thinking:
                explanation_panel = Panel(
                    Text(explanation, style="white"),
                    title="🧠 Explanation",
                    border_style="cyan",
                    padding=(1, 2)
                )
                self.console.print(explanation_panel)
            
            # Show step-by-step if available
            if steps and self.config.show_step_by_step:
                steps_text = Text()
                for i, step in enumerate(steps, 1):
                    steps_text.append(f"Step {i}: ", style="bold cyan")
                    steps_text.append(f"{step}\n", style="white")
                
                steps_panel = Panel(
                    steps_text,
                    title="📝 Step-by-Step Solution",
                    border_style="yellow",
                    padding=(1, 2)
                )
                self.console.print(steps_panel)
            
            # Show metadata
            meta_table = Table(show_header=False, box=None)
            meta_table.add_row("⏱️ Solve Time:", f"{solve_time:.2f}s")
            meta_table.add_row("🎯 Confidence:", f"{confidence:.1%}" if confidence > 0 else "N/A")
            meta_table.add_row("🔢 Problem #:", str(self.session_stats["problems_solved"]))
            
            meta_panel = Panel(
                meta_table,
                title="📊 Metadata",
                border_style="dim",
                width=40
            )
            self.console.print(meta_panel)
            
        else:
            # Fallback for non-rich display
            print("\n" + "="*60)
            print("🎯 FINAL ANSWER")
            print("="*60)
            print(f"✅ {solution}")
            
            if explanation:
                print(f"\n🧠 Explanation:\n{explanation}")
            
            if steps:
                print(f"\n📝 Steps:")
                for i, step in enumerate(steps, 1):
                    print(f"  {i}. {step}")
            
            print(f"\n⏱️ Solved in {solve_time:.2f}s")
            print("="*60)
    
    def explain_concept(self, concept: str):
        """Explain a mathematical concept"""
        if RICH_AVAILABLE:
            concept_panel = Panel(
                Text(concept, style="bold white"),
                title="📚 Concept to Explain",
                border_style="purple",
                padding=(1, 2)
            )
            self.console.print(concept_panel)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("📖 Preparing explanation...", total=None)
                result = self.agent.explain_concept(concept)
                progress.update(task, description="📚 Explanation ready!")
                time.sleep(0.2)
        else:
            print(f"\n📚 Explaining: {concept}")
            print("📖 Preparing explanation...")
            result = self.agent.explain_concept(concept)
        
        if result.get("success", False):
            explanation = result.get("explanation", "No explanation available")
            examples = result.get("examples", [])
            
            if RICH_AVAILABLE:
                # Main explanation
                explanation_panel = Panel(
                    Text(explanation, style="white"),
                    title=f"📚 {concept.title()}",
                    border_style="purple",
                    padding=(1, 2)
                )
                self.console.print(explanation_panel)
                
                # Examples if available
                if examples:
                    examples_text = Text()
                    for i, example in enumerate(examples, 1):
                        examples_text.append(f"Example {i}: ", style="bold purple")
                        examples_text.append(f"{example}\n", style="white")
                    
                    examples_panel = Panel(
                        examples_text,
                        title="💡 Examples",
                        border_style="yellow",
                        padding=(1, 2)
                    )
                    self.console.print(examples_panel)
            else:
                print(f"\n📚 {concept.title()}")
                print("="*50)
                print(explanation)
                
                if examples:
                    print("\n💡 Examples:")
                    for i, example in enumerate(examples, 1):
                        print(f"  {i}. {example}")
        else:
            self.error_panel("❌ Explanation Failed", result.get("error", "Unknown error"))
    
    def show_session_stats(self):
        """Show session statistics"""
        session_time = time.time() - self.session_stats["start_time"]
        avg_time = (self.session_stats["total_thinking_time"] / 
                   max(1, self.session_stats["problems_solved"]))
        
        if RICH_AVAILABLE:
            stats_table = Table(show_header=True, header_style="bold cyan")
            stats_table.add_column("📊 Metric", style="cyan")
            stats_table.add_column("📈 Value", style="white")
            
            stats_table.add_row("Problems Solved", str(self.session_stats["problems_solved"]))
            stats_table.add_row("Session Time", f"{session_time:.1f}s")
            stats_table.add_row("Total Thinking Time", f"{self.session_stats['total_thinking_time']:.1f}s")
            stats_table.add_row("Avg. Time per Problem", f"{avg_time:.1f}s")
            
            stats_panel = Panel(
                stats_table,
                title="📊 Session Statistics",
                border_style="cyan",
                padding=(1, 2)
            )
            self.console.print(stats_panel)
        else:
            print("\n📊 Session Statistics")
            print("="*30)
            print(f"Problems Solved: {self.session_stats['problems_solved']}")
            print(f"Session Time: {session_time:.1f}s")
            print(f"Avg. Time per Problem: {avg_time:.1f}s")
    
    def show_help(self):
        """Show help information"""
        if RICH_AVAILABLE:
            help_table = Table(show_header=True, header_style="bold cyan")
            help_table.add_column("🔧 Command", style="cyan", width=20)
            help_table.add_column("📝 Description", style="white")
            
            help_table.add_row("solve <problem>", "Solve a mathematical problem")
            help_table.add_row("explain <concept>", "Explain a mathematical concept")
            help_table.add_row("verify <problem> <solution>", "Verify a proposed solution")
            help_table.add_row("stats", "Show session statistics")
            help_table.add_row("config", "Show/modify configuration")
            help_table.add_row("help", "Show this help message")
            help_table.add_row("quit/exit", "Exit MathBot")
            
            help_panel = Panel(
                help_table,
                title="🆘 MathBot Help",
                border_style="yellow",
                padding=(1, 2)
            )
            self.console.print(help_panel)
            
            # Show examples
            examples_text = Text()
            examples_text.append("💡 Examples:\n\n", style="bold yellow")
            examples_text.append("• ", style="yellow")
            examples_text.append("Find the derivative of x² + 3x - 5\n", style="white")
            examples_text.append("• ", style="yellow") 
            examples_text.append("Solve 2x + 5 = 13\n", style="white")
            examples_text.append("• ", style="yellow")
            examples_text.append("explain calculus\n", style="white")
            examples_text.append("• ", style="yellow")
            examples_text.append("verify '2x + 5 = 13' 'x = 4'\n", style="white")
            
            examples_panel = Panel(
                examples_text,
                title="💡 Usage Examples",
                border_style="green",
                padding=(1, 2)
            )
            self.console.print(examples_panel)
        else:
            print("\n🆘 MathBot Help")
            print("="*30)
            print("Commands:")
            print("  solve <problem>         - Solve a mathematical problem")
            print("  explain <concept>       - Explain a mathematical concept") 
            print("  verify <problem> <sol>  - Verify a proposed solution")
            print("  stats                   - Show session statistics")
            print("  help                    - Show this help")
            print("  quit/exit              - Exit MathBot")
            print("\nExamples:")
            print("  Find the derivative of x² + 3x - 5")
            print("  explain calculus")
            print("  verify '2x + 5 = 13' 'x = 4'")
    
    def error_panel(self, title: str, message: str):
        """Display error in a panel"""
        if RICH_AVAILABLE:
            error_panel = Panel(
                Text(message, style="bold red"),
                title=title,
                border_style="red",
                padding=(1, 2)
            )
            self.console.print(error_panel)
        else:
            print(f"\n❌ {title}")
            print(f"   {message}")
    
    def interactive_mode(self):
        """Run MathBot in interactive mode"""
        if RICH_AVAILABLE:
            welcome_text = Text()
            welcome_text.append("🤖 Welcome to MathBot Interactive Mode!\n\n", style="bold cyan")
            welcome_text.append("Type ", style="white")
            welcome_text.append("help", style="bold yellow")
            welcome_text.append(" for commands, ", style="white")
            welcome_text.append("quit", style="bold red")
            welcome_text.append(" to exit", style="white")
            
            welcome_panel = Panel(
                welcome_text,
                title="🎉 Interactive Mode",
                border_style="cyan",
                padding=(1, 2)
            )
            self.console.print(welcome_panel)
        else:
            print("\n🤖 MathBot Interactive Mode")
            print("Type 'help' for commands, 'quit' to exit\n")
        
        while True:
            try:
                if RICH_AVAILABLE:
                    user_input = Prompt.ask("🤖 [bold cyan]MathBot[/bold cyan]").strip()
                else:
                    user_input = input("🤖 MathBot> ").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command in ['quit', 'exit', 'q']:
                    if RICH_AVAILABLE:
                        goodbye_panel = Panel(
                            Align.center(Text("👋 Thanks for using MathBot!\nKeep solving! 🚀", style="bold cyan")),
                            title="Goodbye!",
                            border_style="cyan",
                            padding=(1, 2)
                        )
                        self.console.print(goodbye_panel)
                    else:
                        print("👋 Thanks for using MathBot!")
                    self.show_session_stats()
                    break
                
                elif command == 'help':
                    self.show_help()
                
                elif command == 'stats':
                    self.show_session_stats()
                
                elif command == 'explain':
                    if args:
                        self.explain_concept(args)
                    else:
                        self.error_panel("❌ Missing Argument", "Usage: explain <concept>")
                
                elif command == 'verify':
                    if args:
                        # Parse problem and solution
                        verify_parts = args.split("'")
                        if len(verify_parts) >= 4:
                            problem = verify_parts[1]
                            solution = verify_parts[3]
                            self.verify_solution(problem, solution)
                        else:
                            self.error_panel("❌ Invalid Format", "Usage: verify '<problem>' '<solution>'")
                    else:
                        self.error_panel("❌ Missing Arguments", "Usage: verify '<problem>' '<solution>'")
                
                elif command == 'solve':
                    if args:
                        self.solve_problem(args)
                    else:
                        self.error_panel("❌ Missing Problem", "Usage: solve <mathematical problem>")
                
                elif command == 'config':
                    self.show_config()
                
                else:
                    # Treat the entire input as a math problem
                    self.solve_problem(user_input)
                
                if RICH_AVAILABLE:
                    self.console.print()  # Add spacing
                else:
                    print()
                
            except KeyboardInterrupt:
                if RICH_AVAILABLE:
                    self.console.print("\n👋 Goodbye!")
                else:
                    print("\n👋 Goodbye!")
                break
            except Exception as e:
                self.error_panel("❌ Error", str(e))
    
    def verify_solution(self, problem: str, solution: str):
        """Verify a proposed solution"""
        if RICH_AVAILABLE:
            verify_text = Text()
            verify_text.append("Problem: ", style="bold blue")
            verify_text.append(f"{problem}\n", style="white")
            verify_text.append("Proposed Solution: ", style="bold green")
            verify_text.append(solution, style="white")
            
            verify_panel = Panel(
                verify_text,
                title="✅ Verification Request",
                border_style="yellow",
                padding=(1, 2)
            )
            self.console.print(verify_panel)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("🔍 Verifying solution...", total=None)
                result = self.agent.verify_solution(problem, solution)
                progress.update(task, description="✅ Verification complete!")
                time.sleep(0.2)
        else:
            print(f"\n✅ Verifying...")
            print(f"Problem: {problem}")
            print(f"Solution: {solution}")
            result = self.agent.verify_solution(problem, solution)
        
        if result.get("success", False):
            is_correct = result.get("is_correct", False)
            explanation = result.get("explanation", "")
            
            if RICH_AVAILABLE:
                status_text = Text()
                if is_correct:
                    status_text.append("✅ CORRECT!", style="bold green")
                else:
                    status_text.append("❌ INCORRECT", style="bold red")
                
                if explanation:
                    status_text.append(f"\n\n{explanation}", style="white")
                
                result_panel = Panel(
                    status_text,
                    title="🔍 Verification Result",
                    border_style="green" if is_correct else "red",
                    padding=(1, 2)
                )
                self.console.print(result_panel)
            else:
                status = "✅ CORRECT!" if is_correct else "❌ INCORRECT"
                print(f"\n{status}")
                if explanation:
                    print(f"Explanation: {explanation}")
        else:
            self.error_panel("❌ Verification Failed", result.get("error", "Unknown error"))
    
    def show_config(self):
        """Show current configuration"""
        if RICH_AVAILABLE:
            config_table = Table(show_header=True, header_style="bold cyan")
            config_table.add_column("⚙️ Setting", style="cyan")
            config_table.add_column("🔧 Value", style="white")
            
            config_table.add_row("Theme", self.config.theme)
            config_table.add_row("Show Thinking", "✅" if self.config.show_thinking else "❌")
            config_table.add_row("Show Steps", "✅" if self.config.show_step_by_step else "❌")
            config_table.add_row("Export Results", "✅" if self.config.export_results else "❌")
            config_table.add_row("Verbose Mode", "✅" if self.config.verbose else "❌")
            
            config_panel = Panel(
                config_table,
                title="⚙️ MathBot Configuration",
                border_style="cyan",
                padding=(1, 2)
            )
            self.console.print(config_panel)
        else:
            print("\n⚙️ MathBot Configuration")
            print("="*30)
            print(f"Theme: {self.config.theme}")
            print(f"Show Thinking: {'✅' if self.config.show_thinking else '❌'}")
            print(f"Show Steps: {'✅' if self.config.show_step_by_step else '❌'}")
            print(f"Export Results: {'✅' if self.config.export_results else '❌'}")
            print(f"Verbose Mode: {'✅' if self.config.verbose else '❌'}")


def main():
    """Main entry point for MathBot"""
    parser = argparse.ArgumentParser(
        description="🤖 MathBot - Beautiful AI-powered mathematics assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mathbot                                    # Interactive mode
  mathbot "solve x^2 + 5x - 6 = 0"         # Direct solve
  mathbot --explain "derivatives"            # Explain concept
  mathbot --verify "2x=6" "x=3"             # Verify solution
  
🎨 MathBot features beautiful ASCII art, colors, and step-by-step solutions!
        """
    )
    
    parser.add_argument(
        "problem",
        nargs="?",
        help="Mathematical problem to solve (if not provided, enters interactive mode)"
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
        "--no-color",
        action="store_true", 
        help="Disable colors and fancy formatting"
    )
    
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Hide thinking process"
    )
    
    parser.add_argument(
        "--no-steps",
        action="store_true", 
        help="Hide step-by-step solutions"
    )
    
    parser.add_argument(
        "--theme",
        choices=["cyan", "green", "blue", "purple", "red"],
        default="cyan",
        help="Color theme for the interface"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show session statistics and exit"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Create and configure MathBot
    bot = MathBot()
    bot.config.theme = args.theme
    bot.config.show_thinking = not args.no_thinking
    bot.config.show_step_by_step = not args.no_steps
    bot.config.verbose = args.verbose
    
    if args.no_color:
        global RICH_AVAILABLE
        RICH_AVAILABLE = False
    
    # Initialize MathBot
    bot.initialize()
    
    try:
        # Handle different modes
        if args.explain:
            bot.explain_concept(args.explain)
        
        elif args.verify:
            problem, solution = args.verify
            bot.verify_solution(problem, solution)
        
        elif args.problem:
            bot.solve_problem(args.problem)
        
        elif args.stats:
            bot.show_session_stats()
        
        else:
            # Interactive mode
            bot.interactive_mode()
    
    except Exception as e:
        bot.error_panel("❌ MathBot Error", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
