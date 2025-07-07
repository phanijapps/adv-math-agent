#!/usr/bin/env python3
"""
🤖 MathBot - Streamlit Web Interface
A beautiful web interface for the MathBot mathematical problem solver
"""

import streamlit as st
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.math_agent import MathAgent, MathAgentConfig
from utils.helpers import load_environment, validate_api_keys, create_directories

# Configure Streamlit page
st.set_page_config(
    page_title="🤖 MathBot",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Problem box styling */
    .problem-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Solution box styling */
    .solution-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Error box styling */
    .error-box {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Explanation box styling */
    .explanation-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Stats box styling */
    .stats-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #667eea;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #667eea;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

class MathBotUI:
    """Streamlit UI for MathBot"""
    
    def __init__(self):
        self.agent = None
        self.session_stats = self._init_session_stats()
        
    def _init_session_stats(self):
        """Initialize session statistics"""
        if 'session_stats' not in st.session_state:
            st.session_state.session_stats = {
                'problems_solved': 0,
                'start_time': time.time(),
                'total_thinking_time': 0,
                'solutions_history': []
            }
        return st.session_state.session_stats
    
    def initialize_agent(self):
        """Initialize the MathBot agent"""
        if 'agent_initialized' not in st.session_state:
            with st.spinner('🧠 Initializing MathBot brain...'):
                try:
                    # Load environment
                    load_environment()
                    create_directories()
                    
                    # Validate API keys
                    api_validation = validate_api_keys()
                    if not api_validation.get("OPENROUTER_API_KEY", False):
                        st.error("🔑 OpenRouter API Key missing! Please set OPENROUTER_API_KEY in your .env file")
                        st.stop()
                    
                    # Initialize agent
                    config = MathAgentConfig()
                    config.verbose = False  # Disable verbose for UI
                    self.agent = MathAgent(config)
                    
                    st.session_state.agent_initialized = True
                    st.session_state.mathbot_agent = self.agent
                    
                    st.success("✅ MathBot initialized successfully!")
                    time.sleep(1)
                    
                except Exception as e:
                    st.error(f"❌ Failed to initialize MathBot: {e}")
                    st.stop()
        else:
            self.agent = st.session_state.mathbot_agent
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 MathBot</h1>
            <p>Your AI-Powered Mathematics Assistant</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_capabilities(self):
        """Render capabilities section"""
        st.markdown("### ✨ Capabilities")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown("""
            <div class="stats-box">
                <h4>🔢 Problem Solving</h4>
                <p>Complex mathematical problems</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="stats-box">
                <h4>🧮 Step-by-Step</h4>
                <p>Detailed explanations</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="stats-box">
                <h4>📚 Concepts</h4>
                <p>Mathematical concepts</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="stats-box">
                <h4>✅ Verification</h4>
                <p>Solution checking</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown("""
            <div class="stats-box">
                <h4>💡 Insights</h4>
                <p>Learning patterns</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_problem_input(self):
        """Render problem input section"""
        st.markdown("### 🔢 Enter Your Math Problem")
        
        # Problem input
        problem = st.text_area(
            "What mathematical problem would you like to solve?",
            placeholder="Enter your mathematical problem here...\nExample: Solve x^2 + 5x - 6 = 0",
            height=100,
            key="problem_input"
        )
        
        # Context input (optional)
        with st.expander("📝 Additional Context (Optional)"):
            context = st.text_area(
                "Provide any additional context or constraints:",
                placeholder="Any additional information about the problem...",
                height=80,
                key="context_input"
            )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            solve_button = st.button("🚀 Solve Problem", use_container_width=True)
        
        return problem, context, solve_button
    
    def solve_problem(self, problem: str, context: Optional[str] = None):
        """Solve a mathematical problem"""
        if not problem.strip():
            st.warning("⚠️ Please enter a mathematical problem to solve")
            return
        
        # Display problem
        st.markdown(f"""
        <div class="problem-box">
            <h3>🔢 Problem to Solve</h3>
            <p style="font-size: 1.2rem; margin: 0;">{problem}</p>
            {f'<p style="margin-top: 1rem;"><strong>Context:</strong> {context}</p>' if context else ''}
        </div>
        """, unsafe_allow_html=True)
        
        # Solve with progress
        start_time = time.time()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate thinking process
        thinking_steps = [
            "🤔 Analyzing problem structure...",
            "🔍 Identifying mathematical concepts...", 
            "🧮 Planning solution strategy...",
            "⚡ Computing solution...",
            "✅ Verifying results..."
        ]
        
        for i, step in enumerate(thinking_steps):
            status_text.text(step)
            progress_bar.progress((i + 1) / len(thinking_steps))
            time.sleep(0.5)
        
        # Actually solve the problem
        try:
            result = self.agent.solve(problem, context)
            solve_time = time.time() - start_time
            
            # Update session stats
            self.session_stats['problems_solved'] += 1
            self.session_stats['total_thinking_time'] += solve_time
            self.session_stats['solutions_history'].append({
                'problem': problem,
                'result': result,
                'timestamp': datetime.now().isoformat(),
                'solve_time': solve_time
            })
            
            # Clear progress
            progress_bar.empty()
            status_text.empty()
            
            # Display result
            self.display_solution(result, solve_time)
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            
            st.markdown(f"""
            <div class="error-box">
                <h3>❌ Solution Failed</h3>
                <p>Error: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def display_solution(self, result: Dict[str, Any], solve_time: float):
        """Display the solution with beautiful formatting"""
        if not result.get("success", False):
            st.markdown(f"""
            <div class="error-box">
                <h3>❌ Solution Failed</h3>
                <p>{result.get('error', 'Unknown error occurred')}</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        solution = result.get("solution", "No solution provided")
        confidence = result.get("confidence_score", 0)
        method = result.get("method", "unknown")
        
        # Main solution display
        st.markdown(f"""
        <div class="solution-box">
            <h3>🎯 FINAL ANSWER</h3>
            <div style="font-size: 1.1rem; line-height: 1.6; margin-top: 1rem;">
                {solution.replace('\n', '<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metadata section
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h4>⏱️ Solve Time</h4>
                <h2>{solve_time:.2f}s</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            confidence_pct = confidence * 100 if confidence > 0 else 0
            st.markdown(f"""
            <div class="metric-container">
                <h4>🎯 Confidence</h4>
                <h2>{confidence_pct:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <h4>🔧 Method</h4>
                <h2>{method.replace('_', ' ').title()}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <h4>🔢 Problem #</h4>
                <h2>{self.session_stats['problems_solved']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Export options
        with st.expander("📄 Export Solution"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Export as Text"):
                    self.export_solution(result, "txt")
            
            with col2:
                if st.button("📊 Export as JSON"):
                    self.export_solution(result, "json")
            
            with col3:
                if st.button("📝 Export as Markdown"):
                    self.export_solution(result, "md")
    
    def export_solution(self, result: Dict[str, Any], format_type: str):
        """Export solution in different formats"""
        problem = result.get("problem", "Unknown Problem")
        solution = result.get("solution", "No solution")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if format_type == "txt":
            content = f"""
MathBot Solution Export
=======================
Problem: {problem}
Solved on: {timestamp}

Solution:
{solution}

Method: {result.get('method', 'unknown')}
Confidence: {result.get('confidence_score', 0):.2%}
            """.strip()
            
            st.download_button(
                label="📄 Download Text File",
                data=content,
                file_name=f"mathbot_solution_{int(time.time())}.txt",
                mime="text/plain"
            )
        
        elif format_type == "json":
            content = json.dumps(result, indent=2)
            
            st.download_button(
                label="📊 Download JSON File",
                data=content,
                file_name=f"mathbot_solution_{int(time.time())}.json",
                mime="application/json"
            )
        
        elif format_type == "md":
            content = f"""
# MathBot Solution

**Problem:** {problem}  
**Solved on:** {timestamp}

## Solution

{solution}

---

**Method:** {result.get('method', 'unknown')}  
**Confidence:** {result.get('confidence_score', 0):.2%}  
**Generated by MathBot** 🤖
            """.strip()
            
            st.download_button(
                label="📝 Download Markdown File",
                data=content,
                file_name=f"mathbot_solution_{int(time.time())}.md",
                mime="text/markdown"
            )
    
    def render_sidebar(self):
        """Render the sidebar with tools and stats"""
        with st.sidebar:
            st.markdown("## 🛠️ MathBot Tools")
            
            # Session Statistics
            st.markdown("### 📊 Session Stats")
            session_time = time.time() - self.session_stats['start_time']
            avg_time = (self.session_stats['total_thinking_time'] / 
                       max(1, self.session_stats['problems_solved']))
            
            st.metric("Problems Solved", self.session_stats['problems_solved'])
            st.metric("Session Time", f"{session_time:.0f}s")
            st.metric("Avg. Solve Time", f"{avg_time:.1f}s")
            
            st.markdown("---")
            
            # Concept Explanation
            st.markdown("### 📚 Explain Concept")
            concept = st.text_input("Enter a mathematical concept:", 
                                   placeholder="e.g., derivatives, calculus")
            
            if st.button("🔍 Explain", use_container_width=True):
                if concept:
                    self.explain_concept(concept)
                else:
                    st.warning("Please enter a concept to explain")
            
            st.markdown("---")
            
            # Solution Verification
            st.markdown("### ✅ Verify Solution")
            verify_problem = st.text_input("Problem:", 
                                         placeholder="e.g., 2x + 5 = 13")
            verify_solution = st.text_input("Your solution:", 
                                          placeholder="e.g., x = 4")
            
            if st.button("🔎 Verify", use_container_width=True):
                if verify_problem and verify_solution:
                    self.verify_solution(verify_problem, verify_solution)
                else:
                    st.warning("Please enter both problem and solution")
            
            st.markdown("---")
            
            # Solution History
            if self.session_stats['solutions_history']:
                st.markdown("### 📝 Recent Solutions")
                for i, sol in enumerate(reversed(self.session_stats['solutions_history'][-3:])):
                    with st.expander(f"Problem {len(self.session_stats['solutions_history']) - i}"):
                        st.write(f"**Problem:** {sol['problem'][:50]}...")
                        st.write(f"**Time:** {sol['solve_time']:.2f}s")
                        st.write(f"**Success:** {'✅' if sol['result'].get('success') else '❌'}")
            
            st.markdown("---")
            
            # Settings
            st.markdown("### ⚙️ Settings")
            if st.button("🔄 Reset Session", use_container_width=True):
                self.reset_session()
            
            if st.button("💾 Clear History", use_container_width=True):
                self.clear_history()
    
    def explain_concept(self, concept: str):
        """Explain a mathematical concept"""
        with st.spinner(f"📖 Explaining {concept}..."):
            try:
                result = self.agent.explain_concept(concept)
                
                if result.get("success", False):
                    explanation = result.get("solution", "No explanation available")
                    
                    st.markdown(f"""
                    <div class="explanation-box">
                        <h3>📚 {concept.title()}</h3>
                        <div style="font-size: 1.1rem; line-height: 1.6;">
                            {explanation.replace('\n', '<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"❌ Failed to explain concept: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Error explaining concept: {e}")
    
    def verify_solution(self, problem: str, solution: str):
        """Verify a proposed solution"""
        with st.spinner("🔍 Verifying solution..."):
            try:
                result = self.agent.verify_solution(problem, solution)
                
                if result.get("success", False):
                    verification = result.get("solution", "No verification available")
                    
                    st.markdown(f"""
                    <div class="explanation-box">
                        <h3>✅ Verification Result</h3>
                        <p><strong>Problem:</strong> {problem}</p>
                        <p><strong>Your Solution:</strong> {solution}</p>
                        <div style="font-size: 1.1rem; line-height: 1.6; margin-top: 1rem;">
                            {verification.replace('\n', '<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"❌ Verification failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Error verifying solution: {e}")
    
    def reset_session(self):
        """Reset session statistics"""
        self.session_stats['problems_solved'] = 0
        self.session_stats['start_time'] = time.time()
        self.session_stats['total_thinking_time'] = 0
        self.session_stats['solutions_history'] = []
        st.success("🔄 Session reset successfully!")
        st.experimental_rerun()
    
    def clear_history(self):
        """Clear solution history"""
        self.session_stats['solutions_history'] = []
        st.success("💾 History cleared successfully!")
        st.experimental_rerun()
    
    def render_examples(self):
        """Render example problems"""
        st.markdown("### 💡 Example Problems")
        
        examples = [
            "Find the derivative of x² + 3x - 5",
            "Solve the equation 2x + 5 = 13",
            "Calculate the integral of sin(x) from 0 to π",
            "Factor the polynomial x² - 5x + 6",
            "Find the limit of (x² - 1)/(x - 1) as x approaches 1"
        ]
        
        cols = st.columns(len(examples))
        
        for i, example in enumerate(examples):
            with cols[i]:
                if st.button(f"📝 {example[:20]}...", key=f"example_{i}", use_container_width=True):
                    st.session_state.problem_input = example
                    st.experimental_rerun()
    
    def run(self):
        """Main application runner"""
        self.render_header()
        self.initialize_agent()
        self.render_capabilities()
        
        # Main content area
        problem, context, solve_button = self.render_problem_input()
        
        if solve_button and problem:
            self.solve_problem(problem, context)
        
        # Examples section
        self.render_examples()
        
        # Sidebar
        self.render_sidebar()


def main():
    """Main entry point"""
    app = MathBotUI()
    app.run()


if __name__ == "__main__":
    main()
