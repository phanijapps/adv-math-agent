# Advanced Math Agent

An advanced mathematical problem-solving agent built with LangChain and OpenRouter, featuring sophisticated mathematical reasoning capabilities, symbolic computation, and interactive problem solving.

## Features

- **Advanced Mathematical Reasoning**: Solves complex calculus, algebra, statistics, and discrete mathematics problems
- **Symbolic Computation**: Uses SymPy for exact mathematical computations
- **Data Visualization**: Creates mathematical plots and visualizations
- **Memory Integration**: Remembers previous solutions and learning patterns
- **Interactive Problem Solving**: Step-by-step solution explanations
- **Multi-Modal Support**: Text, equations, and visual representations

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the agent:
```bash
python main.py
```

## Architecture

- `main.py`: Main application entry point
- `agents/`: Math agent implementations
- `tools/`: Mathematical computation tools
- `llm/`: OpenRouter LLM integration
- `memory/`: Memory management system
- `utils/`: Utility functions

## Usage

The agent can solve various types of mathematical problems:

- **Calculus**: Derivatives, integrals, limits, series
- **Algebra**: Equation solving, factorization, simplification
- **Statistics**: Probability, distributions, hypothesis testing
- **Geometry**: Coordinate geometry, trigonometry
- **Linear Algebra**: Matrix operations, eigenvalues, vector spaces
- **Discrete Math**: Combinatorics, graph theory, number theory

## Examples

```python
from agents.math_agent import MathAgent

agent = MathAgent()

# Solve calculus problem
result = agent.solve("Find the derivative of x^3 + 2x^2 - 5x + 1")

# Solve algebra problem
result = agent.solve("Solve the system: 2x + 3y = 7, x - y = 1")

# Data analysis
result = agent.solve("Plot the function f(x) = x^2 - 4x + 3")
```
