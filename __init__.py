"""
Advanced Math Agent Package
"""
__version__ = "1.0.0"
__author__ = "Math Agent Developer"
__description__ = "Advanced mathematical problem-solving agent using LangChain and OpenRouter"

from .agents import MathAgent, MathAgentConfig
from .memory import MemoryManager
from .llm import OpenRouterLLM, ModelManager

__all__ = [
    "MathAgent", 
    "MathAgentConfig",
    "MemoryManager",
    "OpenRouterLLM",
    "ModelManager"
]
