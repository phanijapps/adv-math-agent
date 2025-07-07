"""
Configuration module for MathAgent
"""
import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class MathAgentConfig:
    """Configuration for the Math Agent."""
    
    model: str = None
    temperature: float = 0.1
    max_iterations: int = 10
    verbose: bool = True
    memory_enabled: bool = True
    user_id: str = "default_user"
    
    def __post_init__(self):
        """Initialize configuration from environment variables if not provided."""
        if self.model is None:
            self.model = os.getenv("DEFAULT_MODEL", "openai/gpt-4-turbo-preview")
        
        # Override with environment variables if they exist
        self.temperature = float(os.getenv("TEMPERATURE", str(self.temperature)))
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", str(self.max_iterations)))
        self.verbose = os.getenv("VERBOSE", str(self.verbose).lower()).lower() == "true"
        self.memory_enabled = os.getenv("MEMORY_ENABLED", str(self.memory_enabled).lower()).lower() == "true"
        self.user_id = os.getenv("USER_ID", self.user_id)
    
    @classmethod
    def create_default(cls) -> "MathAgentConfig":
        """Create default configuration."""
        return cls()
    
    @classmethod
    def create_quiet(cls) -> "MathAgentConfig":
        """Create configuration with minimal output."""
        return cls(verbose=False)
    
    @classmethod
    def create_fast(cls) -> "MathAgentConfig":
        """Create configuration optimized for speed."""
        return cls(
            temperature=0.0,
            max_iterations=5,
            verbose=False
        )
