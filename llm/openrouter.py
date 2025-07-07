"""
OpenRouter LLM Integration for Math Agent
"""
from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage
import os
import logging

logger = logging.getLogger(__name__)


class OpenRouterLLM:
    """
    OpenRouter LLM integration following the pattern from user's memories.
    Provides a clean interface to OpenRouter's API through LangChain.
    """
    
    def __init__(
        self,
        model: str = "google/gemini-2.5-flash",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenRouter LLM.
        
        Args:
            model: Model name (e.g., "google/gemini-2.0-flash-001")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            api_key: OpenRouter API key
            **kwargs: Additional parameters
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
        
        # Initialize the ChatOpenAI client with OpenRouter configuration
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=self.api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            **kwargs
        )
        
        logger.info(f"Initialized OpenRouter LLM with model: {model}")
    
    def invoke(self, messages: List[BaseMessage], **kwargs) -> Any:
        """
        Invoke the LLM with messages.
        
        Args:
            messages: List of messages to send
            **kwargs: Additional parameters
            
        Returns:
            LLM response
        """
        try:
            return self.llm.invoke(messages, **kwargs)
        except Exception as e:
            logger.error(f"Error invoking OpenRouter LLM: {e}")
            raise
    
    def stream(self, messages: List[BaseMessage], **kwargs):
        """
        Stream responses from the LLM.
        
        Args:
            messages: List of messages to send
            **kwargs: Additional parameters
            
        Yields:
            Streamed response chunks
        """
        try:
            for chunk in self.llm.stream(messages, **kwargs):
                yield chunk
        except Exception as e:
            logger.error(f"Error streaming from OpenRouter LLM: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Model information dictionary
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "provider": "OpenRouter"
        }
    
    @classmethod
    def create_math_optimized(cls, **kwargs) -> "OpenRouterLLM":
        """
        Create an OpenRouter LLM instance optimized for mathematical reasoning.
        
        Returns:
            Configured OpenRouterLLM instance
        """
        default_config = {
            "model": "openai/gpt-4-turbo-preview",
            "temperature": 0.1,  # Lower temperature for consistent math
            "max_tokens": 4000,
        }
        default_config.update(kwargs)
        
        return cls(**default_config)


class ModelManager:
    """
    Manages multiple models and provides fallback capabilities.
    """
    
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.fallback_model = None
        
        # Initialize default models
        self._setup_default_models()
    
    def _setup_default_models(self):
        """Setup default models from environment."""
        try:
            primary_model = os.getenv("DEFAULT_MODEL", "openai/gpt-4-turbo-preview")
            fallback_model = os.getenv("FALLBACK_MODEL", "openai/gpt-3.5-turbo")
            
            self.add_model("primary", primary_model)
            self.add_model("fallback", fallback_model)
            
            self.current_model = "primary"
            self.fallback_model = "fallback"
            
            logger.info(f"Setup models - Primary: {primary_model}, Fallback: {fallback_model}")
            
        except Exception as e:
            logger.error(f"Error setting up default models: {e}")
    
    def add_model(self, name: str, model_name: str, **kwargs):
        """
        Add a model to the manager.
        
        Args:
            name: Model identifier
            model_name: OpenRouter model name
            **kwargs: Additional model parameters
        """
        try:
            self.models[name] = OpenRouterLLM(model=model_name, **kwargs)
            logger.info(f"Added model '{name}': {model_name}")
        except Exception as e:
            logger.error(f"Error adding model '{name}': {e}")
    
    def get_model(self, name: Optional[str] = None) -> OpenRouterLLM:
        """
        Get a model by name or current model.
        
        Args:
            name: Model name (optional)
            
        Returns:
            OpenRouterLLM instance
        """
        if name is None:
            name = self.current_model
        
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found")
        
        return self.models[name]
    
    def switch_model(self, name: str):
        """
        Switch to a different model.
        
        Args:
            name: Model name to switch to
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found")
        
        self.current_model = name
        logger.info(f"Switched to model: {name}")
    
    def invoke_with_fallback(self, messages: List[BaseMessage], **kwargs) -> Any:
        """
        Invoke with automatic fallback on failure.
        
        Args:
            messages: Messages to send
            **kwargs: Additional parameters
            
        Returns:
            LLM response
        """
        try:
            # Try primary model
            primary = self.get_model(self.current_model)
            return primary.invoke(messages, **kwargs)
        
        except Exception as e:
            logger.warning(f"Primary model failed: {e}")
            
            if self.fallback_model and self.fallback_model in self.models:
                try:
                    # Try fallback model
                    fallback = self.get_model(self.fallback_model)
                    logger.info("Using fallback model")
                    return fallback.invoke(messages, **kwargs)
                
                except Exception as fe:
                    logger.error(f"Fallback model also failed: {fe}")
                    raise fe
            else:
                raise e
