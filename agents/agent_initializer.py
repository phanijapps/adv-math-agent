"""
Agent initialization and setup utilities
"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Conditional imports for LangChain
try:
    from langchain import hub
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain.schema import BaseMessage, HumanMessage, SystemMessage
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.callbacks import StreamingStdOutCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not available. Agent will use fallback implementation.")
    LANGCHAIN_AVAILABLE = False


class AgentInitializer:
    """Handles initialization of LangChain agents and related components."""
    
    @staticmethod
    def create_langchain_agent(
        model_manager, 
        tools: List, 
        config
    ) -> Optional[object]:
        """Create the LangChain ReAct agent following user's preferred pattern."""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available, skipping agent creation")
            return None
        
        try:
            # Get the primary model
            llm = model_manager.get_model("primary")
            
            # Load the ReAct prompt from LangChain Hub
            prompt = hub.pull("hwchase17/react")
            
            # Create the agent
            agent = create_react_agent(
                llm=llm.llm,  # Use the underlying ChatOpenAI instance
                tools=tools,
                prompt=prompt
            )
            
            # Create memory
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                k=10,  # Remember last 10 interactions
                return_messages=True
            ) if config.memory_enabled else None
            
            # Create agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                memory=memory,
                verbose=config.verbose,
                max_iterations=config.max_iterations,
                handle_parsing_errors=True,
                callbacks=[StreamingStdOutCallbackHandler()] if config.verbose else None
            )
            
            return agent_executor
            
        except Exception as e:
            logger.error(f"Error creating LangChain agent: {e}")
            return None
    
    @staticmethod
    def load_tools() -> List:
        """Load all available mathematical tools."""
        try:
            from tools import get_all_tools
            tools = get_all_tools()
            logger.info(f"Loaded {len(tools)} mathematical tools")
            return tools
        except Exception as e:
            logger.error(f"Error loading tools: {e}")
            return []
    
    @staticmethod
    def validate_setup(model_manager, memory_manager, tools) -> Dict[str, bool]:
        """Validate that all components are properly initialized."""
        validation = {
            "model_manager": model_manager is not None,
            "memory_manager": memory_manager is not None,
            "tools_loaded": len(tools) > 0,
            "langchain_available": LANGCHAIN_AVAILABLE
        }
        
        # Log validation results
        for component, status in validation.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"{status_icon} {component}: {'OK' if status else 'FAILED'}")
        
        return validation
