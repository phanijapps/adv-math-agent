"""
Tools module initialization
"""
# Conditional imports to handle missing dependencies gracefully

def get_all_tools():
    """Get all available mathematical tools."""
    tools = []
    
    try:
        from .math_tools import get_math_tools
        tools.extend(get_math_tools())
    except ImportError as e:
        print(f"Warning: Core math tools not available: {e}")
    except Exception as e:
        print(f"Warning: Error loading math tools: {e}")
        # Return empty list if tools fail to load due to Pydantic issues
        return []
    
    try:
        from .extended_tools import get_extended_math_tools
        tools.extend(get_extended_math_tools())
    except ImportError as e:
        print(f"Warning: Extended math tools not available: {e}")
    except Exception as e:
        print(f"Warning: Error loading extended tools: {e}")
    
    return tools

# Export main functions
__all__ = ["get_all_tools"]
