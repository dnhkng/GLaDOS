# Import individual tools
from .slow_clap import tool_definition as slow_clap_def, SlowClap

# Export all tool definitions
tool_definitions = [
    slow_clap_def,
]

# Export all tool classes
tool_classes = {
    "slow clap": SlowClap,
}

# Export all tool names
all_tools = list(tool_classes.keys())
