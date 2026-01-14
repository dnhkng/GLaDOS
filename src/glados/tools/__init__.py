# Import individual tools
from .slow_clap import tool_definition as slow_clap_def, SlowClap
from .vision_look import tool_definition as vision_look_def, VisionLook

# Export all tool definitions
tool_definitions = [
    slow_clap_def,
    vision_look_def,
]

# Export all tool classes
tool_classes = {
    "slow clap": SlowClap,
    "vision_look": VisionLook,
}

# Export all tool names
all_tools = list(tool_classes.keys())
