# Import individual tools
from .do_nothing import tool_definition as do_nothing_def, DoNothing
from .slow_clap import tool_definition as slow_clap_def, SlowClap
from .speak import tool_definition as speak_def, Speak
from .vision_look import tool_definition as vision_look_def, VisionLook

# Export all tool definitions
tool_definitions = [
    do_nothing_def,
    slow_clap_def,
    speak_def,
    vision_look_def,
]

# Export all tool classes
tool_classes = {
    "do_nothing": DoNothing,
    "slow clap": SlowClap,
    "speak": Speak,
    "vision_look": VisionLook,
}

# Export all tool names
all_tools = list(tool_classes.keys())
