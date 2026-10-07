from .core_controls import ManageMemory, CancelTask, memory_definition, cancel_definition
# Import individual tools
from .get_report import GetReport
from .get_report import tool_definition as get_report_def
from .get_time import GetTime
from .get_time import tool_definition as get_time_def
from .manage_slot import ManageSlot
from .manage_slot import tool_definition as manage_slot_def
from .safe_command import RunSafeCommand
from .safe_command import tool_definition as safe_command_def
from .preferences import (
    GetPreferences,
    SetPreference,
    get_preferences_definition,
    set_preference_definition,
)
from .slow_clap import SlowClap
from .slow_clap import tool_definition as slow_clap_def
from .vision_look import VisionLook
from .vision_look import tool_definition as vision_look_def

# Export all tool definitions
tool_definitions = [
    memory_definition, cancel_definition,
    safe_command_def,
    get_time_def,
    manage_slot_def,
    get_report_def,
    slow_clap_def,
    vision_look_def,
    get_preferences_definition,
    set_preference_definition,
]

# Export all tool classes
tool_classes = {
    "manage_memory": ManageMemory, "cancel_task": CancelTask,
    "run_safe_command": RunSafeCommand,
    "get_time": GetTime,
    "get_report": GetReport,
    "slow clap": SlowClap,
    "vision_look": VisionLook,
    "get_preferences": GetPreferences,
    "set_preference": SetPreference,
    "manage_slot": ManageSlot,
}

# Export all tool names
all_tools = list(tool_classes.keys())
