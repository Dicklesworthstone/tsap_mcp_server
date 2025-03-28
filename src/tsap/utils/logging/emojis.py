"""
Emoji definitions for TSAP logging system.

This module contains constants for emojis used in logging to provide visual cues
about the type and severity of log messages.
"""
from typing import Dict

# Log level emojis
INFO = "ℹ️"
DEBUG = "🔍"
WARNING = "⚠️"
ERROR = "❌"
CRITICAL = "🚨"
SUCCESS = "✅"
TRACE = "📍"

# Status emojis
RUNNING = "🔄"
PENDING = "⏳"
COMPLETED = "🏁"
FAILED = "👎"
STARTING = "🚀"
STOPPING = "🛑"
RESTARTING = "🔁"
LOADING = "📥"
SAVING = "📤"
CANCELLED = "🚫"
TIMEOUT = "⏱️"
SKIPPED = "⏭️"

# Operation emojis
SEARCH = "🔎"
ANALYZE = "🔬"
TRANSFORM = "🔀"
FILTER = "🧹"
EXTRACT = "📋"
COMPUTE = "🧮"
PROFILE = "📊"
OPTIMIZE = "⚡"
CACHE = "💾"
VALIDATE = "✔️"
COMPARE = "⚖️"
MERGE = "🔗"
SPLIT = "✂️"
UPDATE = "📝"
CONNECT = "🔌"
DISCONNECT = "🔌"

# Component emojis
CORE = "⚙️"
COMPOSITE = "🧩"
ANALYSIS = "🧠"
EVOLUTION = "🧬"
PLUGIN = "🔌"
TEMPLATE = "📋"
CACHE = "📦"
OUTPUT = "📤"
API = "🌐"
MCP = "📡"
PROJECT = "📂"
STORAGE = "💽"

# Tool emojis
RIPGREP = "🔍"
AWK = "🔧"
JQ = "🧰"
SQLITE = "🗃️"
HTML = "🌐"
PDF = "📄"
TABLE = "📊"

# Performance emojis
FAST = "🐇"
STANDARD = "🐢"
DEEP = "🐢"  # Same emoji but will use different colors
MEMORY = "🧠"
CPU = "💻"
DISK = "💽"
NETWORK = "🌐"

# Result emojis
FOUND = "🎯"
NOT_FOUND = "🔍"
PARTIAL = "◐"
UNKNOWN = "❓"
HIGH_CONFIDENCE = "🔒"
MEDIUM_CONFIDENCE = "🔓"
LOW_CONFIDENCE = "🚪"

# Evolution emojis
EVOLVING = "🌱"
EVOLVED = "🌳"
MUTATION = "🧬"
SELECTION = "👍"
CROSSOVER = "✂️"
GENERATION = "👪"
IMPROVEMENT = "📈"
REGRESSION = "📉"
PLATEAU = "➡️"

# System emojis
STARTUP = "🔆"
SHUTDOWN = "🔅"
CONFIG = "⚙️"
ERROR = "⛔"
WARNING = "⚠️"
PLUGIN_LOAD = "🔌"
PLUGIN_UNLOAD = "🔌"
DEPENDENCY = "🧱"
VERSION = "🏷️"
UPDATE_AVAILABLE = "🆕"

# User interaction emojis
INPUT = "⌨️"
OUTPUT = "📺"
HELP = "❓"
HINT = "💡"
EXAMPLE = "📋"
QUESTION = "❓"
ANSWER = "💬"

# Time emojis
TIMING = "⏱️"
SCHEDULED = "📅"
DELAYED = "⏰"
OVERTIME = "⌛"

# Convenience mapping for log levels
LEVEL_EMOJIS: Dict[str, str] = {
    "info": INFO,
    "debug": DEBUG,
    "warning": WARNING,
    "error": ERROR,
    "critical": CRITICAL,
    "success": SUCCESS,
    "trace": TRACE,
}

# Get emoji by name function for more dynamic access
def get_emoji(category: str, name: str) -> str:
    """Get an emoji by category and name.
    
    Args:
        category: The category of emoji (e.g., 'level', 'status', 'operation')
        name: The name of the emoji within that category
    
    Returns:
        The emoji string or a default '?' if not found
    """
    # Convert to uppercase for constant lookup
    name_upper = name.upper()
    
    # Handle special case for log levels
    if category.lower() == "level":
        return LEVEL_EMOJIS.get(name.lower(), "?")
    
    # Get the module global variables
    globals_dict = globals()
    
    # Try direct lookup first
    if name_upper in globals_dict:
        return globals_dict[name_upper]
    
    # Return question mark for unknown emojis
    return "❓"