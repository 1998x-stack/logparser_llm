"""
LogParser-LLM: Efficient log parsing using LLM and prefix tree.

Main exports for easy access.
"""

# Version
__version__ = "0.1.0"

# Core components
from logparser_llm.core.parser import LogParserLLM
from logparser_llm.config_manager import (
    Config,
    LLMConfig,
    ParsingConfig,
    PrefixTreeConfig,
    MergingConfig,
    PreprocessingConfig,
    PerformanceConfig,
    StorageConfig,
    load_config,
    create_default_config,
)

# Models
from logparser_llm.models.log_entry import (
    LogEntry,
    Template,
    ParsedLog,
    ParsingStatistics,
)

# Utilities
from logparser_llm.utils.logger import setup_logger

__all__ = [
    # Core
    "LogParserLLM",
    
    # Config
    "Config",
    "LLMConfig",
    "ParsingConfig",
    "PrefixTreeConfig",
    "MergingConfig",
    "PreprocessingConfig",
    "PerformanceConfig",
    "StorageConfig",
    "load_config",
    "create_default_config",
    
    # Models
    "LogEntry",
    "Template",
    "ParsedLog",
    "ParsingStatistics",
    
    # Utils
    "setup_logger",
    
    # Version
    "__version__",
]