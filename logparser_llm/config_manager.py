"""
Configuration management for LogParser-LLM.
Centralized config loading and validation.
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    
    provider: str = Field(default="openai", description="LLM provider")
    model: str = Field(default="gpt-4-turbo-preview", description="Model name")
    api_key: str = Field(..., description="API key")
    api_base: Optional[str] = Field(None, description="API base URL")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, gt=0)
    timeout: int = Field(default=30, gt=0)
    max_retries: int = Field(default=3, ge=0)
    retry_delay: float = Field(default=1.0, ge=0.0)
    
    class Config:
        env_prefix = "LLM_"


class ParsingConfig(BaseModel):
    """Parsing configuration."""
    
    use_cache: bool = True
    cache_type: str = "memory"  # memory, redis, file
    cache_ttl: int = 86400
    batch_size: int = 20
    enable_batch_processing: bool = True
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_llm_cache: bool = True
    max_llm_calls_per_batch: int = 100
    use_icl: bool = False
    icl_shots: int = 5
    enable_ner: bool = True
    
    class Config:
        env_prefix = "PARSING_"


class PrefixTreeConfig(BaseModel):
    """Prefix tree configuration."""
    
    max_depth: int = Field(default=5, ge=1)
    min_cluster_size: int = Field(default=3, ge=1)
    token_delimiter: str = " "
    enable_fuzzy_matching: bool = True
    fuzzy_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    
    class Config:
        env_prefix = "TREE_"


class MergingConfig(BaseModel):
    """Template merging configuration."""
    
    enable_auto_merge: bool = True
    merge_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    max_edit_distance: int = Field(default=3, ge=0)
    check_semantic_similarity: bool = True
    
    class Config:
        env_prefix = "MERGE_"


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration."""
    
    remove_timestamps: bool = False
    remove_ip_addresses: bool = False
    normalize_numbers: bool = True
    normalize_paths: bool = True
    lowercase: bool = False
    remove_duplicates: bool = True
    
    class Config:
        env_prefix = "PREPROC_"


class PerformanceConfig(BaseModel):
    """Performance configuration."""
    
    max_workers: int = Field(default=4, ge=1)
    enable_async: bool = True
    memory_limit_mb: int = Field(default=2048, ge=256)
    queue_size: int = Field(default=1000, ge=1)
    
    class Config:
        env_prefix = "PERF_"


class StorageConfig(BaseModel):
    """Storage configuration."""
    
    templates_file: str = "data/templates.json"
    cache_directory: str = "cache/"
    enable_persistence: bool = True
    auto_save_interval: int = 300
    
    class Config:
        env_prefix = "STORAGE_"


class Config(BaseModel):
    """Main configuration class."""
    
    llm: LLMConfig
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    prefix_tree: PrefixTreeConfig = Field(default_factory=PrefixTreeConfig)
    merging: MergingConfig = Field(default_factory=MergingConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Config object
        """
        # Load environment variables first
        load_dotenv()
        
        # Read YAML file
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Replace environment variables in config
        config_dict = cls._replace_env_vars(config_dict)
        
        # Create config objects with defaults
        return cls(
            llm=LLMConfig(**config_dict.get('llm', {})),
            parsing=ParsingConfig(**config_dict.get('parsing', {})),
            prefix_tree=PrefixTreeConfig(**config_dict.get('prefix_tree', {})),
            merging=MergingConfig(**config_dict.get('merging', {})),
            preprocessing=PreprocessingConfig(**config_dict.get('preprocessing', {})),
            performance=PerformanceConfig(**config_dict.get('performance', {})),
            storage=StorageConfig(**config_dict.get('storage', {}))
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Load configuration from dictionary."""
        return cls(
            llm=LLMConfig(**config_dict.get('llm', {})),
            parsing=ParsingConfig(**config_dict.get('parsing', {})),
            prefix_tree=PrefixTreeConfig(**config_dict.get('prefix_tree', {})),
            merging=MergingConfig(**config_dict.get('merging', {})),
            preprocessing=PreprocessingConfig(**config_dict.get('preprocessing', {})),
            performance=PerformanceConfig(**config_dict.get('performance', {})),
            storage=StorageConfig(**config_dict.get('storage', {}))
        )
    
    @staticmethod
    def _replace_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Replace ${VAR} patterns with environment variables."""
        import re
        
        def replace_value(value):
            if isinstance(value, str):
                # Pattern: ${VAR_NAME} or ${VAR_NAME:default_value}
                pattern = r'\$\{([^:}]+)(?::([^}]+))?\}'
                
                def replacer(match):
                    var_name = match.group(1)
                    default_value = match.group(2)
                    env_value = os.getenv(var_name)
                    
                    if env_value is not None:
                        return env_value
                    elif default_value is not None:
                        return default_value
                    else:
                        return match.group(0)  # Keep original if no env var
                
                return re.sub(pattern, replacer, value)
            elif isinstance(value, dict):
                return {k: replace_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [replace_value(item) for item in value]
            return value
        
        return replace_value(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'llm': self.llm.dict(),
            'parsing': self.parsing.dict(),
            'prefix_tree': self.prefix_tree.dict(),
            'merging': self.merging.dict(),
            'preprocessing': self.preprocessing.dict(),
            'performance': self.performance.dict(),
            'storage': self.storage.dict()
        }
    
    def save_to_yaml(self, output_path: str):
        """Save configuration to YAML file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or environment variables.
    
    Priority:
    1. Provided config_path
    2. Default locations (./config/config.yaml, ./config.yaml)
    3. Environment variables only
    
    Args:
        config_path: Optional path to YAML configuration file
        
    Returns:
        Config object
        
    Example:
        >>> config = load_config("config/config.yaml")
        >>> config = load_config()  # Auto-detect
    """
    # Load .env file if exists
    load_dotenv()
    
    # If path provided, use it
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            return Config.from_yaml(str(config_file))
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Try default locations
    default_paths = [
        Path("config/config.yaml"),
        Path("config.yaml"),
        Path(__file__).parent.parent / "config" / "config.yaml",
    ]
    
    for path in default_paths:
        if path.exists():
            return Config.from_yaml(str(path))
    
    # Fall back to environment variables
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('LLM_API_KEY')
    if not api_key:
        raise ValueError(
            "No configuration file found and OPENAI_API_KEY not set. "
            "Please provide config file or set environment variables."
        )
    
    return Config(
        llm=LLMConfig(
            api_key=api_key,
            provider=os.getenv('LLM_PROVIDER', 'openai'),
            model=os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
        )
    )


def create_default_config(output_path: str = "config/config.yaml"):
    """
    Create a default configuration file.
    
    Args:
        output_path: Where to save the config file
    """
    config = Config(
        llm=LLMConfig(
            api_key="${OPENAI_API_KEY}",
            provider="openai",
            model="gpt-3.5-turbo"
        )
    )
    
    config.save_to_yaml(output_path)
    print(f"Default configuration created at: {output_path}")
    print("Please edit it and set your API key.")


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Load from YAML
    try:
        config = load_config("config/config.yaml")
        print("✓ Loaded from config/config.yaml")
        print(f"  Model: {config.llm.model}")
        print(f"  Batch size: {config.parsing.batch_size}")
    except FileNotFoundError:
        print("✗ Config file not found")
    
    # Example 2: Load from environment variables
    os.environ['OPENAI_API_KEY'] = 'test-key'
    config = load_config()
    print("\n✓ Loaded from environment variables")
    print(f"  Provider: {config.llm.provider}")
    
    # Example 3: Create default config
    create_default_config("config/config_example.yaml")