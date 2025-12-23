"""
Configuration management for LogParser-LLM.
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from dotenv import load_dotenv


class LLMConfig(BaseSettings):
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


class ParsingConfig(BaseSettings):
    """Parsing configuration."""
    
    use_cache: bool = True
    cache_type: str = "memory"
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


class PrefixTreeConfig(BaseSettings):
    """Prefix tree configuration."""
    
    max_depth: int = Field(default=5, ge=1)
    min_cluster_size: int = Field(default=3, ge=1)
    token_delimiter: str = " "
    enable_fuzzy_matching: bool = True
    fuzzy_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    
    class Config:
        env_prefix = "TREE_"


class MergingConfig(BaseSettings):
    """Template merging configuration."""
    
    enable_auto_merge: bool = True
    merge_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    max_edit_distance: int = Field(default=3, ge=0)
    check_semantic_similarity: bool = True
    
    class Config:
        env_prefix = "MERGE_"


class PerformanceConfig(BaseSettings):
    """Performance configuration."""
    
    max_workers: int = Field(default=4, ge=1)
    enable_async: bool = True
    memory_limit_mb: int = Field(default=2048, ge=256)
    queue_size: int = Field(default=1000, ge=1)
    
    class Config:
        env_prefix = "PERF_"


class Config(BaseSettings):
    """Main configuration class."""
    
    llm: LLMConfig
    parsing: ParsingConfig
    prefix_tree: PrefixTreeConfig
    merging: MergingConfig
    performance: PerformanceConfig
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        # Load environment variables
        load_dotenv()
        
        # Read YAML file
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Replace environment variables
        config_dict = cls._replace_env_vars(config_dict)
        
        # Create config objects
        return cls(
            llm=LLMConfig(**config_dict.get('llm', {})),
            parsing=ParsingConfig(**config_dict.get('parsing', {})),
            prefix_tree=PrefixTreeConfig(**config_dict.get('prefix_tree', {})),
            merging=MergingConfig(**config_dict.get('merging', {})),
            performance=PerformanceConfig(**config_dict.get('performance', {}))
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Load configuration from dictionary."""
        return cls(
            llm=LLMConfig(**config_dict.get('llm', {})),
            parsing=ParsingConfig(**config_dict.get('parsing', {})),
            prefix_tree=PrefixTreeConfig(**config_dict.get('prefix_tree', {})),
            merging=MergingConfig(**config_dict.get('merging', {})),
            performance=PerformanceConfig(**config_dict.get('performance', {}))
        )
    
    @staticmethod
    def _replace_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Replace ${VAR} patterns with environment variables."""
        import re
        
        def replace_value(value):
            if isinstance(value, str):
                # Pattern: ${VAR_NAME}
                pattern = r'\$\{([^}]+)\}'
                matches = re.findall(pattern, value)
                for match in matches:
                    env_value = os.getenv(match)
                    if env_value is not None:
                        value = value.replace(f'${{{match}}}', env_value)
                return value
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
            'performance': self.performance.dict()
        }
    
    def save_to_yaml(self, output_path: str):
        """Save configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or environment variables.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object
    """
    if config_path and Path(config_path).exists():
        return Config.from_yaml(config_path)
    
    # Try default locations
    default_paths = [
        'config/config.yaml',
        'config.yaml',
        '../config/config.yaml'
    ]
    
    for path in default_paths:
        if Path(path).exists():
            return Config.from_yaml(path)
    
    # Fall back to environment variables
    load_dotenv()
    
    return Config(
        llm=LLMConfig(api_key=os.getenv('OPENAI_API_KEY', '')),
        parsing=ParsingConfig(),
        prefix_tree=PrefixTreeConfig(),
        merging=MergingConfig(),
        performance=PerformanceConfig()
    )


# Example usage and testing
if __name__ == "__main__":
    # Create default config
    config = Config(
        llm=LLMConfig(api_key="test-key"),
        parsing=ParsingConfig(),
        prefix_tree=PrefixTreeConfig(),
        merging=MergingConfig(),
        performance=PerformanceConfig()
    )
    
    print("Configuration loaded successfully:")
    print(f"LLM Provider: {config.llm.provider}")
    print(f"Model: {config.llm.model}")
    print(f"Batch Size: {config.parsing.batch_size}")
    print(f"Cache Enabled: {config.parsing.use_cache}")