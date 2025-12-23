"""
Data models for log entries and parsed results.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
import hashlib


class LogEntry(BaseModel):
    """Raw log entry model."""
    
    content: str = Field(..., description="Original log content")
    timestamp: Optional[datetime] = Field(None, description="Log timestamp")
    log_level: Optional[str] = Field(None, description="Log level (INFO, ERROR, etc.)")
    source: Optional[str] = Field(None, description="Log source/component")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('content')
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Log content cannot be empty")
        return v.strip()
    
    def get_hash(self) -> str:
        """Generate unique hash for the log content."""
        return hashlib.md5(self.content.encode()).hexdigest()
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "2024-01-01 10:00:00 ERROR Failed to connect to database",
                "timestamp": "2024-01-01T10:00:00",
                "log_level": "ERROR",
                "source": "database.connector"
            }
        }


class Template(BaseModel):
    """Log template model."""
    
    template_id: str = Field(..., description="Unique template identifier")
    template_pattern: str = Field(..., description="Template pattern with placeholders")
    static_tokens: List[str] = Field(default_factory=list, description="Static tokens")
    variable_positions: List[int] = Field(default_factory=list, description="Variable token positions")
    example_logs: List[str] = Field(default_factory=list, description="Example logs matching this template")
    count: int = Field(default=1, description="Number of logs matched")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Template confidence score")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @validator('template_pattern')
    def pattern_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Template pattern cannot be empty")
        return v.strip()
    
    def to_regex_pattern(self) -> str:
        """Convert template to regex pattern for matching."""
        import re
        pattern = re.escape(self.template_pattern)
        pattern = pattern.replace(r'\<\*\>', r'.*?')
        return f"^{pattern}$"
    
    class Config:
        json_schema_extra = {
            "example": {
                "template_id": "tmpl_001",
                "template_pattern": "Failed to connect to <*>",
                "static_tokens": ["Failed", "to", "connect", "to"],
                "variable_positions": [4],
                "example_logs": ["Failed to connect to database"],
                "count": 5,
                "confidence": 0.95
            }
        }


class ParsedLog(BaseModel):
    """Parsed log result model."""
    
    original: str = Field(..., description="Original log content")
    template_id: str = Field(..., description="Matched template ID")
    template_pattern: str = Field(..., description="Template pattern")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Extracted variables")
    timestamp: Optional[datetime] = Field(None, description="Parsed timestamp")
    log_level: Optional[str] = Field(None, description="Parsed log level")
    component: Optional[str] = Field(None, description="Component/source")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Parsing confidence")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Processing metadata
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")
    used_llm: bool = Field(default=False, description="Whether LLM was used")
    cache_hit: bool = Field(default=False, description="Whether result was from cache")
    
    def to_structured_dict(self) -> Dict[str, Any]:
        """Convert to structured dictionary for analysis."""
        return {
            "template_id": self.template_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "log_level": self.log_level,
            "component": self.component,
            "variables": self.variables,
            "confidence": self.confidence
        }
    
    class Config:
        json_schema_extra = {
            "example": {
                "original": "2024-01-01 10:00:00 ERROR Failed to connect to database",
                "template_id": "tmpl_001",
                "template_pattern": "<*> <*> <*> Failed to connect to <*>",
                "variables": {
                    "timestamp": "2024-01-01 10:00:00",
                    "level": "ERROR",
                    "target": "database"
                },
                "log_level": "ERROR",
                "confidence": 0.95,
                "used_llm": True,
                "cache_hit": False
            }
        }


class ParsingStatistics(BaseModel):
    """Statistics for parsing operations."""
    
    total_logs: int = 0
    successfully_parsed: int = 0
    failed: int = 0
    unique_templates: int = 0
    llm_calls: int = 0
    cache_hits: int = 0
    total_processing_time_ms: float = 0.0
    average_confidence: float = 0.0
    cost_usd: float = 0.0
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_logs == 0:
            return 0.0
        return self.cache_hits / self.total_logs
    
    def get_success_rate(self) -> float:
        """Calculate parsing success rate."""
        if self.total_logs == 0:
            return 0.0
        return self.successfully_parsed / self.total_logs
    
    def get_llm_efficiency(self) -> float:
        """Calculate LLM efficiency (logs per LLM call)."""
        if self.llm_calls == 0:
            return float('inf')
        return self.total_logs / self.llm_calls
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with computed metrics."""
        return {
            **self.dict(),
            "cache_hit_rate": self.get_cache_hit_rate(),
            "success_rate": self.get_success_rate(),
            "llm_efficiency": self.get_llm_efficiency(),
            "avg_time_per_log_ms": (
                self.total_processing_time_ms / self.total_logs 
                if self.total_logs > 0 else 0.0
            )
        }