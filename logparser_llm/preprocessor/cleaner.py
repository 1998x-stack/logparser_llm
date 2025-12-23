"""
Log preprocessing and cleaning module.
"""
import re
from typing import List, Optional, Tuple
from logparser_llm.config_manager import PreprocessingConfig


class LogCleaner:
    """Clean and normalize log messages."""
    
    # Common regex patterns
    PATTERNS = {
        'timestamp': [
            r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?',
            r'\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2}',
            r'\w{3}\s+\d{1,2}\s\d{2}:\d{2}:\d{2}',
        ],
        'ip': r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
        'ipv6': r'(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}',
        'mac': r'(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}',
        'url': r'https?://[^\s]+',
        'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'path': r'(?:/[a-zA-Z0-9._-]+)+/?',
        'uuid': r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
        'hex': r'0x[0-9a-fA-F]+',
        'number': r'\b\d+(?:\.\d+)?\b',
    }
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize log cleaner.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        
    def clean(self, log: str) -> str:
        """
        Clean and normalize a log message.
        
        Args:
            log: Raw log message
            
        Returns:
            Cleaned log message
        """
        if not log or not log.strip():
            return ""
        
        cleaned = log.strip()
        
        # Remove timestamps if configured
        if self.config.remove_timestamps:
            cleaned = self._remove_timestamps(cleaned)
        
        # Remove IP addresses if configured
        if self.config.remove_ip_addresses:
            cleaned = self._remove_ips(cleaned)
        
        # Normalize numbers
        if self.config.normalize_numbers:
            cleaned = self._normalize_numbers(cleaned)
        
        # Normalize paths
        if self.config.normalize_paths:
            cleaned = self._normalize_paths(cleaned)
        
        # Lowercase if configured
        if self.config.lowercase:
            cleaned = cleaned.lower()
        
        # Clean whitespace
        cleaned = self._clean_whitespace(cleaned)
        
        return cleaned
    
    def clean_batch(self, logs: List[str]) -> List[str]:
        """
        Clean multiple logs.
        
        Args:
            logs: List of log messages
            
        Returns:
            List of cleaned logs
        """
        cleaned = [self.clean(log) for log in logs]
        
        # Remove duplicates if configured
        if self.config.remove_duplicates:
            cleaned = list(dict.fromkeys(cleaned))  # Preserve order
        
        return cleaned
    
    def extract_timestamp(self, log: str) -> Optional[str]:
        """
        Extract timestamp from log message.
        
        Args:
            log: Log message
            
        Returns:
            Extracted timestamp or None
        """
        for pattern in self.PATTERNS['timestamp']:
            match = re.search(pattern, log)
            if match:
                return match.group(0)
        return None
    
    def extract_log_level(self, log: str) -> Optional[str]:
        """
        Extract log level (INFO, ERROR, DEBUG, etc.).
        
        Args:
            log: Log message
            
        Returns:
            Log level or None
        """
        levels = ['TRACE', 'DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR', 'FATAL', 'CRITICAL']
        pattern = r'\b(' + '|'.join(levels) + r')\b'
        match = re.search(pattern, log, re.IGNORECASE)
        return match.group(0).upper() if match else None
    
    def split_log_components(self, log: str) -> Tuple[Optional[str], Optional[str], str]:
        """
        Split log into timestamp, level, and message.
        
        Args:
            log: Log message
            
        Returns:
            Tuple of (timestamp, level, message)
        """
        timestamp = self.extract_timestamp(log)
        level = self.extract_log_level(log)
        
        # Remove timestamp and level from message
        message = log
        if timestamp:
            message = message.replace(timestamp, '').strip()
        if level:
            message = re.sub(r'\b' + re.escape(level) + r'\b', '', message, flags=re.IGNORECASE).strip()
        
        return timestamp, level, message
    
    def _remove_timestamps(self, text: str) -> str:
        """Remove all timestamps from text."""
        for pattern in self.PATTERNS['timestamp']:
            text = re.sub(pattern, '', text)
        return text
    
    def _remove_ips(self, text: str) -> str:
        """Remove IP addresses from text."""
        text = re.sub(self.PATTERNS['ip'], '', text)
        text = re.sub(self.PATTERNS['ipv6'], '', text)
        return text
    
    def _normalize_numbers(self, text: str) -> str:
        """Normalize numbers to a placeholder."""
        # Keep numbers that are likely part of log structure (e.g., "ERROR404")
        # Replace standalone numbers
        text = re.sub(r'\b\d+\.\d+\b', '<NUM>', text)  # Decimals
        text = re.sub(r'\b\d{4,}\b', '<NUM>', text)  # Large numbers
        return text
    
    def _normalize_paths(self, text: str) -> str:
        """Normalize file paths."""
        # Unix/Linux paths
        text = re.sub(r'/(?:[a-zA-Z0-9._-]+/)+[a-zA-Z0-9._-]+', '<PATH>', text)
        # Windows paths
        text = re.sub(r'[A-Z]:\\(?:[^\\]+\\)+[^\\]+', '<PATH>', text)
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean and normalize whitespace."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def is_valid_log(log: str) -> bool:
        """
        Check if a string is a valid log message.
        
        Args:
            log: String to validate
            
        Returns:
            True if valid
        """
        if not log or not log.strip():
            return False
        
        # Must have some meaningful content
        if len(log.strip()) < 10:
            return False
        
        # Should not be all special characters
        if re.match(r'^[\W_]+$', log.strip()):
            return False
        
        return True


# Example usage
if __name__ == "__main__":
    from logparser_llm.config_manager import PreprocessingConfig
    
    # Create cleaner with config
    config = PreprocessingConfig(
        remove_timestamps=True,
        normalize_numbers=True,
        normalize_paths=True
    )
    cleaner = LogCleaner(config)
    
    # Test logs
    test_logs = [
        "2024-01-01 10:00:00 INFO User 12345 logged in from 192.168.1.1",
        "2024-01-01 10:05:00 ERROR Failed to connect to database on port 5432",
        "2024-01-01 10:10:00 DEBUG Reading file /var/log/app.log",
    ]
    
    print("Original logs:")
    for log in test_logs:
        print(f"  {log}")
    
    print("\nCleaned logs:")
    cleaned = cleaner.clean_batch(test_logs)
    for log in cleaned:
        print(f"  {log}")
    
    print("\nExtracted components:")
    for log in test_logs:
        ts, level, msg = cleaner.split_log_components(log)
        print(f"  Timestamp: {ts}")
        print(f"  Level: {level}")
        print(f"  Message: {msg}")
        print()