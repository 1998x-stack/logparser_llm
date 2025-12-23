"""
Main parsing engine that orchestrates all components.
"""
import time
from typing import List, Dict, Optional, Any
from pathlib import Path
import pandas as pd

from logparser_llm.config_manager import Config
from logparser_llm.core.prefix_tree import PrefixTree
from logparser_llm.core.merger import TemplateMerger
from logparser_llm.llm.client import LLMClient
from logparser_llm.storage.template_pool import TemplatePool
from logparser_llm.storage.cache import CacheManager
from logparser_llm.preprocessor.cleaner import LogCleaner
from logparser_llm.models.log_entry import LogEntry, ParsedLog, Template, ParsingStatistics
from logparser_llm.utils.logger import get_logger

logger = get_logger("parser")


class LogParserLLM:
    """
    Main log parser integrating LLM with prefix tree.
    
    This is the primary interface for log parsing operations.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the parser.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize components
        self.prefix_tree = PrefixTree(
            max_depth=config.prefix_tree.max_depth,
            min_cluster_size=config.prefix_tree.min_cluster_size,
            token_delimiter=config.prefix_tree.token_delimiter,
            enable_fuzzy=config.prefix_tree.enable_fuzzy_matching,
            fuzzy_threshold=config.prefix_tree.fuzzy_threshold
        )
        
        self.llm_client = LLMClient(config.llm)
        self.template_pool = TemplatePool()
        self.cleaner = LogCleaner(config.preprocessing)
        self.merger = TemplateMerger(config.merging)
        
        # Initialize cache
        self.cache = CacheManager(
            backend=config.parsing.cache_type,
            ttl=config.parsing.cache_ttl
        ) if config.parsing.use_cache else None
        
        # Statistics
        self.stats = ParsingStatistics()
        
        logger.info("LogParser-LLM initialized successfully")
        logger.info(f"  LLM: {config.llm.provider}/{config.llm.model}")
        logger.info(f"  Cache: {config.parsing.cache_type if config.parsing.use_cache else 'disabled'}")
        logger.info(f"  Prefix tree depth: {config.prefix_tree.max_depth}")
    
    def parse(self, log: str, log_id: Optional[str] = None) -> ParsedLog:
        """
        Parse a single log message.
        
        Args:
            log: Log message to parse
            log_id: Optional log identifier
            
        Returns:
            ParsedLog object
        """
        start_time = time.time()
        
        if not log_id:
            log_id = f"log_{int(time.time() * 1000000)}"
        
        # Preprocess log
        original_log = log
        cleaned_log = self.cleaner.clean(log)
        
        if not self.cleaner.is_valid_log(cleaned_log):
            logger.warning(f"Invalid log: {log[:50]}...")
            self.stats.failed += 1
            return self._create_fallback_result(original_log, start_time)
        
        log_entry = LogEntry(content=cleaned_log)
        
        # Step 1: Check cache
        if self.cache:
            cache_key = f"log:{log_entry.get_hash()}"
            cached_template_id = self.cache.get(cache_key)
            if cached_template_id:
                template = self.template_pool.get_template(cached_template_id)
                if template:
                    self.stats.cache_hits += 1
                    self.stats.total_logs += 1
                    self.stats.successfully_parsed += 1
                    return self._create_parsed_log(
                        original_log, template, time.time() - start_time, 
                        used_llm=False, cache_hit=True
                    )
        
        # Step 2: Try prefix tree clustering
        tree_result = self.prefix_tree.search(cleaned_log)
        if tree_result:
            template_id, node = tree_result
            template = self.template_pool.get_template(template_id)
            if template:
                # Cache this result
                if self.cache:
                    cache_key = f"log:{log_entry.get_hash()}"
                    self.cache.set(cache_key, template_id)
                
                self.template_pool.associate_log_with_template(cleaned_log, template_id)
                self.stats.total_logs += 1
                self.stats.successfully_parsed += 1
                return self._create_parsed_log(
                    original_log, template, time.time() - start_time,
                    used_llm=False, cache_hit=False
                )
        
        # Step 3: Use LLM for new pattern
        try:
            template_pattern = self.llm_client.extract_template(
                cleaned_log,
                use_ner=self.config.parsing.enable_ner
            )
            
            self.stats.llm_calls += 1
            
            # Create new template
            template = Template(
                template_id=f"tmpl_{len(self.template_pool.templates):04d}",
                template_pattern=template_pattern,
                example_logs=[original_log],
                confidence=0.9
            )
            
            # Add to pool and tree
            template_id = self.template_pool.add_template(template)
            self.prefix_tree.insert(cleaned_log, log_id)
            self.template_pool.associate_log_with_template(cleaned_log, template_id)
            
            # Cache this result
            if self.cache:
                cache_key = f"log:{log_entry.get_hash()}"
                self.cache.set(cache_key, template_id)
            
            self.stats.successfully_parsed += 1
            self.stats.total_logs += 1
            
            return self._create_parsed_log(
                original_log, template, time.time() - start_time,
                used_llm=True, cache_hit=False
            )
            
        except Exception as e:
            logger.error(f"Failed to parse log: {e}")
            self.stats.failed += 1
            self.stats.total_logs += 1
            return self._create_fallback_result(original_log, start_time)
    
    def parse_batch(
        self,
        logs: List[str],
        use_parallel: bool = True
    ) -> List[ParsedLog]:
        """
        Parse multiple logs efficiently.
        
        Args:
            logs: List of log messages
            use_parallel: Whether to use parallel processing
            
        Returns:
            List of ParsedLog objects
        """
        logger.info(f"Parsing batch of {len(logs)} logs")
        
        results = []
        uncached_logs = []
        uncached_indices = []
        
        # Step 1: Check cache for all logs
        for i, log in enumerate(logs):
            # Try cache first
            if self.cache:
                cleaned = self.cleaner.clean(log)
                cache_key = f"log:{LogEntry(content=cleaned).get_hash()}"
                cached_template_id = self.cache.get(cache_key)
                
                if cached_template_id:
                    template = self.template_pool.get_template(cached_template_id)
                    if template:
                        results.append(self._create_parsed_log(
                            log, template, 0, used_llm=False, cache_hit=True
                        ))
                        self.stats.cache_hits += 1
                        self.stats.total_logs += 1
                        self.stats.successfully_parsed += 1
                        continue
            
            # Not in cache
            uncached_logs.append(log)
            uncached_indices.append(i)
            results.append(None)  # Placeholder
        
        # Step 2: Process uncached logs
        if uncached_logs:
            logger.info(f"Processing {len(uncached_logs)} uncached logs")
            
            if self.config.parsing.enable_batch_processing and len(uncached_logs) > 3:
                # Batch LLM processing
                parsed_logs = self._parse_batch_with_llm(uncached_logs)
            else:
                # Individual processing
                parsed_logs = [self.parse(log) for log in uncached_logs]
            
            # Fill in results
            for idx, parsed_log in zip(uncached_indices, parsed_logs):
                results[idx] = parsed_log
        
        # Auto-merge templates periodically
        if self.config.merging.enable_auto_merge:
            if len(self.template_pool.templates) > 100:
                logger.info("Running template auto-merge...")
                self._auto_merge_templates()
        
        logger.info(
            f"Batch complete: {self.stats.cache_hits} cache hits, "
            f"{self.stats.llm_calls} LLM calls"
        )
        
        return results
    
    def parse_file(
        self,
        file_path: str,
        log_column: str = "log",
        output_path: Optional[str] = None,
        chunk_size: int = 1000
    ) -> pd.DataFrame:
        """
        Parse logs from a file.
        
        Args:
            file_path: Input file path (CSV, JSON, or text)
            log_column: Column name for logs (CSV/JSON)
            output_path: Optional output path for results
            chunk_size: Process logs in chunks
            
        Returns:
            DataFrame with parsing results
        """
        logger.info(f"Parsing file: {file_path}")
        
        # Read file
        file_path = Path(file_path)
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            logs = df[log_column].tolist()
        elif file_path.suffix == '.json':
            df = pd.read_json(file_path)
            logs = df[log_column].tolist()
        else:
            # Plain text file
            with open(file_path, 'r') as f:
                logs = [line.strip() for line in f if line.strip()]
            df = pd.DataFrame({'log': logs})
        
        # Parse logs in chunks
        all_results = []
        for i in range(0, len(logs), chunk_size):
            chunk = logs[i:i+chunk_size]
            logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(logs)-1)//chunk_size + 1}")
            parsed_chunk = self.parse_batch(chunk)
            all_results.extend(parsed_chunk)
        
        # Create output DataFrame
        results = []
        for parsed in all_results:
            results.append({
                'original': parsed.original,
                'template_id': parsed.template_id,
                'template': parsed.template_pattern,
                'confidence': parsed.confidence,
                'used_llm': parsed.used_llm,
                'cache_hit': parsed.cache_hit,
                'processing_time_ms': parsed.processing_time_ms
            })
        
        result_df = pd.DataFrame(results)
        
        # Save if output path provided
        if output_path:
            result_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to: {output_path}")
        
        return result_df
    
    def _parse_batch_with_llm(self, logs: List[str]) -> List[ParsedLog]:
        """Parse multiple logs using batch LLM call."""
        try:
            # Clean logs first
            cleaned_logs = [self.cleaner.clean(log) for log in logs]
            
            templates = self.llm_client.extract_template_batch(cleaned_logs)
            self.stats.llm_calls += 1
            
            results = []
            for log, template_pattern in zip(logs, templates):
                # Create template
                template = Template(
                    template_id=f"tmpl_{len(self.template_pool.templates):04d}",
                    template_pattern=template_pattern,
                    example_logs=[log],
                    confidence=0.85
                )
                
                template_id = self.template_pool.add_template(template)
                cleaned = self.cleaner.clean(log)
                self.template_pool.associate_log_with_template(cleaned, template_id)
                
                # Cache
                if self.cache:
                    cache_key = f"log:{LogEntry(content=cleaned).get_hash()}"
                    self.cache.set(cache_key, template_id)
                
                parsed = self._create_parsed_log(
                    log, template, 0, used_llm=True, cache_hit=False
                )
                results.append(parsed)
                
                self.stats.successfully_parsed += 1
                self.stats.total_logs += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Batch LLM processing failed: {e}")
            # Fallback to individual processing
            return [self.parse(log) for log in logs]
    
    def _auto_merge_templates(self):
        """Automatically merge similar templates."""
        templates = self.template_pool.get_all_templates()
        merged = self.merger.merge_batch(templates)
        
        if len(merged) < len(templates):
            logger.info(f"Merged {len(templates)} templates into {len(merged)}")
            # Update pool (simplified - in production would need careful handling)
            self.template_pool.clear()
            for t in merged:
                self.template_pool.add_template(t)
    
    def _create_parsed_log(
        self,
        log: str,
        template: Template,
        processing_time: float,
        used_llm: bool,
        cache_hit: bool
    ) -> ParsedLog:
        """Create ParsedLog object."""
        # Extract variables (simplified)
        variables = self._extract_variables(log, template.template_pattern)
        
        # Extract timestamp and level
        timestamp_str, level, _ = self.cleaner.split_log_components(log)
        
        return ParsedLog(
            original=log,
            template_id=template.template_id,
            template_pattern=template.template_pattern,
            variables=variables,
            log_level=level,
            confidence=template.confidence,
            processing_time_ms=processing_time * 1000,
            used_llm=used_llm,
            cache_hit=cache_hit
        )
    
    def _create_fallback_result(self, log: str, start_time: float) -> ParsedLog:
        """Create fallback result for failed parsing."""
        return ParsedLog(
            original=log,
            template_id="unknown",
            template_pattern=log,
            confidence=0.0,
            processing_time_ms=(time.time() - start_time) * 1000,
            used_llm=False,
            cache_hit=False
        )
    
    @staticmethod
    def _extract_variables(log: str, template: str) -> Dict[str, Any]:
        """Extract variable values from log using template."""
        import re
        
        # Convert template to regex
        pattern = re.escape(template)
        pattern = pattern.replace(r'\<\*\>', r'(.+?)')
        
        try:
            match = re.match(pattern, log)
            if match:
                return {f"var_{i}": val for i, val in enumerate(match.groups(), 1)}
        except Exception:
            pass
        
        return {}
    
    def get_statistics(self) -> Dict:
        """Get comprehensive parsing statistics."""
        stats = self.stats.to_dict()
        stats.update({
            'prefix_tree': self.prefix_tree.get_statistics(),
            'template_pool': self.template_pool.get_statistics(),
            'llm': self.llm_client.get_statistics()
        })
        
        if self.cache:
            stats['cache'] = self.cache.get_statistics()
        
        return stats
    
    def save_state(self, directory: str):
        """Save parser state."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save templates
        self.template_pool.save_to_file(str(dir_path / "templates.json"))
        
        # Save statistics
        import json
        with open(dir_path / "statistics.json", 'w') as f:
            json.dump(self.get_statistics(), f, indent=2, default=str)
        
        logger.info(f"Parser state saved to: {directory}")
    
    def load_state(self, directory: str):
        """Load parser state."""
        dir_path = Path(directory)
        
        template_file = dir_path / "templates.json"
        if template_file.exists():
            self.template_pool.load_from_file(str(template_file))
            logger.info(f"Loaded {len(self.template_pool.templates)} templates")
        else:
            logger.warning(f"No template file found at {template_file}")