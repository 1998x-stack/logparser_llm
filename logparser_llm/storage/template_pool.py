"""
Template pool for storing and managing log templates.
"""
import json
from typing import Dict, List, Optional, Set
from pathlib import Path
from datetime import datetime
from loguru import logger

from ..models.log_entry import Template


class TemplatePool:
    """
    Storage and management for log templates.
    Provides CRUD operations and template matching.
    """
    
    def __init__(self):
        """Initialize template pool."""
        self.templates: Dict[str, Template] = {}
        self.template_index: Dict[str, Set[str]] = {}  # pattern hash -> template_ids
        self.log_to_template: Dict[str, str] = {}  # log hash -> template_id
        
        logger.info("Template pool initialized")
    
    def add_template(self, template: Template) -> str:
        """
        Add a new template to the pool.
        
        Args:
            template: Template object
            
        Returns:
            Template ID
        """
        # Check if similar template exists
        similar_id = self._find_similar_template(template.template_pattern)
        if similar_id:
            # Update existing template
            self.templates[similar_id].count += 1
            self.templates[similar_id].updated_at = datetime.now()
            if template.example_logs:
                self.templates[similar_id].example_logs.extend(template.example_logs)
            logger.debug(f"Updated existing template: {similar_id}")
            return similar_id
        
        # Add new template
        template_id = template.template_id
        self.templates[template_id] = template
        
        # Update index
        pattern_hash = self._hash_pattern(template.template_pattern)
        if pattern_hash not in self.template_index:
            self.template_index[pattern_hash] = set()
        self.template_index[pattern_hash].add(template_id)
        
        logger.info(f"Added new template: {template_id}")
        return template_id
    
    def get_template(self, template_id: str) -> Optional[Template]:
        """
        Get template by ID.
        
        Args:
            template_id: Template identifier
            
        Returns:
            Template object if found
        """
        return self.templates.get(template_id)
    
    def find_template_for_log(self, log: str) -> Optional[str]:
        """
        Find matching template for a log message.
        
        Args:
            log: Log message
            
        Returns:
            Template ID if found
        """
        log_hash = self._hash_log(log)
        return self.log_to_template.get(log_hash)
    
    def associate_log_with_template(self, log: str, template_id: str):
        """
        Associate a log with a template.
        
        Args:
            log: Log message
            template_id: Template ID
        """
        log_hash = self._hash_log(log)
        self.log_to_template[log_hash] = template_id
        
        # Update template
        if template_id in self.templates:
            template = self.templates[template_id]
            template.count += 1
            if log not in template.example_logs:
                template.example_logs.append(log)
                # Keep only recent examples
                if len(template.example_logs) > 10:
                    template.example_logs = template.example_logs[-10:]
    
    def get_all_templates(self) -> List[Template]:
        """Get all templates."""
        return list(self.templates.values())
    
    def get_top_templates(self, n: int = 10) -> List[Template]:
        """
        Get top N templates by frequency.
        
        Args:
            n: Number of templates to return
            
        Returns:
            List of top templates
        """
        sorted_templates = sorted(
            self.templates.values(),
            key=lambda t: t.count,
            reverse=True
        )
        return sorted_templates[:n]
    
    def merge_templates(self, template_id1: str, template_id2: str) -> Optional[str]:
        """
        Merge two similar templates.
        
        Args:
            template_id1: First template ID
            template_id2: Second template ID
            
        Returns:
            ID of merged template
        """
        t1 = self.templates.get(template_id1)
        t2 = self.templates.get(template_id2)
        
        if not t1 or not t2:
            return None
        
        # Create merged template
        merged_pattern = self._merge_patterns(t1.template_pattern, t2.template_pattern)
        
        merged = Template(
            template_id=template_id1,  # Keep first ID
            template_pattern=merged_pattern,
            count=t1.count + t2.count,
            example_logs=t1.example_logs + t2.example_logs,
            confidence=min(t1.confidence, t2.confidence)
        )
        
        # Update pool
        self.templates[template_id1] = merged
        del self.templates[template_id2]
        
        # Update log associations
        for log_hash, tid in list(self.log_to_template.items()):
            if tid == template_id2:
                self.log_to_template[log_hash] = template_id1
        
        logger.info(f"Merged templates: {template_id1} + {template_id2}")
        return template_id1
    
    def remove_template(self, template_id: str) -> bool:
        """
        Remove a template.
        
        Args:
            template_id: Template ID
            
        Returns:
            True if removed
        """
        if template_id not in self.templates:
            return False
        
        del self.templates[template_id]
        
        # Clean up index
        for pattern_hash, ids in list(self.template_index.items()):
            if template_id in ids:
                ids.remove(template_id)
                if not ids:
                    del self.template_index[pattern_hash]
        
        # Clean up log associations
        for log_hash, tid in list(self.log_to_template.items()):
            if tid == template_id:
                del self.log_to_template[log_hash]
        
        logger.info(f"Removed template: {template_id}")
        return True
    
    def save_to_file(self, file_path: str):
        """
        Save templates to JSON file.
        
        Args:
            file_path: Output file path
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "templates": [t.dict() for t in self.templates.values()],
            "metadata": {
                "total_templates": len(self.templates),
                "saved_at": datetime.now().isoformat()
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved {len(self.templates)} templates to {file_path}")
    
    def load_from_file(self, file_path: str):
        """
        Load templates from JSON file.
        
        Args:
            file_path: Input file path
        """
        if not Path(file_path).exists():
            logger.warning(f"Template file not found: {file_path}")
            return
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Load templates
        for template_dict in data.get("templates", []):
            template = Template(**template_dict)
            self.templates[template.template_id] = template
            
            # Update index
            pattern_hash = self._hash_pattern(template.template_pattern)
            if pattern_hash not in self.template_index:
                self.template_index[pattern_hash] = set()
            self.template_index[pattern_hash].add(template.template_id)
        
        logger.info(f"Loaded {len(self.templates)} templates from {file_path}")
    
    def get_statistics(self) -> Dict:
        """Get pool statistics."""
        total_logs = sum(t.count for t in self.templates.values())
        avg_confidence = (
            sum(t.confidence for t in self.templates.values()) / len(self.templates)
            if self.templates else 0.0
        )
        
        return {
            "total_templates": len(self.templates),
            "total_logs_parsed": total_logs,
            "avg_logs_per_template": total_logs / len(self.templates) if self.templates else 0,
            "avg_confidence": avg_confidence,
            "cache_size": len(self.log_to_template)
        }
    
    def clear(self):
        """Clear all templates."""
        self.templates.clear()
        self.template_index.clear()
        self.log_to_template.clear()
        logger.info("Template pool cleared")
    
    def _find_similar_template(self, pattern: str) -> Optional[str]:
        """Find similar template by pattern."""
        pattern_hash = self._hash_pattern(pattern)
        template_ids = self.template_index.get(pattern_hash, set())
        
        for tid in template_ids:
            if self.templates[tid].template_pattern == pattern:
                return tid
        
        return None
    
    def _merge_patterns(self, pattern1: str, pattern2: str) -> str:
        """Merge two similar patterns."""
        tokens1 = pattern1.split()
        tokens2 = pattern2.split()
        
        merged = []
        for t1, t2 in zip(tokens1, tokens2):
            if t1 == t2:
                merged.append(t1)
            else:
                merged.append("<*>")
        
        return " ".join(merged)
    
    @staticmethod
    def _hash_pattern(pattern: str) -> str:
        """Create hash for pattern indexing."""
        import hashlib
        # Use first few tokens for quick lookup
        tokens = pattern.split()[:3]
        key = "_".join(tokens)
        return hashlib.md5(key.encode()).hexdigest()[:8]
    
    @staticmethod
    def _hash_log(log: str) -> str:
        """Create hash for log message."""
        import hashlib
        return hashlib.md5(log.encode()).hexdigest()


# Example usage
if __name__ == "__main__":
    pool = TemplatePool()
    
    # Add templates
    t1 = Template(
        template_id="tmpl_001",
        template_pattern="User <*> logged in",
        example_logs=["User john logged in"]
    )
    pool.add_template(t1)
    
    t2 = Template(
        template_id="tmpl_002",
        template_pattern="Failed to connect to <*>",
        example_logs=["Failed to connect to database"]
    )
    pool.add_template(t2)
    
    print("Statistics:", pool.get_statistics())
    
    # Save and load
    pool.save_to_file("templates.json")
    
    pool2 = TemplatePool()
    pool2.load_from_file("templates.json")
    print("Loaded templates:", len(pool2.get_all_templates()))