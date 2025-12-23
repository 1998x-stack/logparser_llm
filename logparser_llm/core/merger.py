"""
Template merging module for combining similar templates.
"""
from typing import List, Tuple, Optional
from logparser_llm.models.log_entry import Template
from logparser_llm.config_manager import MergingConfig


class TemplateMerger:
    """Merge similar log templates to reduce redundancy."""
    
    def __init__(self, config: Optional[MergingConfig] = None):
        """
        Initialize template merger.
        
        Args:
            config: Merging configuration
        """
        self.config = config or MergingConfig()
    
    def should_merge(self, template1: Template, template2: Template) -> bool:
        """
        Determine if two templates should be merged.
        
        Args:
            template1: First template
            template2: Second template
            
        Returns:
            True if templates should be merged
        """
        if not self.config.enable_auto_merge:
            return False
        
        # Check similarity
        similarity = self.calculate_similarity(
            template1.template_pattern,
            template2.template_pattern
        )
        
        if similarity < self.config.merge_threshold:
            return False
        
        # Check edit distance
        if self.config.max_edit_distance > 0:
            distance = self.levenshtein_distance(
                template1.template_pattern,
                template2.template_pattern
            )
            if distance > self.config.max_edit_distance:
                return False
        
        return True
    
    def merge(self, template1: Template, template2: Template) -> Template:
        """
        Merge two templates into one.
        
        Args:
            template1: First template
            template2: Second template
            
        Returns:
            Merged template
        """
        # Create merged pattern
        merged_pattern = self._merge_patterns(
            template1.template_pattern,
            template2.template_pattern
        )
        
        # Combine metadata
        merged = Template(
            template_id=template1.template_id,  # Keep first ID
            template_pattern=merged_pattern,
            static_tokens=self._merge_static_tokens(
                template1.static_tokens,
                template2.static_tokens
            ),
            example_logs=template1.example_logs + template2.example_logs,
            count=template1.count + template2.count,
            confidence=min(template1.confidence, template2.confidence) * 0.95,
            created_at=template1.created_at
        )
        
        # Limit example logs
        if len(merged.example_logs) > 10:
            merged.example_logs = merged.example_logs[:10]
        
        return merged
    
    def merge_batch(self, templates: List[Template]) -> List[Template]:
        """
        Merge a batch of templates.
        
        Args:
            templates: List of templates
            
        Returns:
            List of merged templates
        """
        if not self.config.enable_auto_merge or len(templates) < 2:
            return templates
        
        # Build similarity matrix
        merged = []
        used = set()
        
        for i, t1 in enumerate(templates):
            if i in used:
                continue
            
            # Find all similar templates
            to_merge = [t1]
            for j, t2 in enumerate(templates[i+1:], start=i+1):
                if j in used:
                    continue
                
                if self.should_merge(t1, t2):
                    to_merge.append(t2)
                    used.add(j)
            
            # Merge all similar templates
            if len(to_merge) > 1:
                result = to_merge[0]
                for t in to_merge[1:]:
                    result = self.merge(result, t)
                merged.append(result)
            else:
                merged.append(t1)
            
            used.add(i)
        
        return merged
    
    def calculate_similarity(self, pattern1: str, pattern2: str) -> float:
        """
        Calculate similarity between two patterns.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Similarity score between 0 and 1
        """
        tokens1 = pattern1.split()
        tokens2 = pattern2.split()
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Exact match
        if pattern1 == pattern2:
            return 1.0
        
        # Token-based similarity
        matching_tokens = 0
        max_len = max(len(tokens1), len(tokens2))
        
        for t1, t2 in zip(tokens1, tokens2):
            if t1 == t2 or (t1 == "<*>" and t2 == "<*>"):
                matching_tokens += 1
        
        # Calculate Jaccard similarity
        set1 = set(tokens1)
        set2 = set(tokens2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        jaccard = intersection / union if union > 0 else 0.0
        
        # Combine metrics
        positional_score = matching_tokens / max_len
        similarity = (positional_score * 0.7) + (jaccard * 0.3)
        
        return similarity
    
    def _merge_patterns(self, pattern1: str, pattern2: str) -> str:
        """
        Merge two patterns into one.
        
        Strategy: Keep matching tokens, replace non-matching with <*>
        """
        tokens1 = pattern1.split()
        tokens2 = pattern2.split()
        
        merged = []
        max_len = max(len(tokens1), len(tokens2))
        
        for i in range(max_len):
            t1 = tokens1[i] if i < len(tokens1) else None
            t2 = tokens2[i] if i < len(tokens2) else None
            
            if t1 == t2:
                merged.append(t1)
            elif t1 == "<*>" or t2 == "<*>":
                merged.append("<*>")
            elif t1 is None:
                merged.append("<*>")
            elif t2 is None:
                merged.append("<*>")
            else:
                # Different tokens, use wildcard
                merged.append("<*>")
        
        return " ".join(merged)
    
    def _merge_static_tokens(
        self,
        tokens1: List[str],
        tokens2: List[str]
    ) -> List[str]:
        """Merge static token lists."""
        # Keep only common tokens
        set1 = set(tokens1)
        set2 = set(tokens2)
        return list(set1 & set2)
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Edit distance
        """
        if len(s1) < len(s2):
            return TemplateMerger.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def find_mergeable_pairs(
        self,
        templates: List[Template]
    ) -> List[Tuple[int, int, float]]:
        """
        Find all pairs of templates that can be merged.
        
        Args:
            templates: List of templates
            
        Returns:
            List of (index1, index2, similarity) tuples
        """
        pairs = []
        
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if self.should_merge(templates[i], templates[j]):
                    similarity = self.calculate_similarity(
                        templates[i].template_pattern,
                        templates[j].template_pattern
                    )
                    pairs.append((i, j, similarity))
        
        # Sort by similarity (descending)
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs


# Example usage
if __name__ == "__main__":
    from datetime import datetime
    
    # Create sample templates
    t1 = Template(
        template_id="tmpl_001",
        template_pattern="User <*> logged in",
        static_tokens=["User", "logged", "in"],
        example_logs=["User john logged in"],
        count=5,
        confidence=0.95
    )
    
    t2 = Template(
        template_id="tmpl_002",
        template_pattern="User <*> logged out",
        static_tokens=["User", "logged", "out"],
        example_logs=["User mary logged out"],
        count=3,
        confidence=0.90
    )
    
    t3 = Template(
        template_id="tmpl_003",
        template_pattern="User <*> logged in",  # Duplicate of t1
        static_tokens=["User", "logged", "in"],
        example_logs=["User bob logged in"],
        count=2,
        confidence=0.92
    )
    
    # Test merger
    merger = TemplateMerger()
    
    print("Template 1:", t1.template_pattern)
    print("Template 2:", t2.template_pattern)
    print("Template 3:", t3.template_pattern)
    print()
    
    # Check if should merge
    print(f"Should merge t1 and t2? {merger.should_merge(t1, t2)}")
    print(f"Should merge t1 and t3? {merger.should_merge(t1, t3)}")
    print()
    
    # Calculate similarity
    sim = merger.calculate_similarity(t1.template_pattern, t2.template_pattern)
    print(f"Similarity between t1 and t2: {sim:.2f}")
    print()
    
    # Merge batch
    templates = [t1, t2, t3]
    merged = merger.merge_batch(templates)
    
    print(f"Original templates: {len(templates)}")
    print(f"After merging: {len(merged)}")
    print("\nMerged templates:")
    for t in merged:
        print(f"  {t.template_id}: {t.template_pattern} (count: {t.count})")