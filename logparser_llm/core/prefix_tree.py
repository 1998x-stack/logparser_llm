"""
Prefix Tree implementation for efficient log clustering.
"""
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import re
from loguru import logger


@dataclass
class PrefixTreeNode:
    """Node in the prefix tree."""
    
    token: Optional[str] = None
    children: Dict[str, 'PrefixTreeNode'] = field(default_factory=dict)
    log_ids: Set[str] = field(default_factory=set)
    is_leaf: bool = False
    template_id: Optional[str] = None
    depth: int = 0
    
    def add_child(self, token: str) -> 'PrefixTreeNode':
        """Add a child node."""
        if token not in self.children:
            self.children[token] = PrefixTreeNode(
                token=token,
                depth=self.depth + 1
            )
        return self.children[token]
    
    def get_child(self, token: str) -> Optional['PrefixTreeNode']:
        """Get child node by token."""
        return self.children.get(token)
    
    def is_template_node(self) -> bool:
        """Check if this node represents a complete template."""
        return self.is_leaf and self.template_id is not None


class PrefixTree:
    """
    Prefix Tree for efficient log clustering based on syntax.
    Groups logs with common prefix patterns.
    """
    
    def __init__(
        self,
        max_depth: int = 5,
        min_cluster_size: int = 3,
        token_delimiter: str = " ",
        enable_fuzzy: bool = True,
        fuzzy_threshold: float = 0.8
    ):
        """
        Initialize prefix tree.
        
        Args:
            max_depth: Maximum depth of tree
            min_cluster_size: Minimum logs to form a cluster
            token_delimiter: Delimiter for tokenization
            enable_fuzzy: Enable fuzzy matching
            fuzzy_threshold: Threshold for fuzzy matching
        """
        self.root = PrefixTreeNode(token="<ROOT>", depth=0)
        self.max_depth = max_depth
        self.min_cluster_size = min_cluster_size
        self.token_delimiter = token_delimiter
        self.enable_fuzzy = enable_fuzzy
        self.fuzzy_threshold = fuzzy_threshold
        
        # Statistics
        self.total_logs = 0
        self.num_clusters = 0
        self.template_map: Dict[str, str] = {}  # template_id -> template_pattern
        
        logger.info(f"PrefixTree initialized with max_depth={max_depth}")
    
    def tokenize(self, log: str) -> List[str]:
        """
        Tokenize log message.
        
        Args:
            log: Log message
            
        Returns:
            List of tokens
        """
        # Simple whitespace tokenization
        tokens = log.strip().split(self.token_delimiter)
        
        # Remove empty tokens
        tokens = [t for t in tokens if t]
        
        return tokens
    
    def insert(self, log: str, log_id: str) -> Optional[str]:
        """
        Insert a log into the prefix tree.
        
        Args:
            log: Log message
            log_id: Unique log identifier
            
        Returns:
            Template ID if log matches existing cluster, None otherwise
        """
        tokens = self.tokenize(log)
        
        if not tokens:
            logger.warning(f"Empty tokens for log: {log}")
            return None
        
        # Traverse tree up to max_depth
        current = self.root
        depth = 0
        
        for token in tokens[:self.max_depth]:
            # Try exact match first
            child = current.get_child(token)
            
            # Try fuzzy match if enabled
            if not child and self.enable_fuzzy:
                child = self._fuzzy_match_child(current, token)
            
            if child:
                current = child
            else:
                # Create new child
                current = current.add_child(token)
            
            depth += 1
            
            # Stop at max depth
            if depth >= self.max_depth:
                break
        
        # Add log ID to the node
        current.log_ids.add(log_id)
        self.total_logs += 1
        
        # Check if this node can be a template node
        if len(current.log_ids) >= self.min_cluster_size:
            if not current.is_template_node():
                # Create new template
                current.is_leaf = True
                current.template_id = self._generate_template_id()
                self.num_clusters += 1
                logger.debug(f"Created new cluster: {current.template_id}")
            
            return current.template_id
        
        return None
    
    def search(self, log: str) -> Optional[Tuple[str, PrefixTreeNode]]:
        """
        Search for a log in the tree and find its cluster.
        
        Args:
            log: Log message
            
        Returns:
            Tuple of (template_id, node) if found, None otherwise
        """
        tokens = self.tokenize(log)
        current = self.root
        
        for token in tokens[:self.max_depth]:
            child = current.get_child(token)
            
            if not child and self.enable_fuzzy:
                child = self._fuzzy_match_child(current, token)
            
            if not child:
                return None
            
            current = child
            
            # Check if we found a template node
            if current.is_template_node():
                return current.template_id, current
        
        # Check last node
        if current.is_template_node():
            return current.template_id, current
        
        return None
    
    def get_cluster_logs(self, template_id: str) -> List[str]:
        """
        Get all log IDs in a cluster.
        
        Args:
            template_id: Template identifier
            
        Returns:
            List of log IDs
        """
        node = self._find_template_node(template_id)
        if node:
            return list(node.log_ids)
        return []
    
    def get_statistics(self) -> Dict:
        """Get tree statistics."""
        return {
            "total_logs": self.total_logs,
            "num_clusters": self.num_clusters,
            "avg_cluster_size": (
                self.total_logs / self.num_clusters if self.num_clusters > 0 else 0
            ),
            "max_depth": self.max_depth,
            "total_nodes": self._count_nodes(self.root)
        }
    
    def _fuzzy_match_child(
        self,
        node: PrefixTreeNode,
        token: str
    ) -> Optional[PrefixTreeNode]:
        """
        Find child node using fuzzy matching.
        
        Args:
            node: Parent node
            token: Token to match
            
        Returns:
            Matching child node if found
        """
        best_match = None
        best_score = 0.0
        
        for child_token, child_node in node.children.items():
            score = self._token_similarity(token, child_token)
            if score > best_score and score >= self.fuzzy_threshold:
                best_score = score
                best_match = child_node
        
        return best_match
    
    @staticmethod
    def _token_similarity(token1: str, token2: str) -> float:
        """
        Calculate similarity between two tokens.
        
        Args:
            token1: First token
            token2: Second token
            
        Returns:
            Similarity score between 0 and 1
        """
        # Exact match
        if token1 == token2:
            return 1.0
        
        # Check if both are numbers
        if token1.replace('.', '').replace('-', '').isdigit() and \
           token2.replace('.', '').replace('-', '').isdigit():
            return 0.9  # High similarity for numbers
        
        # Check if both match common patterns
        patterns = [
            r'^\d+\.\d+\.\d+\.\d+$',  # IP
            r'^[a-f0-9]{32}$',  # MD5
            r'^[a-f0-9]{64}$',  # SHA256
            r'^\d{4}-\d{2}-\d{2}$',  # Date
        ]
        
        for pattern in patterns:
            if re.match(pattern, token1) and re.match(pattern, token2):
                return 0.85
        
        # Levenshtein distance-based similarity
        distance = PrefixTree._levenshtein_distance(token1, token2)
        max_len = max(len(token1), len(token2))
        if max_len == 0:
            return 1.0
        return 1.0 - (distance / max_len)
    
    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return PrefixTree._levenshtein_distance(s2, s1)
        
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
    
    def _find_template_node(self, template_id: str) -> Optional[PrefixTreeNode]:
        """Find node by template ID."""
        def dfs(node: PrefixTreeNode) -> Optional[PrefixTreeNode]:
            if node.template_id == template_id:
                return node
            for child in node.children.values():
                result = dfs(child)
                if result:
                    return result
            return None
        
        return dfs(self.root)
    
    def _generate_template_id(self) -> str:
        """Generate unique template ID."""
        return f"tmpl_{self.num_clusters:04d}"
    
    def _count_nodes(self, node: PrefixTreeNode) -> int:
        """Count total nodes in tree."""
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count
    
    def visualize(self, max_depth: Optional[int] = None) -> str:
        """
        Create a text visualization of the tree.
        
        Args:
            max_depth: Maximum depth to visualize
            
        Returns:
            String representation of tree
        """
        lines = []
        max_d = max_depth or self.max_depth
        
        def dfs(node: PrefixTreeNode, prefix: str, is_last: bool):
            if node.depth > max_d:
                return
            
            # Format node info
            node_info = f"{node.token}"
            if node.is_template_node():
                node_info += f" [{node.template_id}, {len(node.log_ids)} logs]"
            
            # Add to output
            connector = "└── " if is_last else "├── "
            if node.depth > 0:
                lines.append(prefix + connector + node_info)
            else:
                lines.append(node_info)
            
            # Recurse to children
            children = list(node.children.values())
            for i, child in enumerate(children):
                is_last_child = (i == len(children) - 1)
                extension = "    " if is_last else "│   "
                dfs(child, prefix + extension, is_last_child)
        
        dfs(self.root, "", True)
        return "\n".join(lines)


# Example usage and testing
if __name__ == "__main__":
    tree = PrefixTree(max_depth=3, min_cluster_size=2)
    
    # Sample logs
    logs = [
        ("log1", "User john logged in at 10:00"),
        ("log2", "User mary logged in at 10:05"),
        ("log3", "User bob logged in at 10:10"),
        ("log4", "Failed to connect to server1"),
        ("log5", "Failed to connect to server2"),
        ("log6", "Started process with PID 1234"),
    ]
    
    print("Inserting logs...")
    for log_id, log_msg in logs:
        template_id = tree.insert(log_msg, log_id)
        print(f"{log_id}: {log_msg} -> {template_id}")
    
    print("\nTree Statistics:")
    print(tree.get_statistics())
    
    print("\nTree Visualization:")
    print(tree.visualize())