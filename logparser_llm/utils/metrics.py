"""
Evaluation metrics for log parsing.
"""
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import numpy as np


class ParsingMetrics:
    """Calculate log parsing accuracy metrics."""
    
    @staticmethod
    def grouping_accuracy(
        predictions: List[str],
        ground_truth: List[str]
    ) -> float:
        """
        Calculate Grouping Accuracy (GA).
        
        GA measures how well logs are clustered into groups.
        
        Args:
            predictions: Predicted template IDs
            ground_truth: Ground truth template IDs
            
        Returns:
            Grouping accuracy (0-1)
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        if not predictions:
            return 0.0
        
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        return correct / len(predictions)
    
    @staticmethod
    def parsing_accuracy(
        predicted_templates: List[str],
        ground_truth_templates: List[str]
    ) -> float:
        """
        Calculate Parsing Accuracy (PA).
        
        PA measures template extraction accuracy.
        
        Args:
            predicted_templates: Predicted template patterns
            ground_truth_templates: Ground truth patterns
            
        Returns:
            Parsing accuracy (0-1)
        """
        if len(predicted_templates) != len(ground_truth_templates):
            raise ValueError("Must have same number of templates")
        
        if not predicted_templates:
            return 0.0
        
        correct = sum(
            1 for p, g in zip(predicted_templates, ground_truth_templates)
            if ParsingMetrics._normalize_template(p) == ParsingMetrics._normalize_template(g)
        )
        
        return correct / len(predicted_templates)
    
    @staticmethod
    def f1_score(
        predictions: List[str],
        ground_truth: List[str]
    ) -> Tuple[float, float, float]:
        """
        Calculate F1 score, precision, and recall.
        
        Args:
            predictions: Predicted template IDs
            ground_truth: Ground truth template IDs
            
        Returns:
            Tuple of (f1, precision, recall)
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        # Build clusters
        pred_clusters = defaultdict(set)
        true_clusters = defaultdict(set)
        
        for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
            pred_clusters[pred].add(i)
            true_clusters[true].add(i)
        
        # Calculate TP, FP, FN
        true_positive = 0
        false_positive = 0
        false_negative = 0
        
        # For each predicted cluster
        for pred_cluster in pred_clusters.values():
            # Find best matching true cluster
            best_overlap = 0
            for true_cluster in true_clusters.values():
                overlap = len(pred_cluster & true_cluster)
                best_overlap = max(best_overlap, overlap)
            
            true_positive += best_overlap
            false_positive += len(pred_cluster) - best_overlap
        
        # Calculate false negatives
        for true_cluster in true_clusters.values():
            best_overlap = 0
            for pred_cluster in pred_clusters.values():
                overlap = len(pred_cluster & true_cluster)
                best_overlap = max(best_overlap, overlap)
            false_negative += len(true_cluster) - best_overlap
        
        # Calculate metrics
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1, precision, recall
    
    @staticmethod
    def template_accuracy(
        predicted_templates: Dict[str, str],
        ground_truth_templates: Dict[str, str]
    ) -> float:
        """
        Calculate template-level accuracy.
        
        Args:
            predicted_templates: Dict of template_id -> pattern
            ground_truth_templates: Dict of template_id -> pattern
            
        Returns:
            Template accuracy (0-1)
        """
        if not ground_truth_templates:
            return 0.0
        
        correct = 0
        for tid, gt_pattern in ground_truth_templates.items():
            if tid in predicted_templates:
                pred_pattern = predicted_templates[tid]
                if ParsingMetrics._normalize_template(pred_pattern) == \
                   ParsingMetrics._normalize_template(gt_pattern):
                    correct += 1
        
        return correct / len(ground_truth_templates)
    
    @staticmethod
    def adjusted_rand_index(
        predictions: List[str],
        ground_truth: List[str]
    ) -> float:
        """
        Calculate Adjusted Rand Index (ARI) for clustering.
        
        Args:
            predictions: Predicted cluster labels
            ground_truth: True cluster labels
            
        Returns:
            ARI score (-1 to 1, 1 is perfect)
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Must have same length")
        
        # Build contingency table
        pred_clusters = defaultdict(set)
        true_clusters = defaultdict(set)
        
        for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
            pred_clusters[pred].add(i)
            true_clusters[true].add(i)
        
        # Calculate combinations
        def n_choose_2(n):
            return n * (n - 1) / 2 if n > 1 else 0
        
        # Sum of combinations for each cluster
        sum_comb_pred = sum(n_choose_2(len(cluster)) for cluster in pred_clusters.values())
        sum_comb_true = sum(n_choose_2(len(cluster)) for cluster in true_clusters.values())
        
        # Intersection combinations
        sum_comb_intersect = 0
        for pred_cluster in pred_clusters.values():
            for true_cluster in true_clusters.values():
                overlap = len(pred_cluster & true_cluster)
                sum_comb_intersect += n_choose_2(overlap)
        
        # Total combinations
        n = len(predictions)
        sum_comb_total = n_choose_2(n)
        
        # Calculate ARI
        if sum_comb_total == 0:
            return 0.0
        
        expected_index = sum_comb_pred * sum_comb_true / sum_comb_total
        max_index = (sum_comb_pred + sum_comb_true) / 2
        
        if max_index == expected_index:
            return 0.0
        
        ari = (sum_comb_intersect - expected_index) / (max_index - expected_index)
        return ari
    
    @staticmethod
    def _normalize_template(template: str) -> str:
        """Normalize template for comparison."""
        # Remove extra whitespace
        normalized = ' '.join(template.split())
        return normalized.strip()
    
    @staticmethod
    def calculate_all_metrics(
        predictions: List[str],
        ground_truth: List[str],
        predicted_templates: Dict[str, str],
        ground_truth_templates: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate all metrics at once.
        
        Args:
            predictions: Predicted template IDs for each log
            ground_truth: Ground truth template IDs
            predicted_templates: Predicted template patterns
            ground_truth_templates: Ground truth patterns
            
        Returns:
            Dictionary of all metrics
        """
        ga = ParsingMetrics.grouping_accuracy(predictions, ground_truth)
        f1, precision, recall = ParsingMetrics.f1_score(predictions, ground_truth)
        ari = ParsingMetrics.adjusted_rand_index(predictions, ground_truth)
        ta = ParsingMetrics.template_accuracy(predicted_templates, ground_truth_templates)
        
        return {
            'grouping_accuracy': ga,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'adjusted_rand_index': ari,
            'template_accuracy': ta
        }


# Example usage
if __name__ == "__main__":
    # Sample data
    predictions = ['tmpl_1', 'tmpl_1', 'tmpl_2', 'tmpl_2', 'tmpl_3']
    ground_truth = ['gt_1', 'gt_1', 'gt_2', 'gt_2', 'gt_3']
    
    # Calculate metrics
    ga = ParsingMetrics.grouping_accuracy(predictions, ground_truth)
    print(f"Grouping Accuracy: {ga:.2%}")
    
    f1, precision, recall = ParsingMetrics.f1_score(predictions, ground_truth)
    print(f"F1 Score: {f1:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    
    ari = ParsingMetrics.adjusted_rand_index(predictions, ground_truth)
    print(f"Adjusted Rand Index: {ari:.4f}")
    
    # Template accuracy
    pred_templates = {
        'tmpl_1': 'User <*> logged in',
        'tmpl_2': 'Failed to connect to <*>',
        'tmpl_3': 'Process <*> started'
    }
    
    gt_templates = {
        'tmpl_1': 'User <*> logged in',
        'tmpl_2': 'Failed to connect to <*>',
        'tmpl_3': 'Process <*> started'
    }
    
    ta = ParsingMetrics.template_accuracy(pred_templates, gt_templates)
    print(f"Template Accuracy: {ta:.2%}")