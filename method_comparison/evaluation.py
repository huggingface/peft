from typing import Dict, Callable, Any, List
import numpy as np
from dataclasses import dataclass
import json
import os

@dataclass
class MetricResult:
    """Container for metric evaluation results."""
    metric_name: str
    value: float
    method_name: str
    dataset: str
    timestamp: str

class MetricRegistry:
    """Registry for managing evaluation metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, Callable] = {}
        self.results_file = "metric_results.json"
    
    def add_metric(self, metric_name: str, metric_fn: Callable) -> None:
        """Add a new evaluation metric to the registry.
        
        Args:
            metric_name: Name of the metric
            metric_fn: Function that computes the metric. Should take predictions and ground truth as input.
        """
        if metric_name in self.metrics:
            raise ValueError(f"Metric {metric_name} already exists in registry")
        
        self.metrics[metric_name] = metric_fn
    
    def evaluate(self, metric_name: str, predictions: Any, ground_truth: Any, method_name: str, dataset: str) -> MetricResult:
        """Evaluate a specific metric on given predictions and ground truth.
        
        Args:
            metric_name: Name of the metric to evaluate
            predictions: Model predictions
            ground_truth: Ground truth values
            method_name: Name of the method being evaluated
            dataset: Name of the dataset being evaluated on
            
        Returns:
            MetricResult containing the evaluation results
        """
        if metric_name not in self.metrics:
            raise ValueError(f"Metric {metric_name} not found in registry")
            
        metric_fn = self.metrics[metric_name]
        value = metric_fn(predictions, ground_truth)
        
        result = MetricResult(
            metric_name=metric_name,
            value=value,
            method_name=method_name,
            dataset=dataset,
            timestamp=str(np.datetime64('now'))
        )
        
        self._save_result(result)
        return result
    
    def _save_result(self, result: MetricResult) -> None:
        """Save a metric result to the results file."""
        results = []
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                results = json.load(f)
        
        results.append({
            "metric_name": result.metric_name,
            "value": result.value,
            "method_name": result.method_name,
            "dataset": result.dataset,
            "timestamp": result.timestamp
        })
        
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def get_results(self, method_name: str = None, metric_name: str = None) -> List[MetricResult]:
        """Retrieve metric results with optional filtering.
        
        Args:
            method_name: Filter results by method name
            metric_name: Filter results by metric name
            
        Returns:
            List of MetricResult objects matching the filters
        """
        if not os.path.exists(self.results_file):
            return []
            
        with open(self.results_file, 'r') as f:
            results = json.load(f)
            
        filtered_results = []
        for result in results:
            if method_name and result['method_name'] != method_name:
                continue
            if metric_name and result['metric_name'] != metric_name:
                continue
                
            filtered_results.append(MetricResult(**result))
            
        return filtered_results

# Create a global registry instance
metric_registry = MetricRegistry()

# Add some common metrics
def accuracy(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """Calculate accuracy metric."""
    return np.mean(predictions == ground_truth)

def mse(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """Calculate mean squared error."""
    return np.mean((predictions - ground_truth) ** 2)

# Register default metrics
metric_registry.add_metric("accuracy", accuracy)
metric_registry.add_metric("mse", mse) 