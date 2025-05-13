import numpy as np
from config import MethodConfig, method_registry
from evaluation import metric_registry

def test_method_comparison():
    # Test adding a new method
    lora_config = MethodConfig(
        name="LoRA",
        parameters={"rank": 8, "alpha": 16},
        memory_requirements={"GPU": 8.0},
        training_time=2.5,
        hardware_requirements={"GPU": "A100"},
        best_use_cases=["Fine-tuning large language models"],
        limitations=["Requires careful rank selection"]
    )
    
    method_registry.add_new_method("LoRA", lora_config)
    
    # Test retrieving the method
    retrieved_config = method_registry.get_method("LoRA")
    assert retrieved_config.name == "LoRA"
    assert retrieved_config.parameters["rank"] == 8
    
    # Test adding a new metric
    def custom_metric(predictions, ground_truth):
        return np.mean(np.abs(predictions - ground_truth))
    
    metric_registry.add_metric("mae", custom_metric)
    
    # Test metric evaluation
    predictions = np.array([1, 2, 3, 4, 5])
    ground_truth = np.array([1, 2, 3, 4, 6])
    
    result = metric_registry.evaluate(
        metric_name="mae",
        predictions=predictions,
        ground_truth=ground_truth,
        method_name="LoRA",
        dataset="test_dataset"
    )
    
    assert result.metric_name == "mae"
    assert result.method_name == "LoRA"
    assert result.dataset == "test_dataset"
    
    # Test retrieving results
    results = metric_registry.get_results(method_name="LoRA")
    assert len(results) > 0
    assert results[0].method_name == "LoRA"

if __name__ == "__main__":
    test_method_comparison()
    print("All tests passed successfully!") 