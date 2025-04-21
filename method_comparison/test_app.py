import pytest
import pandas as pd
import numpy as np
from app import build_app

# Create sample data for testing
def create_sample_data():
    data = {
        "task_name": ["MetaMathQA"] * 6,
        "model_id": ["model1"] * 6,
        "peft_type": ["LoRA", "LoRA", "LoRA-FA", "LoRA-FA", "Bone", "Bone"],
        "test_accuracy": [0.85, 0.87, 0.88, 0.89, 0.86, 0.84],
        "cuda_memory_max": [8.0, 8.2, 7.5, 7.8, 9.0, 9.2],
        "total_time": [2.5, 2.6, 2.3, 2.4, 2.8, 2.9],
        "experiment_name": ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6"],
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_df():
    return create_sample_data()

def test_app_initialization(sample_df):
    demo = build_app(sample_df)
    assert demo is not None

def test_pareto_plot(sample_df):
    from app import compute_pareto_frontier, generate_pareto_plot
    
    # Test Pareto frontier computation
    pareto_df = compute_pareto_frontier(sample_df, "cuda_memory_max", "test_accuracy")
    assert not pareto_df.empty
    assert len(pareto_df) <= len(sample_df)
    
    # Test Pareto plot generation
    fig = generate_pareto_plot(sample_df, "cuda_memory_max", "test_accuracy")
    assert fig is not None
    assert "data" in fig

def test_method_filtering(sample_df):
    from app import filter_data
    
    # Test filtering by task and model
    filtered = filter_data("MetaMathQA", "model1", sample_df)
    assert not filtered.empty
    assert len(filtered) == len(sample_df)
    
    # Test filtering with non-existent task/model
    filtered = filter_data("NonExistentTask", "model1", sample_df)
    assert filtered.empty

def test_model_ids(sample_df):
    from app import get_model_ids
    
    # Test getting model IDs for a task
    model_ids = get_model_ids("MetaMathQA", sample_df)
    assert len(model_ids) == 1
    assert "model1" in model_ids
    
    # Test with non-existent task
    model_ids = get_model_ids("NonExistentTask", sample_df)
    assert len(model_ids) == 0

def test_pareto_summary(sample_df):
    from app import compute_pareto_frontier, compute_pareto_summary
    
    # Test summary computation
    pareto_df = compute_pareto_frontier(sample_df, "cuda_memory_max", "test_accuracy")
    summary = compute_pareto_summary(sample_df, pareto_df, "cuda_memory_max", "test_accuracy")
    assert isinstance(summary, str)
    assert "Total points" in summary
    assert "Pareto frontier points" in summary

if __name__ == "__main__":
    pytest.main([__file__]) 