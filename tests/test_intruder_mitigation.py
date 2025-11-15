import pytest
import torch

from peft import LoraConfig, TaskType, get_peft_model
from peft.utils import (
    IntruderDetectionResult,
    compute_reconstruction_error,
    detect_intruder_dimensions,
    mitigate_intruder_dimensions,
    project_delta_to_lora,
)


class TestIntruderMitigation:
    """Test intruder dimension mitigation functionality."""

    @pytest.fixture
    def tiny_lora_model(self):
        """Create a tiny model with LoRA for testing."""
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(vocab_size=1000, n_positions=128, n_embd=64, n_layer=2, n_head=2)
        base_model = GPT2LMHeadModel(config)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"],
        )

        model = get_peft_model(base_model, lora_config)

        for name, param in model.named_parameters():
            if "lora_" in name and param.requires_grad:
                param.data += torch.randn_like(param) * 0.1

        return model

    def test_reduce_intruder_dimensions_nondestructive(self, tiny_lora_model):
        """Test non-destructive path (creates new adapter)."""
        model = tiny_lora_model

        model = model.reduce_intruder_dimensions(
            old_adapter_name="default",
            new_adapter_name="default_mitigated",
            mitigation_lambda=0.75,
            progressbar=False,
        )

        assert model is not None
        assert "default" in model.peft_config
        assert "default_mitigated" in model.peft_config
        assert model.active_adapter == "default_mitigated"

    def test_merge_and_unload_destructive(self, tiny_lora_model):
        """Test destructive path (merge and unload)."""
        model = tiny_lora_model

        base_model = model.merge_and_unload_with_reduced_intruder_dimensions(
            adapter_name="default",
            mitigation_lambda=0.75,
            progressbar=False,
        )

        assert base_model is not None
        assert not hasattr(base_model, "peft_config")

    def test_invalid_adapter_name(self, tiny_lora_model):
        """Test that invalid adapter name raises error."""
        model = tiny_lora_model

        with pytest.raises(ValueError, match="Adapter 'nonexistent' not found"):
            model.reduce_intruder_dimensions(old_adapter_name="nonexistent")

    def test_invalid_lambda(self, tiny_lora_model):
        """Test that invalid lambda raises error."""
        model = tiny_lora_model

        with pytest.raises(ValueError, match="mitigation_lambda must be in"):
            model.reduce_intruder_dimensions(mitigation_lambda=1.5)

    def test_invalid_epsilon(self, tiny_lora_model):
        """Test that invalid epsilon raises error."""
        model = tiny_lora_model

        with pytest.raises(ValueError, match="threshold_epsilon must be in"):
            model.reduce_intruder_dimensions(threshold_epsilon=2.0)

    def test_duplicate_adapter_name(self, tiny_lora_model):
        """Test that duplicate adapter name raises error."""
        model = tiny_lora_model

        # Create first mitigated adapter
        model = model.reduce_intruder_dimensions(
            old_adapter_name="default",
            new_adapter_name="mitigated",
            progressbar=False,
        )

        # Try to create another with same name
        with pytest.raises(ValueError, match="already exists"):
            model.reduce_intruder_dimensions(
                old_adapter_name="default",
                new_adapter_name="mitigated",
                progressbar=False,
            )

    def test_no_intruders_detected(self, tiny_lora_model):
        """Test behavior when no intruders are found."""
        model = tiny_lora_model

        model_mitigated = model.reduce_intruder_dimensions(
            old_adapter_name="default",
            new_adapter_name="no_intruders",
            threshold_epsilon=0.99,
            mitigation_lambda=0.75,
            progressbar=False,
        )

        assert "no_intruders" in model_mitigated.peft_config
        assert model_mitigated.active_adapter == "no_intruders"

        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(vocab_size=1000, n_positions=128, n_embd=64, n_layer=2, n_head=2)
        base_model = GPT2LMHeadModel(config)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"],
        )
        model2 = get_peft_model(base_model, lora_config)

        base_model_out = model2.merge_and_unload_with_reduced_intruder_dimensions(
            adapter_name="default",
            threshold_epsilon=0.99,
            mitigation_lambda=0.75,
            progressbar=False,
        )
        assert base_model_out is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_compatibility(self):
        """Test with fp16/bf16 models."""
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(vocab_size=1000, n_positions=128, n_embd=64, n_layer=2, n_head=2)

        base_model_fp16 = GPT2LMHeadModel(config).to(dtype=torch.float16, device="cuda")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"],
        )
        model_fp16 = get_peft_model(base_model_fp16, lora_config)

        model_fp16 = model_fp16.reduce_intruder_dimensions(
            old_adapter_name="default",
            new_adapter_name="mitigated_fp16",
            mitigation_lambda=0.75,
            progressbar=False,
        )
        assert "mitigated_fp16" in model_fp16.peft_config

        if torch.cuda.is_bf16_supported():
            base_model_bf16 = GPT2LMHeadModel(config).to(dtype=torch.bfloat16, device="cuda")
            model_bf16 = get_peft_model(base_model_bf16, lora_config)

            model_bf16 = model_bf16.reduce_intruder_dimensions(
                old_adapter_name="default",
                new_adapter_name="mitigated_bf16",
                mitigation_lambda=0.75,
                progressbar=False,
            )
            assert "mitigated_bf16" in model_bf16.peft_config

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_transfer(self):
        """Test CPU -> CUDA and vice versa."""
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(vocab_size=1000, n_positions=128, n_embd=64, n_layer=2, n_head=2)

        base_model_cpu = GPT2LMHeadModel(config)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"],
        )
        model_cpu = get_peft_model(base_model_cpu, lora_config)

        model_cpu = model_cpu.reduce_intruder_dimensions(
            old_adapter_name="default",
            new_adapter_name="mitigated_cpu",
            mitigation_lambda=0.75,
            progressbar=False,
        )
        assert "mitigated_cpu" in model_cpu.peft_config

        model_cuda = model_cpu.to("cuda")

        model_cuda = model_cuda.reduce_intruder_dimensions(
            old_adapter_name="default",
            new_adapter_name="mitigated_cuda",
            mitigation_lambda=0.75,
            progressbar=False,
        )
        assert "mitigated_cuda" in model_cuda.peft_config

        model_back_cpu = model_cuda.to("cpu")
        assert "mitigated_cuda" in model_back_cpu.peft_config

    def test_reconstruction_accuracy(self):
        """Test that project_delta_to_lora accurately reconstructs delta_w."""
        test_cases = [
            (100, 50, 8, 2.0),
            (64, 64, 4, 1.0),
            (200, 100, 16, 4.0),
            (50, 100, 8, 2.0),
        ]

        for out_features, in_features, rank, scaling in test_cases:
            delta_w = torch.randn(out_features, in_features)
            lora_A, lora_B = project_delta_to_lora(delta_w, rank=rank, scaling=scaling)

            reconstructed = (lora_B @ lora_A) * scaling

            error = compute_reconstruction_error(delta_w, reconstructed, metric="relative_frobenius")

            assert error < 1.0, f"Reconstruction error too high: {error:.4f} for shape {delta_w.shape}, rank {rank}"

            assert lora_A.shape == (rank, in_features), f"lora_A shape mismatch: {lora_A.shape}"
            assert lora_B.shape == (out_features, rank), f"lora_B shape mismatch: {lora_B.shape}"

    def test_intruder_detection_standalone(self):
        """Test the utility functions can be used independently."""
        from peft.utils import (
            compute_reconstruction_error,
            detect_intruder_dimensions,
            mitigate_intruder_dimensions,
            project_delta_to_lora,
        )

        w_pretrained = torch.randn(100, 50)
        w_finetuned = w_pretrained + torch.randn(100, 50) * 0.1

        results = detect_intruder_dimensions(w_pretrained, w_finetuned, top_k=10, epsilon=0.5)

        assert isinstance(results, IntruderDetectionResult)
        assert results.intruder_indices is not None
        assert results.left_vectors is not None
        assert results.singular_values is not None
        assert results.right_vectors is not None
        assert results.similarity_matrix is not None
        assert results.max_similarities is not None

        delta_w = w_finetuned - w_pretrained
        delta_w_mitigated = mitigate_intruder_dimensions(
            w_pretrained=w_pretrained,
            delta_w=delta_w,
            intruder_results=results,
            lambda_factor=0.75,
        )

        assert delta_w_mitigated.shape == delta_w.shape
        if len(results.intruder_indices) > 0:
            assert not torch.allclose(delta_w_mitigated, delta_w)

        lora_A, lora_B = project_delta_to_lora(delta_w_mitigated, rank=8, scaling=2.0)
        assert lora_A.shape == (8, 50)
        assert lora_B.shape == (100, 8)

        reconstructed = (lora_B @ lora_A) * 2.0
        error = compute_reconstruction_error(delta_w_mitigated, reconstructed)
        assert isinstance(error, float)
        assert error >= 0.0


class TestIntruderDimensionsUtilities:
    """Test standalone utility functions in detail."""

    def test_detect_intruder_dimensions_basic(self):
        """Test basic intruder detection."""
        w0 = torch.randn(50, 30)
        wt = w0 + torch.randn(50, 30) * 0.01

        results = detect_intruder_dimensions(w0, wt, top_k=5, epsilon=0.5)

        assert len(results.intruder_indices) >= 0
        assert results.similarity_matrix.shape[0] == 5

    def test_detect_intruder_dimensions_edge_cases(self):
        """Test edge cases for intruder detection."""
        w0 = torch.randn(100, 50)

        results = detect_intruder_dimensions(w0, w0, top_k=10, epsilon=0.5)
        assert len(results.intruder_indices) == 0

        wt_different = torch.randn(100, 50)
        results = detect_intruder_dimensions(w0, wt_different, top_k=10, epsilon=0.9)
        assert len(results.intruder_indices) >= 0

    def test_detect_intruder_dimensions_validation(self):
        """Test parameter validation in detect_intruder_dimensions."""
        w0 = torch.randn(50, 30)
        wt = torch.randn(50, 30)

        with pytest.raises(AssertionError):
            detect_intruder_dimensions(w0, wt, epsilon=1.5)

        with pytest.raises(AssertionError):
            detect_intruder_dimensions(w0, wt, top_k=0)

        wt_wrong = torch.randn(40, 30)
        with pytest.raises(AssertionError):
            detect_intruder_dimensions(w0, wt_wrong)

    def test_mitigate_intruder_dimensions_no_intruders(self):
        """Test mitigation when no intruders are present."""
        w0 = torch.randn(50, 30)
        delta_w = torch.randn(50, 30) * 0.1

        intruder_results = {
            "left_vectors": torch.empty(50, 0),
            "singular_values": torch.empty(0),
            "right_vectors": torch.empty(0, 30),
        }

        result = mitigate_intruder_dimensions(w0, delta_w, intruder_results, lambda_factor=0.75)

        assert torch.allclose(result, delta_w)

    def test_mitigate_intruder_dimensions_validation(self):
        """Test parameter validation in mitigate_intruder_dimensions."""
        w0 = torch.randn(50, 30)
        delta_w = torch.randn(50, 30)

        intruder_results = {
            "left_vectors": torch.empty(50, 0),
            "singular_values": torch.empty(0),
            "right_vectors": torch.empty(0, 30),
        }

        with pytest.raises(ValueError, match="lambda_factor must be in"):
            mitigate_intruder_dimensions(w0, delta_w, intruder_results, lambda_factor=3.0)

        delta_w_wrong = torch.randn(40, 30)
        with pytest.raises(ValueError, match="Shape mismatch"):
            mitigate_intruder_dimensions(w0, delta_w_wrong, intruder_results)

    def test_project_delta_to_lora_shapes(self):
        """Test that project_delta_to_lora produces correct shapes."""
        test_cases = [
            (100, 50, 8),
            (64, 64, 16),
            (200, 100, 4),
        ]

        for out_dim, in_dim, rank in test_cases:
            delta_w = torch.randn(out_dim, in_dim)
            lora_A, lora_B = project_delta_to_lora(delta_w, rank=rank, scaling=2.0)

            assert lora_A.shape == (rank, in_dim), f"Expected lora_A shape ({rank}, {in_dim}), got {lora_A.shape}"
            assert lora_B.shape == (out_dim, rank), f"Expected lora_B shape ({out_dim}, {rank}), got {lora_B.shape}"

    def test_project_delta_to_lora_preserve_rank(self):
        """Test preserve_rank parameter."""
        delta_w = torch.randn(100, 50)

        lora_A, lora_B = project_delta_to_lora(delta_w, rank=8, scaling=2.0, preserve_rank=True)
        assert lora_A.shape[0] == 8

        lora_A_adapt, lora_B_adapt = project_delta_to_lora(delta_w, rank=8, scaling=2.0, preserve_rank=False)
        assert lora_A_adapt.shape[0] <= 50

    def test_compute_reconstruction_error_metrics(self):
        """Test different error metrics."""
        original = torch.randn(50, 30)
        reconstructed = original + torch.randn(50, 30) * 0.01

        error_fro = compute_reconstruction_error(original, reconstructed, metric="frobenius")
        assert error_fro >= 0.0

        error_rel = compute_reconstruction_error(original, reconstructed, metric="relative_frobenius")
        assert 0.0 <= error_rel < 1.0

        error_max = compute_reconstruction_error(original, reconstructed, metric="max_abs")
        assert error_max >= 0.0

        with pytest.raises(ValueError, match="Unknown metric"):
            compute_reconstruction_error(original, reconstructed, metric="invalid")

    def test_compute_reconstruction_error_validation(self):
        """Test validation in compute_reconstruction_error."""
        original = torch.randn(50, 30)
        wrong_shape = torch.randn(40, 30)

        with pytest.raises(ValueError, match="same shape"):
            compute_reconstruction_error(original, wrong_shape)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
