"""
=============================================================================
HEAVILY ANNOTATED: LoraLayer.merge() - Permanently Merge Adapters
=============================================================================

File Location: src/peft/tuners/lora/layer.py:655-777

WHAT: Merges adapter weights permanently into base model weights.
      After merging: W' = W + B @ A * scaling

WHY: Inference optimization - eliminates adapter computation overhead
     After merge, model behaves identically but forward pass is faster

TRADE-OFF: Can't switch/unload adapters after merging without reloading

CALL CHAIN:
    model.merge_and_unload()
        → _unload_and_optionally_merge(merge=True)
            → target.merge() ← YOU ARE HERE (for each LoRA layer)

CRITICAL EQUATION:
    W' = W + ΔW
    W' = W + B @ A * scaling

After merge: forward(x) = W' x instead of W x + B(A x) * scaling
"""

import torch
from torch import nn


class Linear(nn.Module, LoraLayer):
    """LoRA-adapted Linear layer"""

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None):
        """
        Merge active adapter weights into base weights.

        PARAMETERS:
            safe_merge: If True, check for NaNs before merging
            adapter_names: Which adapters to merge (None = all active)

        AFTER MERGE:
            base_layer.weight contains W + B @ A * scaling
            Adapter still exists but is marked as merged
            Forward pass uses only base_layer (faster!)
        """

        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return  # Nothing to merge

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()

                if safe_merge:
                    # ===========================================================
                    # SAFE MERGE PATH - Check for NaNs
                    # ===========================================================
                    # Clone weights to test merge without modifying
                    orig_weight = base_layer.weight.data.clone()
                    orig_dtype = orig_weight.dtype

                    if active_adapter not in self.lora_variant:
                        # Standard LoRA: compute ΔW = B @ A * scaling
                        delta_weight = self.get_delta_weight(active_adapter)
                        orig_weight += delta_weight.to(orig_dtype)
                    else:
                        # Variant (DoRA, etc): use variant merge
                        orig_weight = self.lora_variant[active_adapter].merge_safe(
                            self, active_adapter, orig_weight
                        )

                    # Check for NaNs
                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in merged weights. "
                            f"Adapter {active_adapter} seems to be broken"
                        )

                    # Safe - apply merge
                    base_layer.weight.data = orig_weight

                    # Handle bias if using lora_bias
                    if self.lora_bias[active_adapter]:
                        if getattr(base_layer, "bias", None) is None:
                            raise RuntimeError(
                                "Cannot merge LoRA with lora_bias=True "
                                "because base layer has no bias"
                            )
                        new_bias = (
                            base_layer.bias
                            + self.lora_B[active_adapter].bias * self.scaling[active_adapter]
                        )
                        if not torch.isfinite(new_bias).all():
                            raise ValueError(f"NaNs in bias for adapter {active_adapter}")
                        base_layer.weight.bias = new_bias.to(orig_dtype)

                else:
                    # ===========================================================
                    # FAST MERGE PATH - No safety checks
                    # ===========================================================
                    if active_adapter not in self.lora_variant:
                        # *** COMPUTE AND ADD ΔW ***
                        delta_weight = self.get_delta_weight(active_adapter)
                        base_layer.weight.data += delta_weight
                    else:
                        # Variant merge
                        self.lora_variant[active_adapter].merge_unsafe(
                            self, active_adapter, base_layer.weight
                        )

                    # Handle bias
                    if self.lora_bias[active_adapter]:
                        if getattr(base_layer, "bias", None) is None:
                            raise RuntimeError("Cannot merge LoRA bias - base has no bias")
                        base_layer.bias.data += (
                            self.lora_B[active_adapter].bias * self.scaling[active_adapter]
                        )

                # Mark adapter as merged
                self.merged_adapters.append(active_adapter)


    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute ΔW = B @ A * scaling for the given adapter.

        This is THE CORE COMPUTATION for merging!

        MATH:
            ΔW = B @ A * scaling

        WHERE:
            B: self.lora_B[adapter].weight (out_features × r)
            A: self.lora_A[adapter].weight (r × in_features)
            B @ A: (out_features × in_features) - same shape as W!
            scaling: self.scaling[adapter] (alpha / r)

        EXAMPLE (GPT-2 attention):
            B: (2304 × 8)
            A: (8 × 768)
            B @ A: (2304 × 768) - same as original W!
            scaling: 2.0
            ΔW = B @ A * 2.0
        """

        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # Handle CPU bfloat16/float16 (slow matmul)
        cast_to_fp32 = device.type == "cpu" and (dtype in [torch.float16, torch.bfloat16])

        # Get A and B weights
        weight_A = self.lora_A[adapter].weight  # (r × in_features)
        weight_B = self.lora_B[adapter].weight  # (out_features × r)

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # *** COMPUTE ΔW = B @ A * scaling ***
        # Transpose if fan_in_fan_out (for Conv1D layers)
        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
        # Shape: (out_features × in_features) * scalar = (out_features × in_features)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)
            # Cast weights back
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor
        # RETURNS: ΔW with same shape as base layer weight


# =============================================================================
# SUMMARY
# =============================================================================
# merge() permanently adds ΔW to base weights:
#   W' = W + B @ A * scaling
#
# After merge:
#   - Base weights contain adapted weights
#   - Forward pass faster (no adapter computation)
#   - Can't switch adapters without reloading
#   - Can save as regular model (not PEFT)
#
# Use cases:
#   - Production deployment (inference only)
#   - Creating fine-tuned base model
#   - Eliminating adapter overhead
#
# See: guides/01_model_injection_deep_dive.md
# =============================================================================
