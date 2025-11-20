"""
=============================================================================
HEAVILY ANNOTATED: LoRA Linear Layer Forward Pass
LINE-BY-LINE BREAKDOWN OF THE COMPUTATION
=============================================================================

File Location: src/peft/tuners/lora/layer.py (Lines 779-820)

This is THE MOST IMPORTANT function to understand in LoRA. It shows exactly
how the adapter weights are combined with the base weights during inference
and training.

=============================================================================
MATHEMATICAL BACKGROUND:
=============================================================================

Standard Linear Layer:
    y = W x + b
    where W ∈ ℝ^{out×in}, x ∈ ℝ^{in×batch}, b ∈ ℝ^{out}

LoRA Adaptation:
    y = W x + ΔW x + b
      = W x + (B A) x + b
      = W x + B(A x) + b

    where:
    - W: Original frozen weights (out_features × in_features)
    - A: Low-rank matrix A (r × in_features) - initialized ~U(-√(1/r), √(1/r))
    - B: Low-rank matrix B (out_features × r) - initialized to zeros
    - r: LoRA rank (typically 8, 16, 32, 64)
    - α: LoRA alpha (scaling parameter, typically 16, 32)
    - scaling = α / r (or α / √r for rsLoRA)

Key Insight: ΔW = B A has rank at most r, which is much smaller than
the rank of W. This is why we get parameter efficiency!

Example dimensions for a GPT-2 attention layer:
    W: (768 × 768) = 589,824 parameters (frozen)
    A: (8 × 768) = 6,144 parameters (trainable)
    B: (768 × 8) = 6,144 parameters (trainable)
    Total trainable: 12,288 parameters (2.08% of original)

=============================================================================
"""

import torch
import torch.nn as nn
from typing import Any, Optional


class Linear(nn.Module, LoraLayer):
    """
    LoRA-adapted Linear layer.

    This wraps a base nn.Linear layer and adds low-rank adaptation matrices.
    During forward pass, it computes: output = base_layer(x) + B(A(dropout(x))) * scaling
    """

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        =======================================================================
        FORWARD PASS: WHERE THE MAGIC HAPPENS
        =======================================================================

        This method implements the core LoRA computation. Let's break it down
        step by step, showing exactly where each matrix is accessed and how
        they're combined.

        INPUT:
            x: Input tensor, shape (batch_size, seq_len, in_features)
               e.g., (32, 512, 768) for GPT-2

        OUTPUT:
            result: Output tensor, shape (batch_size, seq_len, out_features)
                    e.g., (32, 512, 768) for GPT-2

        =======================================================================
        """

        # ===================================================================
        # STEP 1: FORWARD ARGUMENT VALIDATION
        # ===================================================================
        # WHAT: Check if adapter_names argument is valid
        # HOW: Validates batch size matches adapter_names length
        # WHY: For mixed-batch inference (different adapters per sample)
        # ===================================================================
        self._check_forward_args(x, *args, **kwargs)

        # ===================================================================
        # STEP 2: EXTRACT SPECIAL ARGUMENTS
        # ===================================================================
        # WHAT: Extract adapter_names from kwargs
        # HOW: Pop from kwargs to avoid passing to base_layer
        # WHY: base_layer (nn.Linear) doesn't understand these args
        # ===================================================================
        adapter_names = kwargs.pop("adapter_names", None)

        # ===================================================================
        # WHAT: Extract variant-specific kwargs (e.g., alora_offsets)
        # HOW: Pop variant kwargs and store separately
        # WHY: Some LoRA variants (DoRA, aLoRA) need extra information
        # ===================================================================
        variant_kwargs = {k: kwargs.pop(k, None) for k in VARIANT_KWARG_KEYS}

        # ===================================================================
        # BRANCH 1: ADAPTERS DISABLED
        # ===================================================================
        # WHAT: If disable_adapters flag is set, use only base model
        # HOW: Call base_layer directly, unmerge if needed
        # WHY: Allows temporarily disabling adapters without unloading them
        # ===================================================================
        if self.disable_adapters:
            if self.merged:
                # If adapters were merged into base weights, unmerge first
                self.unmerge()
            # Return pure base model output: y = W x + b
            result = self.base_layer(x, *args, **kwargs)

        # ===================================================================
        # BRANCH 2: MIXED-BATCH INFERENCE (Different adapters per sample)
        # ===================================================================
        # WHAT: Handle case where different samples use different adapters
        # HOW: Delegate to _mixed_batch_forward()
        # WHY: Enables batch inference with multiple adapters simultaneously
        # EXAMPLE: adapter_names = ["task_A", "task_B", "task_A", "task_B"]
        #          for a batch of 4 samples
        # ===================================================================
        elif adapter_names is not None:
            result = self._mixed_batch_forward(
                x, *args, adapter_names=adapter_names, **variant_kwargs, **kwargs
            )

        # ===================================================================
        # BRANCH 3: MERGED ADAPTERS (Adapters permanently merged into W)
        # ===================================================================
        # WHAT: If adapters are merged, just use base_layer
        # HOW: W' = W + B A, so base_layer already includes adapter
        # WHY: Faster inference - no extra computation needed
        # TRADE-OFF: Can't switch adapters or unload without reloading
        # ===================================================================
        elif self.merged:
            # Base layer already contains merged weights: W' = W + ΔW
            result = self.base_layer(x, *args, **kwargs)

        # ===================================================================
        # BRANCH 4: STANDARD LORA FORWARD (Most common case)
        # ===================================================================
        # WHAT: Standard LoRA computation with active adapters
        # HOW: Compute base output + adapter outputs
        # WHY: Keeps base and adapter weights separate for flexibility
        # ===================================================================
        else:
            # ===============================================================
            # STEP 4A: COMPUTE BASE MODEL OUTPUT
            # ===============================================================
            # WHAT: Forward pass through the frozen base layer
            # HOW: Standard nn.Linear computation: W x + b
            # WHERE: W is self.base_layer.weight (frozen)
            #        b is self.base_layer.bias (frozen, optional)
            # ===============================================================
            result = self.base_layer(x, *args, **kwargs)
            #
            # AT THIS POINT:
            # result = W x + b
            # shape: (batch_size, seq_len, out_features)
            #
            # WHERE W LIVES: self.base_layer.weight
            # - Shape: (out_features, in_features)
            # - Frozen: requires_grad = False
            # - Unchanged from original model
            # ===============================================================

            # Store original dtype for final conversion
            torch_result_dtype = result.dtype

            # ===============================================================
            # STEP 4B: ITERATE OVER ACTIVE ADAPTERS
            # ===============================================================
            # WHAT: Add contribution from each active adapter
            # HOW: Loop through self.active_adapters
            # WHY: Supports multiple adapters active simultaneously
            # NOTE: Usually just one adapter (["default"])
            # ===============================================================
            lora_A_keys = self.lora_A.keys()

            for active_adapter in self.active_adapters:
                # ===========================================================
                # STEP 4B-1: SKIP NON-EXISTENT ADAPTERS
                # ===========================================================
                # WHAT: Check if this layer has the active adapter
                # HOW: Look for adapter_name in lora_A ModuleDict
                # WHY: Not all layers have all adapters
                #      (e.g., target_modules filtering)
                # ===========================================================
                if active_adapter not in lora_A_keys:
                    continue

                # ===========================================================
                # STEP 4B-2: RETRIEVE ADAPTER COMPONENTS
                # ===========================================================
                # WHAT: Get the LoRA matrices and parameters for this adapter
                # WHERE THEY LIVE:
                # ===========================================================

                # --- Matrix A ---
                # WHERE: self.lora_A[active_adapter]
                # TYPE: nn.Linear(in_features, r, bias=False)
                # SHAPE: A.weight is (r, in_features)
                # INIT: Kaiming uniform (default) or custom init
                # TRAINABLE: Yes (requires_grad = True)
                lora_A = self.lora_A[active_adapter]

                # --- Matrix B ---
                # WHERE: self.lora_B[active_adapter]
                # TYPE: nn.Linear(r, out_features, bias=lora_bias)
                # SHAPE: B.weight is (out_features, r)
                # INIT: Zeros (default) - ensures ΔW = 0 before training
                # TRAINABLE: Yes (requires_grad = True)
                lora_B = self.lora_B[active_adapter]

                # --- Dropout Layer ---
                # WHERE: self.lora_dropout[active_adapter]
                # TYPE: nn.Dropout(p=lora_dropout) or nn.Identity()
                # PURPOSE: Regularization during training
                # NOTE: nn.Identity() if lora_dropout=0.0
                dropout = self.lora_dropout[active_adapter]

                # --- Scaling Factor ---
                # WHERE: self.scaling[active_adapter]
                # TYPE: float
                # VALUE: lora_alpha / r  (standard LoRA)
                #        or lora_alpha / sqrt(r)  (rsLoRA)
                # PURPOSE: Controls the magnitude of adapter contribution
                # TYPICAL: If r=8, alpha=16, then scaling=2.0
                scaling = self.scaling[active_adapter]

                # ===========================================================
                # STEP 4B-3: INPUT DTYPE CASTING
                # ===========================================================
                # WHAT: Cast input to match lora_A weight dtype
                # WHY: Ensures dtype compatibility for matmul
                # EXAMPLE: If model is bf16 but adapters are fp32
                # ===========================================================
                x = self._cast_input_dtype(x, lora_A.weight.dtype)

                # ===========================================================
                # STEP 4B-4: VANILLA LORA COMPUTATION
                # ===========================================================
                # WHAT: Compute and add the adapter contribution
                # HOW: result += B(A(dropout(x))) * scaling
                # WHY: This is the core LoRA equation!
                # ===========================================================

                if active_adapter not in self.lora_variant:
                    # =======================================================
                    # THE CORE LORA COMPUTATION - LINE BY LINE
                    # =======================================================

                    # STEP 1: Apply dropout to input
                    # ------------------------------------------------------
                    # dropped_x = dropout(x)
                    # - Only active during training (dropout.training=True)
                    # - During inference, this is identity function
                    # - Shape: same as x

                    # STEP 2: Multiply by matrix A
                    # ------------------------------------------------------
                    # a_out = lora_A(dropped_x)
                    # - Internally: a_out = dropped_x @ A.T
                    # - A.weight shape: (r, in_features)
                    # - After transpose: (in_features, r)
                    # - Output shape: (batch_size, seq_len, r)
                    # WHERE A LIVES: lora_A.weight
                    # - Trainable parameter
                    # - Initialized with Kaiming uniform
                    # - Learns to extract r-dimensional features

                    # STEP 3: Multiply by matrix B
                    # ------------------------------------------------------
                    # b_out = lora_B(a_out)
                    # - Internally: b_out = a_out @ B.T
                    # - B.weight shape: (out_features, r)
                    # - After transpose: (r, out_features)
                    # - Output shape: (batch_size, seq_len, out_features)
                    # WHERE B LIVES: lora_B.weight
                    # - Trainable parameter
                    # - Initialized to zeros
                    # - Learns to project back to output space

                    # STEP 4: Scale the output
                    # ------------------------------------------------------
                    # scaled_out = b_out * scaling
                    # - scaling = alpha / r (or alpha / sqrt(r))
                    # - Controls magnitude of adapter contribution
                    # - Typically makes adapter contribution significant

                    # STEP 5: Add to base model output
                    # ------------------------------------------------------
                    # result = result + scaled_out
                    # - Combines base model output with adapter output
                    # - This is WHERE THE ADDITION HAPPENS
                    # - result already contains W x + b
                    # - scaled_out contains B(A(dropout(x))) * scaling

                    # ALL IN ONE LINE:
                    result = result + lora_B(lora_A(dropout(x))) * scaling

                    # =======================================================
                    # MATHEMATICAL BREAKDOWN:
                    # =======================================================
                    #
                    # Before this line:
                    #   result = W x + b
                    #
                    # This line computes:
                    #   ΔW x = (B A)(dropout(x)) * scaling
                    #        = B (A (dropout(x))) * scaling
                    #        = lora_B(lora_A(dropout(x))) * scaling
                    #
                    # After this line:
                    #   result = W x + b + ΔW x
                    #          = W x + b + (B A)(dropout(x)) * scaling
                    #          = (W + B A) x + b  (approximately, ignoring dropout)
                    #
                    # DIMENSIONS EXAMPLE (GPT-2 attention):
                    #   x: (32, 512, 768)
                    #   dropout(x): (32, 512, 768)
                    #   A: (8, 768) → A(dropout(x)): (32, 512, 8)
                    #   B: (768, 8) → B(A(dropout(x))): (32, 512, 768)
                    #   scaling: 2.0 (if r=8, alpha=16)
                    #   final: (32, 512, 768)
                    #
                    # MATRIX ACCESS LOCATIONS:
                    #   Base W: self.base_layer.weight (frozen, ℝ^{768×768})
                    #   Matrix A: self.lora_A[adapter].weight (trainable, ℝ^{8×768})
                    #   Matrix B: self.lora_B[adapter].weight (trainable, ℝ^{768×8})
                    # =======================================================

                else:
                    # =======================================================
                    # LORA VARIANT COMPUTATION (DoRA, aLoRA, Arrow, etc.)
                    # =======================================================
                    # WHAT: Use variant-specific forward method
                    # HOW: Delegate to self.lora_variant[adapter].forward()
                    # WHY: Variants modify the standard LoRA computation
                    #
                    # VARIANTS:
                    # - DoRA: Weight-decomposed LoRA (magnitude + direction)
                    # - aLoRA: Activated LoRA (token-selective activation)
                    # - Arrow: Routing-based adapter combination
                    # =======================================================
                    result = self.lora_variant[active_adapter].forward(
                        self,
                        active_adapter=active_adapter,
                        x=x,
                        result=result,
                        **variant_kwargs,
                        **kwargs,
                    )

            # ===============================================================
            # STEP 4C: RESTORE ORIGINAL DTYPE
            # ===============================================================
            # WHAT: Cast result back to original dtype
            # WHY: Adapters might use different dtype (e.g., fp32) than
            #      base model (e.g., bf16) for training stability
            # ===============================================================
            result = result.to(torch_result_dtype)

        # ===================================================================
        # STEP 5: RETURN FINAL OUTPUT
        # ===================================================================
        # WHAT: Return the computed result
        # VALUE: W x + b + Σ(B_i A_i dropout(x) * scaling_i)
        #        for all active adapters i
        # ===================================================================
        return result


# =============================================================================
# KEY TAKEAWAYS FROM THE FORWARD PASS:
# =============================================================================
#
# 1. **MATRIX LOCATIONS**:
#    - Base weight W: self.base_layer.weight (frozen)
#    - LoRA matrix A: self.lora_A[adapter_name].weight (trainable)
#    - LoRA matrix B: self.lora_B[adapter_name].weight (trainable)
#
# 2. **COMPUTATION ORDER**:
#    - First: Compute base output (W x + b)
#    - Then: Compute adapter output (B(A(dropout(x))) * scaling)
#    - Finally: Add them together
#
# 3. **EFFICIENT COMPUTATION**:
#    - Don't compute B A explicitly (would be out_features × in_features)
#    - Instead compute B(A x), which only uses rank r intermediates
#    - This saves both memory and computation
#
# 4. **PARAMETER EFFICIENCY**:
#    - Base model: 589,824 params frozen
#    - LoRA adapter: 12,288 params trainable (2.08%)
#    - Same expressive power for downstream tasks!
#
# 5. **FLEXIBILITY**:
#    - Can disable adapters (disable_adapters=True)
#    - Can merge adapters into base (merge())
#    - Can use multiple adapters simultaneously
#    - Can switch adapters without reloading model
#
# 6. **INITIALIZATION ENSURES NO-OP**:
#    - B initialized to zeros → B A = 0 before training
#    - Ensures LoRA starts as identity transformation
#    - Model behaves exactly like base model before training
#
# =============================================================================
# COMPARISON WITH FULL FINE-TUNING:
# =============================================================================
#
# Full Fine-Tuning:
#   - All parameters trainable
#   - Updates W directly: W_new = W_old + ΔW
#   - ΔW is full rank (expensive to store and compute)
#
# LoRA:
#   - W stays frozen
#   - Learns low-rank ΔW = B A
#   - Only B and A trainable (much smaller)
#   - Equivalent representation power for most tasks
#
# Memory Comparison (GPT-2 attention layer):
#   - Full fine-tuning: 589,824 * 4 bytes = 2.36 MB per layer
#   - LoRA (r=8): 12,288 * 4 bytes = 0.05 MB per layer
#   - Reduction: 97.9%!
#
# =============================================================================
