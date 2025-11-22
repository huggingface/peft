"""
=============================================================================
HEAVILY ANNOTATED: LoraLayer.update_layer() - Creating A and B Matrices
=============================================================================

File Location: src/peft/tuners/lora/layer.py:144-235

WHAT: This function creates and initializes the LoRA adapter matrices (A and B)
      and adds them to the layer's ModuleDict.

HOW: Creates nn.Linear modules for A and B, initializes weights, computes scaling

WHY: This is WHERE THE ADAPTER PARAMETERS ARE BORN!
     The A and B matrices that will be trained are allocated here.

CALL CHAIN:
    _create_and_replace()
        → _create_new_module() → LoraLinear.__init__()
            → update_layer() ← YOU ARE HERE

OR (multi-adapter):
    _create_and_replace()
        → update_layer() ← YOU ARE HERE (for existing LoRA layers)

CRITICAL INSIGHT:
    After this function:
    - self.lora_A[adapter_name] contains trainable matrix A (r × in_features)
    - self.lora_B[adapter_name] contains trainable matrix B (out_features × r)
    - These are stored in ModuleDicts for multi-adapter support

=============================================================================
"""

import math
import warnings
import torch
from torch import nn
from typing import Union


class LoraLayer:
    """Base class for LoRA-enhanced layers"""

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        use_alora: bool = False,
        use_qalora: bool = False,
        lora_bias: bool = False,
        arrow_config: ArrowConfig = None,
        qalora_group_size: int = 32,
        inference_mode: bool = False,
        **kwargs,
    ):
        """
        Add a new adapter to this LoRA layer or update existing configuration.

        =======================================================================
        PARAMETERS EXPLAINED:
        =======================================================================

        adapter_name: str
            Name of the adapter (e.g., "default", "task_A")
            Used as key in lora_A, lora_B ModuleDicts

        r: int
            Rank of the adapter matrices
            Determines size: A is (r × in_features), B is (out_features × r)
            EXAMPLE: r=8 for GPT-2 attention → A:(8×768), B:(2304×8)

        lora_alpha: int/float
            Scaling parameter (not rank!)
            Used to compute scaling = alpha / r (or alpha / √r)
            EXAMPLE: alpha=16, r=8 → scaling=2.0

        lora_dropout: float
            Dropout probability (0.0 to 1.0)
            Applied to input before matrix A
            0.0 means no dropout (uses nn.Identity instead)

        init_lora_weights: bool or str
            How to initialize A and B matrices
            - True: Standard (B=0, A=Kaiming)
            - "gaussian": Gaussian initialization
            - "pissa": PiSSA initialization
            - "olora": OLoRA initialization
            - "loftq": LoftQ quantization-aware
            - "eva": EVA data-driven
            - etc.

        use_rslora: bool
            Use rank-stabilized LoRA scaling (alpha / √r instead of alpha / r)
            Better for low ranks

        use_dora: bool
            Enable DoRA (weight-decomposed LoRA)
            Adds magnitude vector for better low-rank performance

        use_alora: bool
            Enable aLoRA (activated LoRA)
            Token-selective adapter activation

        use_qalora: bool
            Enable QALoRA (quantization-aware LoRA)
            For GPTQ quantized models

        lora_bias: bool
            Add bias term to B matrix
            Normally False, True for extracted LoRA from full fine-tuning

        =======================================================================
        """

        # ===================================================================
        # STEP 1: COLLECT ALL KWARGS
        # ===================================================================
        # WHAT: Save all parameters for later use
        # WHY: Some initialization methods need access to all config
        # ===================================================================
        kwargs = locals().copy()
        del kwargs["self"]

        # ===================================================================
        # STEP 2: VALIDATION
        # ===================================================================

        # -------------------------------------------------------------------
        # Validate Rank
        # -------------------------------------------------------------------
        # WHAT: Ensure rank is positive
        # WHY: Rank 0 or negative makes no sense
        # -------------------------------------------------------------------
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        # -------------------------------------------------------------------
        # Validate lora_bias
        # -------------------------------------------------------------------
        # WHAT: Warn if lora_bias=True but base layer has no bias
        # WHY: Can't merge LoRA with bias if base has none
        # -------------------------------------------------------------------
        if lora_bias and (getattr(self.get_base_layer(), "bias", None) is None):
            warnings.warn(
                f"`lora_bias=True` was passed but the targeted layer of type "
                f"{type(self.get_base_layer()).__name__} has no bias. "
                "This means that merging LoRA weights won't be possible.",
                PeftWarning,
            )

        # ===================================================================
        # STEP 3: RESOLVE LORA VARIANT (DoRA, aLoRA, Arrow, etc.)
        # ===================================================================
        # WHAT: Determine if using a LoRA variant
        # HOW: Check use_dora, use_alora, arrow_config flags
        # WHY: Variants need special handling in forward pass
        # ===================================================================
        lora_variant = self.resolve_lora_variant(
            use_dora=use_dora,
            use_alora=use_alora,
            use_qalora=use_qalora,
            qalora_group_size=qalora_group_size,
            arrow_config=arrow_config,
        )
        # RETURNS:
        # - None: Standard LoRA
        # - DoraLinearVariant: DoRA decomposition
        # - ALoraLinearVariant: Activated LoRA
        # - ArrowLinearVariant: Arrow routing
        # - QALoraLinearVariant: QALoRA quantization-aware

        if lora_variant is not None:
            # Store variant for this adapter
            self.lora_variant[adapter_name] = lora_variant
            # EXAMPLE: self.lora_variant = {"default": DoraLinearVariant()}

        # ===================================================================
        # STEP 4: STORE ADAPTER CONFIGURATION
        # ===================================================================
        # WHAT: Save adapter-specific config in dictionaries
        # WHY: Each adapter can have different rank, alpha, etc.
        # ===================================================================

        # -------------------------------------------------------------------
        # Store Rank
        # -------------------------------------------------------------------
        self.r[adapter_name] = r
        # EXAMPLE: self.r = {"default": 8, "task_A": 16}

        # -------------------------------------------------------------------
        # Store Alpha
        # -------------------------------------------------------------------
        self.lora_alpha[adapter_name] = lora_alpha
        # EXAMPLE: self.lora_alpha = {"default": 16, "task_A": 32}

        # ===================================================================
        # STEP 5: CREATE DROPOUT LAYER
        # ===================================================================
        # WHAT: Create dropout module for this adapter
        # HOW: nn.Dropout if lora_dropout > 0, else nn.Identity
        # WHY: Regularization during training
        # ===================================================================
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            # No dropout - use identity function
            lora_dropout_layer = nn.Identity()

        # Store in ModuleDict
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # EXAMPLE: self.lora_dropout = {
        #     "default": Dropout(p=0.1),
        #     "task_A": Identity(),
        # }

        # ===================================================================
        # STEP 6: CREATE ADAPTER MATRICES A AND B
        # ===================================================================
        # *** THIS IS WHERE THE MAGIC HAPPENS ***
        # WHAT: Allocate the trainable LoRA parameters
        # HOW: Create nn.Linear modules for A and B
        # WHY: These are the parameters that will be trained!
        # ===================================================================

        # -------------------------------------------------------------------
        # Create Matrix A
        # -------------------------------------------------------------------
        # WHAT: Down-projection matrix (projects input to low-rank space)
        # SHAPE: (r, in_features)
        # BIAS: False (standard LoRA doesn't use bias in A)
        # TRAINABLE: Yes!
        # -------------------------------------------------------------------
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        # EXAMPLE for GPT-2 attention (in_features=768, r=8):
        # self.lora_A["default"] = nn.Linear(768, 8, bias=False)
        # Weight shape: (8, 768) = 6,144 parameters

        # -------------------------------------------------------------------
        # Create Matrix B
        # -------------------------------------------------------------------
        # WHAT: Up-projection matrix (projects back to output space)
        # SHAPE: (out_features, r)
        # BIAS: lora_bias parameter (normally False)
        # TRAINABLE: Yes!
        # -------------------------------------------------------------------
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=lora_bias)
        # EXAMPLE for GPT-2 attention (out_features=2304, r=8):
        # self.lora_B["default"] = nn.Linear(8, 2304, bias=False)
        # Weight shape: (2304, 8) = 18,432 parameters
        #
        # TOTAL TRAINABLE: 6,144 + 18,432 = 24,576 parameters
        # vs BASE: 768 × 2304 = 1,769,472 parameters (98.6% reduction!)

        # -------------------------------------------------------------------
        # Store lora_bias Flag
        # -------------------------------------------------------------------
        self.lora_bias[adapter_name] = lora_bias

        # ===================================================================
        # STEP 7: COMPUTE SCALING FACTOR
        # ===================================================================
        # WHAT: Calculate the scaling multiplier for adapter output
        # HOW: alpha / r (standard) or alpha / √r (rsLoRA)
        # WHY: Controls magnitude of adapter contribution
        # ===================================================================

        if use_rslora:
            # Rank-Stabilized LoRA: scale by alpha / √r
            # Better for low ranks
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
            # EXAMPLE: alpha=16, r=8 → scaling = 16/√8 = 5.66
        else:
            # Standard LoRA: scale by alpha / r
            self.scaling[adapter_name] = lora_alpha / r
            # EXAMPLE: alpha=16, r=8 → scaling = 16/8 = 2.0

        # Store rsLoRA flag
        self.use_rslora[adapter_name] = use_rslora

        # -------------------------------------------------------------------
        # Store DoRA Flag
        # -------------------------------------------------------------------
        self.use_dora[adapter_name] = use_dora
        # If DoRA, additional magnitude vector will be added by variant.init()

        # ===================================================================
        # STEP 8: INITIALIZE WEIGHTS
        # ===================================================================
        # WHAT: Set initial values for A and B matrices
        # HOW: Different methods based on init_lora_weights parameter
        # WHY: Initialization affects convergence and final performance
        # ===================================================================

        # -------------------------------------------------------------------
        # Special Initializations (that modify base weights)
        # -------------------------------------------------------------------
        # These initializations need access to base layer weights
        # Use gather_params_ctx for DeepSpeed compatibility

        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            # ---------------------------------------------------------------
            # PiSSA Initialization
            # ---------------------------------------------------------------
            # WHAT: Principal Singular values and Singular vectors Adaptation
            # HOW: SVD of base weights, modify base to W_residual
            # WHY: Faster convergence than standard init
            # ---------------------------------------------------------------
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
            # RESULT:
            # - W = U S V^T (SVD of original weights)
            # - A = √S V^T, B = U √S
            # - Base weights modified to W_residual = W - B @ A

        elif isinstance(init_lora_weights, str) and init_lora_weights.startswith("corda"):
            # ---------------------------------------------------------------
            # CorDA Initialization
            # ---------------------------------------------------------------
            # WHAT: Context-oriented decomposition
            # HOW: Context-aware matrix factorization
            # WHY: Better than PiSSA for certain tasks
            # ---------------------------------------------------------------
            with gather_params_ctx(self.get_base_layer().weight):
                self.corda_init(adapter_name, init_lora_weights)

        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            # ---------------------------------------------------------------
            # OLoRA Initialization
            # ---------------------------------------------------------------
            # WHAT: Orthogonal LoRA initialization
            # HOW: QR decomposition of base weights
            # WHY: Orthogonal structure, modifies base weights
            # ---------------------------------------------------------------
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)

        elif init_lora_weights == "loftq":
            # ---------------------------------------------------------------
            # LoftQ Initialization
            # ---------------------------------------------------------------
            # WHAT: Quantization-aware LoRA initialization
            # HOW: Alternating quantization and SVD
            # WHY: Better for quantized models (4-bit, 8-bit)
            # ---------------------------------------------------------------
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)

        elif init_lora_weights == "eva":
            # ---------------------------------------------------------------
            # EVA Initialization
            # ---------------------------------------------------------------
            # WHAT: Explained Variance Adaptation
            # HOW: Data-driven rank allocation based on SVD
            # WHY: Automatically distributes ranks based on layer importance
            # NOTE: Only B is zeroed here, A is set during EVA-specific init
            # ---------------------------------------------------------------
            nn.init.zeros_(self.lora_B[adapter_name].weight)

        elif init_lora_weights == "orthogonal":
            # ---------------------------------------------------------------
            # Orthogonal Initialization
            # ---------------------------------------------------------------
            # WHAT: Orthogonal matrix initialization
            # HOW: Orthogonal initialization for A, zeros for B
            # WHY: Better gradient flow
            # ---------------------------------------------------------------
            with gather_params_ctx(self.get_base_layer().weight):
                self.orthogonal_init(adapter_name)

        elif init_lora_weights:
            # ---------------------------------------------------------------
            # Standard Initialization (DEFAULT)
            # ---------------------------------------------------------------
            # WHAT: Standard Kaiming uniform for A, zeros for B
            # HOW: reset_lora_parameters() called
            # WHY: Ensures ΔW = B @ A = 0 initially (no-op)
            # ---------------------------------------------------------------
            self.reset_lora_parameters(adapter_name, init_lora_weights)
            # RESULT:
            # - A.weight: Kaiming uniform ~U(-√(5/r), √(5/r))
            # - B.weight: Zeros
            # - Therefore: B @ A = 0 (adapter starts as identity)

        # ===================================================================
        # STEP 9: MOVE ADAPTER TO CORRECT DEVICE
        # ===================================================================
        # WHAT: Move A and B to same device as base layer
        # WHY: Base might be on GPU, adapters initialized on CPU
        # ===================================================================
        self._move_adapter_to_device_of_base_layer(adapter_name)
        # EXAMPLE:
        # If base_layer.weight is on cuda:0
        # Move lora_A and lora_B to cuda:0

        # ===================================================================
        # STEP 10: INITIALIZE LORA VARIANT (if using DoRA, aLoRA, etc.)
        # ===================================================================
        # WHAT: Initialize variant-specific parameters
        # WHY: Variants need additional setup (e.g., DoRA magnitude vector)
        # ===================================================================
        if adapter_name in self.lora_variant:
            self.lora_variant[adapter_name].init(self, **kwargs)
            # For DoRA: Adds lora_magnitude_vector parameter
            # For aLoRA: Sets up activation mechanisms
            # For Arrow: Initializes routing

        # ===================================================================
        # STEP 11: SET ADAPTER AS ACTIVE (if in training mode)
        # ===================================================================
        # WHAT: Update active_adapters list
        # WHY: Controls which adapters are used in forward pass
        # ===================================================================
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)


# =============================================================================
# SUMMARY: What update_layer() Does
# =============================================================================
#
# INPUT:
#   adapter_name: "default"
#   r: 8
#   lora_alpha: 16
#   in_features: 768 (from base layer)
#   out_features: 2304 (from base layer)
#
# PROCESS:
#   1. Validate parameters
#   2. Resolve LoRA variant (if any)
#   3. Create dropout layer
#   4. *** Create A and B matrices ***
#      - lora_A: nn.Linear(768, 8, bias=False)
#      - lora_B: nn.Linear(8, 2304, bias=False)
#   5. Compute scaling factor
#   6. Initialize weights
#   7. Move to correct device
#   8. Initialize variant (if DoRA/aLoRA/etc.)
#
# OUTPUT (stored in self):
#   self.lora_A = {"default": nn.Linear(768, 8)}      # 6,144 params
#   self.lora_B = {"default": nn.Linear(8, 2304)}     # 18,432 params
#   self.scaling = {"default": 2.0}
#   self.lora_dropout = {"default": Dropout(0.1)}
#   Total trainable: 24,576 parameters
#
# =============================================================================
# KEY INSIGHTS:
# =============================================================================
#
# 1. **ModuleDict Structure**: All adapter params stored in dicts
#    - Enables multi-adapter support
#    - Each adapter has independent A, B, dropout, scaling
#    - Access via: self.lora_A[adapter_name]
#
# 2. **Matrix Dimensions**:
#    - A: (r, in_features) - down-projection
#    - B: (out_features, r) - up-projection
#    - Total params: r × (in_features + out_features)
#    - MUCH smaller than in_features × out_features
#
# 3. **Initialization Ensures No-Op**:
#    - Standard init: B=0, A=Kaiming
#    - Result: B @ A = 0
#    - Model behaves identically before training
#    - No disruption to base model
#
# 4. **Scaling Factor**:
#    - Standard: alpha / r
#    - rsLoRA: alpha / √r
#    - Controls magnitude of adapter contribution
#    - Independent of parameter count
#
# 5. **Variant Support**:
#    - DoRA: Adds magnitude vector
#    - aLoRA: Adds activation mechanism
#    - Arrow: Adds routing
#    - QALoRA: Quantization-aware setup
#
# =============================================================================
# AFTER THIS FUNCTION:
# =============================================================================
#
# The LoRA layer now has:
# - Trainable A and B matrices ready for training
# - Proper initialization (usually ΔW = 0 initially)
# - Correct device placement
# - Scaling factor computed
# - Ready for forward pass!
#
# Next: Forward pass will use these matrices
# See: annotated_code/02_lora_forward_pass_annotated.py
#
# =============================================================================
