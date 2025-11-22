"""
=============================================================================
HEAVILY ANNOTATED: LoraModel._create_and_replace() - Layer Replacement
=============================================================================

File Location: src/peft/tuners/lora/model.py:159-282

WHAT: This function CREATES a new LoRA-enhanced layer and REPLACES the
      original layer in the parent module using setattr().

HOW: Three main steps:
     1. Determine layer-specific configuration (rank, alpha)
     2. Create new LoRA layer (or update existing one)
     3. Replace old layer with new layer using setattr()

WHY: This is WHERE the actual layer replacement happens!
     nn.Linear → LoraLinear transformation occurs here.

CALL CHAIN:
    inject_adapter()
        → _create_and_replace() ← YOU ARE HERE
            → _create_new_module()  [creates LoraLinear]
            → _replace_module()     [performs setattr()]

CRITICAL INSIGHT:
    This function uses Python's setattr() to replace layers:
    setattr(parent, "c_attn", new_lora_linear)

    This MODIFIES the model in-place!

=============================================================================
"""

from typing import Optional
import torch
from torch import nn


class LoraModel(BaseTuner):
    """LoRA-specific tuner that handles LoRA layer injection"""

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        *,
        parameter_name: Optional[str] = None,
    ) -> None:
        """
        Create and replace the target module with a LoRA-enhanced version.

        =======================================================================
        PARAMETERS EXPLAINED:
        =======================================================================

        lora_config: LoraConfig
            Configuration for this adapter (r, alpha, dropout, etc.)

        adapter_name: str
            Name of the adapter being added (e.g., "default", "task_A")

        target: nn.Module
            The actual module to be replaced (e.g., nn.Linear object)

        target_name: str
            The attribute name of target in parent (e.g., "c_attn")

        parent: nn.Module
            The parent module containing target (e.g., GPT2Attention)

        current_key: str
            Full path to the module (e.g., "transformer.h.0.attn.c_attn")

        parameter_name: str, optional
            If targeting nn.Parameter, this is the parameter name

        =======================================================================
        EXAMPLE CALL:
        =======================================================================

        _create_and_replace(
            lora_config=LoraConfig(r=8, lora_alpha=16, ...),
            adapter_name="default",
            target=<Conv1D object at 0x...>,  # The actual layer
            target_name="c_attn",              # Its name in parent
            parent=<GPT2Attention object>,     # Its parent module
            current_key="transformer.h.0.attn.c_attn",  # Full path
        )

        =======================================================================
        """

        # ===================================================================
        # STEP 1: VALIDATION AND SPECIAL CASES
        # ===================================================================

        # -------------------------------------------------------------------
        # Validate Current Key
        # -------------------------------------------------------------------
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # -------------------------------------------------------------------
        # Check target_parameters Compatibility
        # -------------------------------------------------------------------
        # WHAT: Ensure only one adapter uses target_parameters at a time
        # WHY: target_parameters uses different injection mechanism
        # -------------------------------------------------------------------
        if lora_config.target_parameters:
            other_configs_use_target_params = any(
                conf.target_parameters
                for key, conf in self.peft_config.items()
                if key != adapter_name
            )
            if other_configs_use_target_params:
                raise ValueError(
                    f"Adding a LoRA config with `target_parameters={lora_config.target_parameters}` but there are "
                    "already other LoRA adapters on this model that use `target_parameters`. At the moment, only "
                    "one LoRA adapter per model with `target_parameters` is allowed."
                )

        # ===================================================================
        # STEP 2: DETERMINE LAYER-SPECIFIC CONFIGURATION
        # ===================================================================
        # WHAT: Get rank and alpha for THIS SPECIFIC LAYER
        # HOW: Check rank_pattern and alpha_pattern for layer-specific values
        # WHY: Allows heterogeneous adapter ranks across layers
        # ===================================================================

        # -------------------------------------------------------------------
        # Get Layer-Specific Rank
        # -------------------------------------------------------------------
        # WHAT: Check if this layer has custom rank in rank_pattern
        # HOW: get_pattern_key() does regex matching on current_key
        # DEFAULT: Use config.r if no pattern matches
        # -------------------------------------------------------------------
        r_key = get_pattern_key(lora_config.rank_pattern.keys(), current_key)
        r = lora_config.rank_pattern.get(r_key, lora_config.r)
        # EXAMPLE:
        # rank_pattern = {"^model.layers.0": 16, "^model.layers.1": 8}
        # current_key = "model.layers.0.attn.q_proj"
        # → r = 16 (instead of default 8)

        # -------------------------------------------------------------------
        # Get Layer-Specific Alpha
        # -------------------------------------------------------------------
        # WHAT: Check if this layer has custom alpha in alpha_pattern
        # DEFAULT: Use config.lora_alpha if no pattern matches
        # -------------------------------------------------------------------
        alpha_key = get_pattern_key(lora_config.alpha_pattern.keys(), current_key)
        alpha = lora_config.alpha_pattern.get(alpha_key, lora_config.lora_alpha)
        # EXAMPLE:
        # alpha_pattern = {"^model.layers.0": 32}
        # current_key = "model.layers.0.attn.q_proj"
        # → alpha = 32 (instead of default 16)

        # ===================================================================
        # STEP 3: BUILD KWARGS FOR NEW MODULE
        # ===================================================================
        # WHAT: Collect all parameters needed to create LoRA layer
        # WHY: Different layer types need different parameters
        # ===================================================================

        kwargs = {
            # Core LoRA parameters
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,

            # Advanced features
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "use_alora": lora_config.alora_invocation_tokens is not None,
            "use_qalora": lora_config.use_qalora,
            "qalora_group_size": lora_config.qalora_group_size,
            "ephemeral_gpu_offload": lora_config.runtime_config.ephemeral_gpu_offload,
            "lora_bias": lora_config.lora_bias,
            "arrow_config": lora_config.arrow_config,

            # Model state
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),

            # For nn.Parameter targeting
            "parameter_name": parameter_name,
        }

        # -------------------------------------------------------------------
        # Add Quantization Config (for TorchAO merging)
        # -------------------------------------------------------------------
        # WHAT: Extract quantization config from model if available
        # WHY: TorchAO needs get_apply_tensor_subclass for merging
        # -------------------------------------------------------------------
        try:
            kwargs["get_apply_tensor_subclass"] = operator.attrgetter(
                "hf_quantizer.quantization_config.get_apply_tensor_subclass"
            )(self.model)
        except AttributeError:
            pass  # Not a quantized model

        # -------------------------------------------------------------------
        # Add Quantization Configs (GPTQ, AQLM, AWQ)
        # -------------------------------------------------------------------
        # WHAT: Check for various quantization methods
        # WHY: Different quantization methods need special handling
        # -------------------------------------------------------------------
        quant_methods = ["gptq", "aqlm", "awq"]
        for quant_method in quant_methods:
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                kwargs[f"{quant_method}_quantization_config"] = quantization_config

        # ===================================================================
        # STEP 4: DECIDE: UPDATE EXISTING OR CREATE NEW?
        # ===================================================================
        # WHAT: Check if target already has LoRA adapter
        # WHY: Multi-adapter support - can add multiple adapters to same layer
        # ===================================================================

        from peft.tuners.adalora import AdaLoraLayer

        # -------------------------------------------------------------------
        # Check if Target is Already a LoRA Layer
        # -------------------------------------------------------------------
        # WHAT: Determine if we're adding to existing LoRA or creating new
        # WHY: Different code paths for first adapter vs. additional adapters
        # -------------------------------------------------------------------
        wrap_target_param = isinstance(target, ParamWrapper) and (adapter_name in target.lora_A)

        if isinstance(target, LoraLayer) and not isinstance(target, AdaLoraLayer) and not wrap_target_param:
            # ===============================================================
            # PATH A: LAYER ALREADY HAS LORA - UPDATE IT
            # ===============================================================
            # WHAT: Add new adapter to existing LoRA layer
            # HOW: Call update_layer() method
            # WHY: Reuse existing LoRA layer structure, just add new A/B
            # ===============================================================

            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
                lora_bias=lora_config.lora_bias,
                arrow_config=lora_config.arrow_config,
                inference_mode=lora_config.inference_mode,
            )
            # AFTER THIS:
            # target.lora_A now has both "default" and "task_A" adapters
            # target.lora_A = {
            #     "default": nn.Linear(768, 8),
            #     "task_A": nn.Linear(768, 16),  # NEW!
            # }

        else:
            # ===============================================================
            # PATH B: FIRST LORA ADAPTER - CREATE NEW LAYER
            # ===============================================================
            # WHAT: Create brand new LoRA-enhanced layer
            # HOW: Call _create_new_module() to get appropriate layer type
            # WHY: Original layer needs to be wrapped with LoRA
            # ===============================================================

            # -----------------------------------------------------------
            # Validate ParamWrapper Case
            # -----------------------------------------------------------
            if isinstance(target, ParamWrapper) and (parameter_name == target.parameter_name):
                raise ValueError(
                    "Trying to target the same nn.Parameter twice, this should not happen. Please open an issue on "
                    "the PEFT repo: https://github.com/huggingface/peft/issues"
                )

            # -----------------------------------------------------------
            # Get Device Map
            # -----------------------------------------------------------
            # WHAT: Extract device map from model if it exists
            # WHY: Need to place adapter on correct device
            # -----------------------------------------------------------
            device_map = self.model.hf_device_map if hasattr(self.model, "hf_device_map") else None

            # ===============================================================
            # *** CREATE NEW LORA MODULE ***
            # ===============================================================
            # WHAT: Create LoRA-enhanced version of the target layer
            # HOW: _create_new_module() dispatches to correct layer type
            # RETURNS: New module (e.g., LoraLinear wrapping nn.Linear)
            # ===============================================================
            new_module = self._create_new_module(
                lora_config,
                adapter_name,
                target,
                device_map=device_map,
                **kwargs
            )
            # EXAMPLE:
            # Input: target = nn.Linear(768, 2304)
            # Output: new_module = LoraLinear(
            #             base_layer=nn.Linear(768, 2304),
            #             lora_A={"default": nn.Linear(768, 8)},
            #             lora_B={"default": nn.Linear(8, 2304)},
            #             ...
            #         )

            # -----------------------------------------------------------
            # Set Trainability
            # -----------------------------------------------------------
            # WHAT: If this adapter is not active, freeze it
            # WHY: Only active adapters should be trainable
            # -----------------------------------------------------------
            if adapter_name not in self.active_adapters:
                # Adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)

            # ===============================================================
            # *** REPLACE OLD MODULE WITH NEW MODULE ***
            # ===============================================================
            # WHAT: Replace target in parent with new_module
            # HOW: Call _replace_module() which uses setattr()
            # RESULT: parent.target_name is now new_module, not target
            # ===============================================================
            self._replace_module(parent, target_name, new_module, target)
            # AFTER THIS CALL:
            # parent.c_attn is now the new LoRA-enhanced layer
            # The original Conv1D is wrapped inside new_module.base_layer


    def _replace_module(self, parent, child_name, new_module, child):
        """
        Replace a child module in parent with new_module.

        =======================================================================
        THIS IS WHERE THE ACTUAL REPLACEMENT HAPPENS!
        =======================================================================

        WHAT: Uses setattr() to replace child module
        HOW: setattr(parent, child_name, new_module)
        WHY: Modifies model in-place without copying

        EXAMPLE:
            parent = GPT2Attention object
            child_name = "c_attn"
            new_module = LoraLinear object
            child = Conv1D object (old layer)

            setattr(parent, "c_attn", new_module)

            BEFORE: parent.c_attn → Conv1D
            AFTER:  parent.c_attn → LoraLinear (wrapping Conv1D)

        =======================================================================
        """

        # ===================================================================
        # STEP 1: REPLACE MODULE USING SETATTR
        # ===================================================================
        # WHAT: Replace child with new_module in parent
        # HOW: Python's setattr() modifies parent's attribute
        # WHY: In-place modification - no copying needed
        # ===================================================================
        setattr(parent, child_name, new_module)
        # THIS IS THE CRITICAL LINE!
        # The model is now modified - parent.child_name is new_module

        # Note: It's not necessary to set requires_grad here, as that is
        # handled by _mark_only_adapters_as_trainable

        # ===================================================================
        # STEP 2: UNWRAP CHILD LAYER
        # ===================================================================
        # WHAT: If child is already a LoRA layer, get its base layer
        # WHY: Need to access actual nn.Linear for device placement
        # ===================================================================
        if hasattr(child, "base_layer"):
            child = child.base_layer

        # ===================================================================
        # STEP 3: MOVE ADAPTER PARAMETERS TO CORRECT DEVICE
        # ===================================================================
        # WHAT: Ensure adapter params are on same device as base weights
        # HOW: Iterate through new_module, move adapter params to device
        # WHY: Base weights might be on GPU, adapters initialized on CPU
        # ===================================================================

        meta = torch.device("meta")

        for name, module in new_module.named_modules():
            # Check if this is an adapter-related module
            if (self.prefix in name) or ("ranknum" in name):
                # Determine which device to use based on child layer's weights

                # Try different weight attributes (different quantization methods)
                if hasattr(child, "qweight"):
                    # GPTQ quantized
                    weight = child.qweight
                elif hasattr(child, "W_q"):
                    # AQLM quantized
                    weight = child.W_q
                elif hasattr(child, "weight"):
                    # Standard nn.Linear
                    weight = child.weight
                elif getattr(child, "in_proj_weight", None) is not None:
                    # MultiheadAttention
                    weight = child.in_proj_weight
                else:
                    # Fallback: use first parameter
                    weight = next(child.parameters())

                # Only move if not on meta device
                if not any(p.device == meta for p in module.parameters()):
                    # Move adapter module to same device as base weight
                    module.to(weight.device)
                    # EXAMPLE:
                    # If base weight is on cuda:0, move lora_A and lora_B to cuda:0


# =============================================================================
# SUMMARY: What _create_and_replace() Does
# =============================================================================
#
# INPUT:
#   target: nn.Linear(768, 2304)  # Original layer
#   target_name: "c_attn"         # Name in parent
#   parent: GPT2Attention          # Parent module
#
# PROCESS:
#   1. Determine layer-specific rank and alpha
#   2. Build kwargs for LoRA layer creation
#   3. Create new LoRA layer wrapping original:
#      new_module = LoraLinear(base_layer=target, ...)
#   4. Replace using setattr:
#      setattr(parent, "c_attn", new_module)
#   5. Move adapter params to correct device
#
# OUTPUT:
#   parent.c_attn is now LoraLinear, not nn.Linear
#   Original nn.Linear is wrapped inside new_module.base_layer
#
# =============================================================================
# KEY INSIGHTS:
# =============================================================================
#
# 1. **In-Place Replacement**: Uses setattr() to modify parent
#    - No copying of model
#    - Original layer wrapped, not replaced
#    - Base weights preserved and shared
#
# 2. **Layer-Specific Configuration**: Supports heterogeneous ranks
#    - rank_pattern allows different r per layer
#    - alpha_pattern allows different alpha per layer
#    - Enables sophisticated adapter architectures
#
# 3. **Multi-Adapter Support**: Can add multiple adapters
#    - If target is already LoRA, call update_layer()
#    - Adds new A/B matrices to existing LoRA layer
#    - Each adapter has independent parameters
#
# 4. **Quantization Awareness**: Handles various quantization methods
#    - GPTQ, AQLM, AWQ, bitsandbytes
#    - Different layer types for different quantization
#    - Dispatched by _create_new_module()
#
# 5. **Device Management**: Ensures adapter on correct device
#    - Base weights might be on GPU
#    - Adapters initialized on CPU (or meta device)
#    - Moved to match base weight device
#
# =============================================================================
# NEXT STEPS:
# =============================================================================
#
# To understand how new_module is created:
# See: annotated_code/11_dispatcher_annotated.py (_create_new_module)
#
# To understand how A/B matrices are initialized:
# See: annotated_code/06_update_layer_annotated.py
#
# To understand the full injection pipeline:
# See: guides/01_model_injection_deep_dive.md
#
# =============================================================================
