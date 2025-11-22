"""
=============================================================================
HEAVILY ANNOTATED: PeftModel.__init__() - The Wrapper Initialization
=============================================================================

File Location: src/peft/peft_model.py:103-144

WHAT: Initializes the PeftModel wrapper around a base model and triggers
      adapter injection.

HOW: Creates tuner instance (e.g., LoraModel) which handles injection,
     or sets up prompt learning if using prompt tuning methods.

WHY: PeftModel is the user-facing wrapper that provides save/load/merge
     functionality while delegating actual adaptation to tuner classes.

CALL CHAIN:
    get_peft_model()
        → PeftModel.__init__() ← YOU ARE HERE
            → LoraModel.__init__() [creates tuner]
                → BaseTuner.__init__()
                    → inject_adapter()

CRITICAL INSIGHT:
    PeftModel is a WRAPPER, not the adapter implementation!
    - Actual adaptation logic is in tuner classes (LoraModel, IA3Model, etc.)
    - PeftModel provides consistent API across all adapter types
    - Handles save/load, adapter management, task-specific forward methods

=============================================================================
"""

from typing import Literal, Optional
import torch
from torch import nn
from transformers import PreTrainedModel


class PeftModel(PushToHubMixin, torch.nn.Module):
    """
    Base model encompassing various PEFT methods.

    This wrapper provides a unified interface for all PEFT methods while
    delegating the actual adaptation to tuner-specific classes.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        adapter_name: str = "default",
        autocast_adapter_dtype: bool = True,
        low_cpu_mem_usage: bool = False,
    ) -> None:
        """
        Initialize PEFT model wrapper.

        =======================================================================
        PARAMETERS EXPLAINED:
        =======================================================================

        model: PreTrainedModel
            The base model to adapt (e.g., GPT-2, BERT, LLaMA)
            Can be any torch.nn.Module, but typically transformers model

        peft_config: PeftConfig
            Configuration object (LoraConfig, IA3Config, etc.)
            Defines adapter type and parameters

        adapter_name: str = "default"
            Name for this adapter
            Allows multiple adapters on same model
            EXAMPLE: "default", "task_A", "task_B"

        autocast_adapter_dtype: bool = True
            Whether to upcast adapter weights to float32
            Helps training stability with mixed precision
            Only affects select adapter types (LoRA, AdaLoRA)

        low_cpu_mem_usage: bool = False
            Create adapter weights on meta device
            Speeds up loading when loading pretrained adapters
            WARNING: Don't use when training from scratch!

        =======================================================================
        """

        # ===================================================================
        # STEP 1: INITIALIZE AS torch.nn.Module
        # ===================================================================
        # WHAT: Call parent constructor
        # WHY: PeftModel is a nn.Module subclass
        # ===================================================================
        super().__init__()

        # ===================================================================
        # STEP 2: STORE ADAPTER METADATA
        # ===================================================================

        # -------------------------------------------------------------------
        # Set Active Adapter
        # -------------------------------------------------------------------
        # WHAT: Track which adapter is currently active
        # WHY: Can switch between adapters at runtime
        # -------------------------------------------------------------------
        self.active_adapter = adapter_name

        # -------------------------------------------------------------------
        # Store PEFT Type
        # -------------------------------------------------------------------
        # WHAT: Store the adapter type (LORA, IA3, etc.)
        # WHY: Different types need different handling
        # -------------------------------------------------------------------
        self.peft_type = peft_config.peft_type
        # EXAMPLE: PeftType.LORA

        # -------------------------------------------------------------------
        # Define Special Forward Arguments
        # -------------------------------------------------------------------
        # WHAT: Arguments that PeftModel handles specially
        # WHY: These args are intercepted before passing to base model
        # -------------------------------------------------------------------
        self.special_peft_forward_args = {"adapter_names", "alora_offsets"}

        # ===================================================================
        # STEP 3: BRANCH: PROMPT LEARNING vs TUNER-BASED
        # ===================================================================
        # WHAT: Different initialization paths for different adapter types
        # WHY: Prompt learning modifies inputs, not weights
        # ===================================================================

        self._is_prompt_learning = peft_config.is_prompt_learning

        if self._is_prompt_learning:
            # ===============================================================
            # PATH A: PROMPT LEARNING (Prefix, Prompt, P-tuning)
            # ===============================================================
            # WHAT: Set up prompt learning adapter
            # HOW: Store config, keep base model as-is, add adapter
            # WHY: Prompt learning adds trainable prompts, not layer adapters
            # ===============================================================

            self._peft_config = {adapter_name: peft_config}
            self.base_model = model
            self.add_adapter(adapter_name, peft_config, low_cpu_mem_usage=low_cpu_mem_usage)
            # For prompt learning:
            # - No layer replacement
            # - Trainable prompt embeddings added
            # - Forward pass prepends prompts to input

        else:
            # ===============================================================
            # PATH B: TUNER-BASED ADAPTERS (LoRA, IA3, AdaLoRA, etc.)
            # ===============================================================
            # WHAT: Create tuner instance that injects adapters
            # HOW: Look up tuner class, instantiate with base model
            # WHY: Tuner handles layer replacement/modification
            # ===============================================================

            self._peft_config = None  # Managed by tuner

            # ---------------------------------------------------------------
            # STEP 3B-1: Get Tuner Class
            # ---------------------------------------------------------------
            # WHAT: Look up appropriate tuner class for this PEFT type
            # HOW: PEFT_TYPE_TO_TUNER_MAPPING[PeftType.LORA] → LoraModel
            # WHY: Each adapter type has custom tuner implementation
            # ---------------------------------------------------------------
            cls = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type]
            # EXAMPLE:
            # peft_config.peft_type = PeftType.LORA
            # cls = LoraModel

            # ---------------------------------------------------------------
            # STEP 3B-2: Prepare Context Manager
            # ---------------------------------------------------------------
            # WHAT: Use init_empty_weights if low_cpu_mem_usage=True
            # WHY: Creates parameters on meta device for fast loading
            # ---------------------------------------------------------------
            ctx = init_empty_weights if low_cpu_mem_usage else nullcontext

            # ===============================================================
            # *** CREATE TUNER INSTANCE ***
            # ===============================================================
            # WHAT: Instantiate tuner class (e.g., LoraModel)
            # HOW: Pass base model, config dict, adapter name
            # RESULT: Tuner's __init__ triggers inject_adapter()
            # ===============================================================
            with ctx():
                self.base_model = cls(
                    model,
                    {adapter_name: peft_config},
                    adapter_name
                )
            # AFTER THIS LINE:
            # - self.base_model is now a LoraModel instance
            # - LoraModel wraps the original model
            # - inject_adapter() has been called
            # - Target layers have been replaced with LoRA layers
            #
            # EXAMPLE:
            # Before: model.transformer.h[0].attn.c_attn = Conv1D(...)
            # After:  model.transformer.h[0].attn.c_attn = LoraLinear(...)

        # ===================================================================
        # STEP 4: ADAPTER DTYPE CASTING (Optional)
        # ===================================================================
        # WHAT: Upcast adapter weights to float32 if requested
        # WHY: Training stability with mixed precision
        # WHEN: Only if tuner supports _cast_adapter_dtype method
        # ===================================================================
        if hasattr(self.base_model, "_cast_adapter_dtype"):
            self.base_model._cast_adapter_dtype(
                adapter_name=adapter_name,
                autocast_adapter_dtype=autocast_adapter_dtype
            )
            # EXAMPLE:
            # If base model is bfloat16, adapter might be float32
            # This prevents precision issues during training

        # ===================================================================
        # STEP 5: GRADIENT CHECKPOINTING SETUP (Optional)
        # ===================================================================
        # WHAT: Prepare model for gradient checkpointing if enabled
        # WHY: Gradient checkpointing saves memory during training
        # WHEN: If base model has gradient checkpointing enabled
        # ===================================================================
        if getattr(model, "is_gradient_checkpointing", True):
            model = self.prepare_model_for_gradient_checkpointing(model)

        # ===================================================================
        # STEP 6: DISABLE TENSOR PARALLELISM (Safety)
        # ===================================================================
        # WHAT: Set pretraining_tp to 1
        # WHY: Avoid numerical differences from tensor parallelism
        # WHEN: If base model has this config attribute
        # ===================================================================
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
            self.base_model.config.pretraining_tp = 1
            # NOTE: Some models use tensor parallelism during inference
            # This can cause numerical differences vs non-TP inference
            # PEFT disables this for consistency

        # ===================================================================
        # STEP 7: INITIALIZE ADAPTER STATE
        # ===================================================================
        # WHAT: Track whether adapters are disabled
        # WHY: Allows temporarily disabling adapters without unloading
        # ===================================================================
        self._adapters_disabled = False


# =============================================================================
# SUMMARY: What PeftModel.__init__() Does
# =============================================================================
#
# INPUT:
#   model: GPT-2 base model
#   peft_config: LoraConfig(r=8, target_modules=["c_attn", "c_proj"])
#   adapter_name: "default"
#
# PROCESS:
#   1. Initialize as nn.Module
#   2. Store adapter metadata
#   3. Branch based on adapter type:
#      - Prompt learning: Add prompt embeddings
#      - Tuner-based: Create tuner instance
#   4. Tuner creation triggers:
#      - LoraModel.__init__()
#      - BaseTuner.__init__()
#      - inject_adapter()
#      - Layer replacement
#   5. Optional: Cast adapter dtype
#   6. Optional: Setup gradient checkpointing
#   7. Disable tensor parallelism
#
# OUTPUT:
#   PeftModel instance where:
#   - self.base_model is LoraModel (for LoRA)
#   - LoraModel wraps original model
#   - Target layers replaced with LoRA layers
#   - Ready for training or inference
#
# =============================================================================
# KEY INSIGHTS:
# =============================================================================
#
# 1. **Wrapper Pattern**: PeftModel wraps the tuner, tuner wraps base model
#    - PeftModel: User-facing API (save/load/merge)
#    - Tuner (LoraModel): Adapter implementation
#    - Base model: Original model (modified in-place)
#
# 2. **Two Initialization Paths**:
#    - Prompt learning: Add prompt embeddings, no layer changes
#    - Tuner-based: Replace layers with adapter-enhanced versions
#
# 3. **Injection Trigger**: Creating tuner instance triggers injection
#    - LoraModel.__init__() calls BaseTuner.__init__()
#    - BaseTuner.__init__() calls inject_adapter()
#    - inject_adapter() performs actual layer replacement
#
# 4. **Low CPU Memory Mode**: For fast checkpoint loading
#    - Creates parameters on meta device
#    - Filled later when loading weights
#    - Don't use for training from scratch!
#
# 5. **Adapter Management**:
#    - active_adapter tracks current adapter
#    - Can switch adapters without reloading
#    - Multiple adapters supported
#
# =============================================================================
# STRUCTURE AFTER INITIALIZATION:
# =============================================================================
#
# PeftModel
#   ├── base_model: LoraModel (tuner)
#   │   ├── model: GPT2LMHeadModel (original, modified)
#   │   │   └── transformer.h[0].attn.c_attn: LoraLinear
#   │   │       ├── base_layer: Conv1D (original, frozen)
#   │   │       ├── lora_A: {"default": nn.Linear(768, 8)}
#   │   │       ├── lora_B: {"default": nn.Linear(8, 2304)}
#   │   │       ├── scaling: {"default": 2.0}
#   │   │       └── ...
#   │   ├── peft_config: {"default": LoraConfig(...)}
#   │   ├── active_adapter: "default"
#   │   └── ...
#   ├── active_adapter: "default"
#   ├── peft_type: PeftType.LORA
#   └── ...
#
# =============================================================================
# NEXT STEPS:
# =============================================================================
#
# After __init__(), the model is ready for:
# - Training: optimizer.step() updates only adapter params
# - Inference: forward() uses adapters
# - Saving: save_pretrained() saves only adapters
# - Loading more adapters: load_adapter()
# - Switching adapters: set_adapter()
#
# To understand the injection process:
# See: annotated_code/04_inject_adapter_annotated.py
#
# To understand save/load:
# See: annotated_code/08_from_pretrained_annotated.py
# See: guides/02_adapter_saving_deep_dive.md
#
# =============================================================================
