"""
=============================================================================
HEAVILY ANNOTATED: PeftModel.from_pretrained() - Loading Saved Adapters
=============================================================================

File Location: src/peft/peft_model.py:388-700+

WHAT: Class method that loads a saved adapter into a base model, creating
      a PeftModel instance ready for inference or further training.

HOW: Five-stage process:
     1. Load adapter configuration from disk/Hub
     2. Create PeftModel with empty adapters
     3. Load adapter weights from checkpoint
     4. Load weights into model
     5. Prepare for inference/training

WHY: This is the REVERSE of save_pretrained() - reconstructs the PEFT model
     from saved adapter checkpoint.

CALL CHAIN (typical usage):
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    peft_model = PeftModel.from_pretrained(base_model, "./my_adapter")
        ← YOU ARE HERE

CRITICAL INSIGHT:
    This method:
    1. Does NOT load the base model (user provides it)
    2. Only loads adapter weights (~3 MB, not 500 MB!)
    3. Injects adapters into provided base model
    4. Returns PeftModel wrapper ready to use

=============================================================================
"""

from typing import Any, Optional, Union
import os
import warnings
import torch
from transformers import PreTrainedModel


class PeftModel:
    """PEFT model wrapper - provides save/load/merge functionality"""

    @classmethod
    def from_pretrained(
        cls,
        model: torch.nn.Module,
        model_id: Union[str, os.PathLike],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        autocast_adapter_dtype: bool = True,
        ephemeral_gpu_offload: bool = False,
        low_cpu_mem_usage: bool = False,
        key_mapping: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> "PeftModel":
        """
        Load a saved PEFT adapter into a base model.

        =======================================================================
        PARAMETERS EXPLAINED:
        =======================================================================

        model: torch.nn.Module
            ** The base model (already loaded) **
            EXAMPLE: model = AutoModelForCausalLM.from_pretrained("gpt2")
            NOTE: User loads this separately, not by this method!

        model_id: str or Path
            ** Where to load adapter from **
            Can be:
            - HuggingFace Hub: "username/adapter-name"
            - Local path: "./my_adapter/" or "path/to/adapter"
            Will look for:
            - adapter_config.json
            - adapter_model.safetensors (or .bin)

        adapter_name: str = "default"
            ** Name to assign loaded adapter **
            Allows loading multiple adapters into same model
            EXAMPLE: "task_A", "task_B", "default"

        is_trainable: bool = False
            ** Whether adapter should be trainable **
            False: Inference only (faster, less memory)
            True: Continue training

        config: PeftConfig = None
            ** Optionally provide config instead of loading **
            If None, loads from model_id/adapter_config.json
            If provided, doesn't load config from disk

        autocast_adapter_dtype: bool = True
            ** Upcast adapters to float32 **
            Helps training stability

        low_cpu_mem_usage: bool = False
            ** Create empty adapters on meta device first **
            Speeds up loading process
            Recommended for loading large adapters

        =======================================================================
        EXAMPLE USAGE:
        =======================================================================

        # 1. Load base model
        from transformers import AutoModelForCausalLM
        base_model = AutoModelForCausalLM.from_pretrained("gpt2")

        # 2. Load adapter into base model
        from peft import PeftModel
        peft_model = PeftModel.from_pretrained(
            base_model,
            "./my_adapter",  # or "username/adapter-name"
            adapter_name="default"
        )

        # 3. Ready for inference!
        outputs = peft_model.generate(...)

        =======================================================================
        """

        # ===================================================================
        # STAGE 1: LOAD ADAPTER CONFIGURATION
        # ===================================================================
        # WHAT: Load adapter_config.json from disk or HuggingFace Hub
        # HOW: PeftConfig.from_pretrained() handles downloading/reading
        # WHY: Need config to know adapter type, rank, target modules, etc.
        # ===================================================================

        if config is None:
            # ---------------------------------------------------------------
            # STEP 1.1: Prepare HuggingFace Hub Arguments
            # ---------------------------------------------------------------
            # WHAT: Extract args for downloading from Hub
            # WHY: Pass to from_pretrained() for auth, caching, etc.
            # ---------------------------------------------------------------
            hf_kwargs = {
                "subfolder": kwargs.get("subfolder", None),
                "revision": kwargs.get("revision", None),
                "cache_dir": kwargs.get("cache_dir", None),
                "token": kwargs.get("token", None),
            }

            # ---------------------------------------------------------------
            # STEP 1.2: Determine PEFT Type and Load Config
            # ---------------------------------------------------------------
            # WHAT: Detect adapter type from config, then load full config
            # HOW: _get_peft_type() reads adapter_config.json "peft_type"
            # RESULT: config is now LoraConfig, IA3Config, etc.
            # ---------------------------------------------------------------
            peft_type = PeftConfig._get_peft_type(model_id, **hf_kwargs)
            # EXAMPLE: peft_type = "LORA"

            config = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type].from_pretrained(
                model_id, **kwargs
            )
            # EXAMPLE: config = LoraConfig(
            #     r=8,
            #     lora_alpha=16,
            #     target_modules=["c_attn", "c_proj"],
            #     ...
            # )

        elif isinstance(config, PeftConfig):
            # ---------------------------------------------------------------
            # Config Provided by User
            # ---------------------------------------------------------------
            # WHAT: User passed config manually
            # HOW: Just set inference mode based on is_trainable
            # ---------------------------------------------------------------
            config.inference_mode = not is_trainable

        else:
            raise ValueError(f"The input config must be a PeftConfig, got {config.__class__}")

        # -------------------------------------------------------------------
        # STEP 1.3: Handle Checkpoint Conversion Mapping
        # -------------------------------------------------------------------
        # WHAT: Some transformers models need state_dict key remapping
        # WHY: Model architecture updates may change parameter names
        # WHEN: If model has _checkpoint_conversion_mapping attribute
        # -------------------------------------------------------------------
        if (key_mapping is None) and (not config.is_prompt_learning):
            key_mapping = getattr(model, "_checkpoint_conversion_mapping", {})

        # -------------------------------------------------------------------
        # STEP 1.4: Set Runtime Configuration
        # -------------------------------------------------------------------
        # WHAT: Configure runtime-only settings (not saved in checkpoint)
        # EXAMPLE: ephemeral_gpu_offload for CPU/GPU hybrid inference
        # -------------------------------------------------------------------
        if hasattr(config, "runtime_config"):
            config.runtime_config.ephemeral_gpu_offload = ephemeral_gpu_offload
        else:
            if ephemeral_gpu_offload:
                warnings.warn("Ephemeral GPU offloading is not supported for this model. Ignoring.")

        # ===================================================================
        # STAGE 2: HANDLE DEVICE MAPPING (Advanced)
        # ===================================================================
        # WHAT: Handle disk-offloaded modules (very large models)
        # WHY: Some models too large for GPU/CPU, offload to disk
        # WHEN: If model has hf_device_map attribute
        # NOTE: Advanced feature, skip for typical usage
        # ===================================================================

        # [Device mapping code omitted for clarity - handles disk offload]
        # See source for full implementation

        # ===================================================================
        # STAGE 3: CREATE PEFT MODEL WITH EMPTY ADAPTERS
        # ===================================================================
        # WHAT: Create PeftModel instance with adapter layers (no weights yet)
        # HOW: Call PeftModel.__init__() which triggers injection
        # RESULT: Model has adapter layers, but weights are random/zero
        # ===================================================================

        # -------------------------------------------------------------------
        # STEP 3.1: Instantiate PeftModel
        # -------------------------------------------------------------------
        # WHAT: Create PeftModel wrapper
        # HOW: Calls __init__() which creates tuner and injects adapters
        # RESULT: Adapter layers created with random initialization
        # -------------------------------------------------------------------
        peft_model = cls(
            model,
            config,
            adapter_name,
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        # AFTER THIS:
        # - Adapter layers injected into model
        # - lora_A and lora_B exist but have wrong weights
        # - Need to load saved weights next!

        # ===================================================================
        # STAGE 4: LOAD ADAPTER WEIGHTS FROM CHECKPOINT
        # ===================================================================
        # WHAT: Load saved adapter weights into the adapter layers
        # HOW: load_peft_weights() reads safetensors/bin file
        # WHY: Replace random init with trained weights!
        # ===================================================================

        # -------------------------------------------------------------------
        # STEP 4.1: Load Weights from Disk/Hub
        # -------------------------------------------------------------------
        # WHAT: Read adapter_model.safetensors (or .bin) from checkpoint
        # HOW: load_peft_weights() handles downloading and loading
        # RESULT: Dictionary of {parameter_name: tensor}
        # -------------------------------------------------------------------
        adapters_weights = load_peft_weights(
            model_id,
            device=infer_device(),  # Auto-detect device
            **kwargs
        )
        # EXAMPLE OUTPUT:
        # {
        #     "transformer.h.0.attn.c_attn.lora_A.weight": Tensor([8, 768]),
        #     "transformer.h.0.attn.c_attn.lora_B.weight": Tensor([2304, 8]),
        #     "transformer.h.0.attn.c_proj.lora_A.weight": Tensor([8, 768]),
        #     ...
        # }

        # -------------------------------------------------------------------
        # STEP 4.2: Handle Key Mapping (if needed)
        # -------------------------------------------------------------------
        # WHAT: Remap state_dict keys if checkpoint format changed
        # WHY: Model architecture updates may change parameter names
        # -------------------------------------------------------------------
        if key_mapping:
            adapters_weights = _apply_key_mapping(adapters_weights, key_mapping)

        # -------------------------------------------------------------------
        # STEP 4.3: Load Weights into Model
        # -------------------------------------------------------------------
        # WHAT: Copy loaded weights into adapter parameters
        # HOW: set_peft_model_state_dict() matches keys and copies tensors
        # RESULT: Adapter layers now have correct trained weights!
        # -------------------------------------------------------------------
        load_result = set_peft_model_state_dict(
            peft_model,
            adapters_weights,
            adapter_name=adapter_name,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        # AFTER THIS:
        # - peft_model.base_model.model.transformer.h[0].attn.c_attn.lora_A["default"]
        #   now has the trained weights from the checkpoint!

        # -------------------------------------------------------------------
        # STEP 4.4: Check for Missing/Unexpected Keys
        # -------------------------------------------------------------------
        # WHAT: Warn if some weights missing or extra weights in checkpoint
        # WHY: Helps debug loading issues
        # -------------------------------------------------------------------
        if (missing_keys := load_result.missing_keys) or (unexpected_keys := load_result.unexpected_keys):
            # Warnings about incompletely loaded adapter
            # [Warning code omitted for clarity]
            pass

        # ===================================================================
        # STAGE 5: FINAL SETUP
        # ===================================================================
        # WHAT: Prepare model for use (inference or training)
        # HOW: Set trainability, prepare tokenizer if needed
        # WHY: Ensure model is in correct state
        # ===================================================================

        # -------------------------------------------------------------------
        # STEP 5.1: Set Training Mode
        # -------------------------------------------------------------------
        # WHAT: Freeze or unfreeze adapter based on is_trainable
        # HOW: Call eval() or train() on model
        # -------------------------------------------------------------------
        if is_trainable:
            peft_model.train()
        else:
            peft_model.eval()

        # -------------------------------------------------------------------
        # STEP 5.2: Handle Tokenizer (Optional)
        # -------------------------------------------------------------------
        # WHAT: Resize tokenizer if vocab was extended
        # WHY: Some adapters train additional tokens
        # WHEN: If save_embedding_layers was used during training
        # -------------------------------------------------------------------
        # [Tokenizer resizing code - handles vocabulary expansion]

        # -------------------------------------------------------------------
        # STEP 5.3: Return Loaded Model
        # -------------------------------------------------------------------
        return peft_model
        # RETURN VALUE: PeftModel with loaded adapter, ready to use!


# =============================================================================
# SUMMARY: What from_pretrained() Does
# =============================================================================
#
# INPUT:
#   model: GPT-2 base model (already loaded by user)
#   model_id: "./my_adapter" (path to saved adapter)
#   adapter_name: "default"
#
# PROCESS:
#   Stage 1: Load Configuration
#     - Read adapter_config.json
#     - config = LoraConfig(r=8, lora_alpha=16, ...)
#
#   Stage 2: Handle Device Mapping (if needed)
#     - For very large models with disk offloading
#
#   Stage 3: Create PeftModel with Empty Adapters
#     - Call PeftModel.__init__()
#     - Triggers inject_adapter()
#     - Creates LoRA layers with random weights
#
#   Stage 4: Load and Set Adapter Weights
#     - Read adapter_model.safetensors (~3 MB)
#     - Load weights into adapter parameters
#     - Adapter now has trained weights!
#
#   Stage 5: Final Setup
#     - Set train/eval mode
#     - Handle tokenizer if needed
#     - Return ready-to-use model
#
# OUTPUT:
#   PeftModel instance with:
#   - Base model: GPT-2 (frozen, 124M params)
#   - Adapter: LoRA (trainable if is_trainable=True, ~1.2M params)
#   - Ready for inference or further training!
#
# =============================================================================
# KEY INSIGHTS:
# =============================================================================
#
# 1. **Base Model NOT Loaded Here**: User loads base model separately
#    - Advantage: Can use different quantization, devices, etc.
#    - Pattern: Load base once, swap adapters easily
#
# 2. **Two-Stage Loading**:
#    - First: Create structure (inject empty adapters)
#    - Second: Fill in weights (load from checkpoint)
#    - Why: Separation allows low_cpu_mem_usage optimization
#
# 3. **Tiny Checkpoint Size**:
#    - Only adapter weights loaded (~3 MB for GPT-2)
#    - Not full model (500 MB for GPT-2)
#    - Enables fast loading and easy sharing
#
# 4. **In-Place Modification**: Base model modified directly
#    - Same model object, now with adapters
#    - No copying of base weights
#    - Memory efficient
#
# 5. **Multi-Adapter Support**: Can load multiple adapters
#    - Load first: from_pretrained(model, "adapter_A", "task_A")
#    - Load second: model.load_adapter("adapter_B", "task_B")
#    - Switch: model.set_adapter("task_B")
#
# =============================================================================
# LOADING WORKFLOW EXAMPLE:
# =============================================================================
#
# ```python
# from transformers import AutoModelForCausalLM
# from peft import PeftModel
#
# # 1. Load base model (500 MB download/load)
# base_model = AutoModelForCausalLM.from_pretrained("gpt2")
#
# # 2. Load adapter (3 MB download/load)
# peft_model = PeftModel.from_pretrained(
#     base_model,
#     "username/my-adapter",
#     adapter_name="default"
# )
#
# # 3. Generate with adapted model
# outputs = peft_model.generate(input_ids, max_length=50)
#
# # 4. (Optional) Load another adapter
# peft_model.load_adapter("username/another-adapter", "task_B")
# peft_model.set_adapter("task_B")
#
# # 5. (Optional) Merge and save full model
# merged_model = peft_model.merge_and_unload()
# merged_model.save_pretrained("merged_model")
# ```
#
# =============================================================================
# COMPARISON WITH AutoPeftModel:
# =============================================================================
#
# PeftModel.from_pretrained():
#   - User loads base model separately
#   - Two-step process: load base, load adapter
#   - More control over base model loading
#
# AutoPeftModelForCausalLM.from_pretrained():
#   - Automatically loads base model + adapter
#   - One-step process
#   - Convenience wrapper
#   - Internally calls PeftModel.from_pretrained()
#
# =============================================================================
# NEXT STEPS:
# =============================================================================
#
# After loading:
#   - For inference: model.generate(), model()
#   - For training: train with optimizer
#   - For saving: model.save_pretrained()
#   - For merging: model.merge_and_unload()
#
# To understand saving:
# See: annotated_code/03_adapter_saving_annotated.py
# See: guides/02_adapter_saving_deep_dive.md
#
# To understand merging:
# See: annotated_code/09_merge_adapter_annotated.py
#
# =============================================================================
