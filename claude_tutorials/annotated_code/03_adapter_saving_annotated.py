"""
=============================================================================
HEAVILY ANNOTATED: Adapter Saving Mechanism
HOW PEFT SAVES ONLY ADAPTER WEIGHTS (NOT BASE MODEL WEIGHTS)
=============================================================================

File Locations:
- Primary: src/peft/utils/save_and_load.py::get_peft_model_state_dict()
- Caller: src/peft/peft_model.py::PeftModel.save_pretrained()

WHAT: This shows how PEFT extracts ONLY the adapter parameters from a model,
      leaving out the base model weights entirely.

WHY THIS MATTERS:
- A full GPT-2 model is ~500MB
- LoRA adapters for GPT-2 can be ~3MB (167x smaller!)
- This allows sharing hundreds of adapters for one base model
- Users can swap adapters without reloading the entire model

HOW IT WORKS:
1. Iterate through full model state_dict
2. Filter keys containing adapter-specific markers (e.g., "lora_")
3. Remove adapter names from keys for portability
4. Handle special cases (bias, DoRA, embeddings)
5. Save to safetensors or PyTorch format

=============================================================================
"""

import os
import warnings
from typing import Optional, Union
import torch
from safetensors.torch import save_file as safe_save_file


# =============================================================================
# FUNCTION 1: get_peft_model_state_dict() - Extract Adapter Weights
# =============================================================================

def get_peft_model_state_dict(
    model,
    state_dict=None,
    adapter_name="default",
    unwrap_compiled=False,
    save_embedding_layers="auto"
):
    """
    Get the state dict of the given adapter of the PEFT model.

    ==========================================================================
    CRITICAL CONCEPT: ADAPTER-ONLY STATE DICT
    ==========================================================================

    This returns ONLY the PEFT parameters, NOT the base model parameters!

    FULL MODEL STATE DICT:
    {
        'transformer.h.0.attn.c_attn.weight': Tensor([768, 2304]),  # 1.77M params
        'transformer.h.0.attn.c_attn.lora_A.default.weight': Tensor([8, 768]),  # 6K params
        'transformer.h.0.attn.c_attn.lora_B.default.weight': Tensor([2304, 8]),  # 18K params
        'transformer.h.0.attn.c_proj.weight': Tensor([768, 768]),  # 590K params
        'transformer.h.0.attn.c_proj.lora_A.default.weight': Tensor([8, 768]),  # 6K params
        'transformer.h.0.attn.c_proj.lora_B.default.weight': Tensor([768, 8]),  # 6K params
        ... (many more layers)
    }

    PEFT STATE DICT (THIS FUNCTION'S RETURN VALUE):
    {
        # Adapter name "default" is REMOVED from keys!
        'transformer.h.0.attn.c_attn.lora_A.weight': Tensor([8, 768]),
        'transformer.h.0.attn.c_attn.lora_B.weight': Tensor([2304, 8]),
        'transformer.h.0.attn.c_proj.lora_A.weight': Tensor([8, 768]),
        'transformer.h.0.attn.c_proj.lora_B.weight': Tensor([768, 8]),
        ... (only adapter params)
    }

    SIZE COMPARISON:
    - Full GPT-2 state dict: ~500 MB
    - LoRA adapter state dict: ~3 MB
    - Reduction: 167x smaller!

    ==========================================================================
    """

    # ==========================================================================
    # STEP 1: HANDLE MODEL UNWRAPPING
    # ==========================================================================
    # WHAT: Handle torch.compile() wrapped models
    # WHY: torch.compile wraps model in _orig_mod, need to unwrap
    # ==========================================================================
    if unwrap_compiled:
        model = getattr(model, "_orig_mod", model)

    # ==========================================================================
    # STEP 2: GET CONFIGURATION AND STATE DICT
    # ==========================================================================
    config = model.peft_config[adapter_name]

    if state_dict is None:
        # Get full model state dict (includes base + adapter weights)
        state_dict = model.state_dict()

    # ==========================================================================
    # STEP 3: ADAPTER-TYPE-SPECIFIC FILTERING
    # ==========================================================================
    # WHAT: Filter state_dict based on adapter type
    # WHY: Different adapter types use different naming conventions
    # ==========================================================================

    if config.peft_type in (PeftType.LORA, PeftType.ADALORA):
        # ======================================================================
        # LORA / ADALORA STATE DICT EXTRACTION
        # ======================================================================
        # WHAT: Extract only keys containing "lora_"
        # HOW: Filter based on key patterns
        # WHY: LoRA parameters have "lora_A", "lora_B", "lora_embedding_A/B"
        # ======================================================================

        bias = config.bias  # Can be: "none", "all", or "lora_only"

        if bias == "none":
            # -------------------------------------------------------------------
            # CASE 1: NO BIAS TRAINING
            # -------------------------------------------------------------------
            # WHAT: Extract only LoRA weight matrices (no biases)
            # KEYS: "lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"
            # -------------------------------------------------------------------
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}

        elif bias == "all":
            # -------------------------------------------------------------------
            # CASE 2: ALL BIASES TRAINABLE
            # -------------------------------------------------------------------
            # WHAT: Extract LoRA weights + all bias terms
            # WHY: User wants to fine-tune biases too
            # WARNING: Increases adapter size
            # -------------------------------------------------------------------
            to_return = {
                k: state_dict[k]
                for k in state_dict
                if "lora_" in k or "bias" in k
            }

        elif bias == "lora_only":
            # -------------------------------------------------------------------
            # CASE 3: ONLY LORA LAYER BIASES
            # -------------------------------------------------------------------
            # WHAT: Extract LoRA weights + biases of LoRA-adapted layers
            # HOW: For each lora_* key, also include corresponding bias
            # EXAMPLE:
            #   If "transformer.h.0.attn.c_attn.lora_B.default" exists,
            #   also include "transformer.h.0.attn.c_attn.bias"
            # -------------------------------------------------------------------
            to_return = {}
            for k in state_dict:
                if "lora_" in k:
                    to_return[k] = state_dict[k]
                    # Derive bias key from lora key
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError(f"bias={bias} not implemented")

        # ======================================================================
        # STEP 3A: FILTER BY ADAPTER NAME
        # ======================================================================
        # WHAT: Keep only parameters belonging to this adapter
        # HOW: Check if adapter_name is in the key
        # WHY: Model may have multiple adapters (e.g., "task_A", "task_B")
        # ======================================================================
        to_return = {
            k: v for k, v in to_return.items()
            if (("lora_" in k and adapter_name in k) or ("bias" in k))
        }
        # EXAMPLE FILTERING:
        # BEFORE: 'transformer.h.0.attn.c_attn.lora_A.default.weight'
        #         'transformer.h.0.attn.c_attn.lora_A.task_A.weight'
        # AFTER (adapter_name="default"):
        #         'transformer.h.0.attn.c_attn.lora_A.default.weight' only

        # ======================================================================
        # STEP 3B: HANDLE ADALORA RANK PATTERNS
        # ======================================================================
        # WHAT: For AdaLoRA, resize state_dict based on learned ranks
        # WHY: AdaLoRA adaptively prunes low-importance ranks
        # HOW: Call resize_state_dict_by_rank_pattern()
        # ======================================================================
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                # Remove adapter name from rank_pattern keys
                rank_pattern = {
                    k.replace(f".{adapter_name}", ""): v
                    for k, v in rank_pattern.items()
                }
                config.rank_pattern = rank_pattern
                to_return = model.resize_state_dict_by_rank_pattern(
                    rank_pattern, to_return, adapter_name
                )

        # ======================================================================
        # STEP 3C: HANDLE DORA WEIGHT NAMING REFACTOR
        # ======================================================================
        # WHAT: Handle DoRA magnitude vector naming convention
        # WHY: DoRA was refactored from ParameterDict to ModuleDict
        #      Old: lora_magnitude_vector.default
        #      New: lora_magnitude_vector.default.weight
        # HOW: Remove ".weight" suffix for backward compatibility
        # ======================================================================
        if config.use_dora:
            new_dora_suffix = f"lora_magnitude_vector.{adapter_name}.weight"

            def renamed_dora_weights(k):
                if k.endswith(new_dora_suffix):
                    k = k[:-7]  # Remove ".weight" suffix
                return k

            to_return = {renamed_dora_weights(k): v for k, v in to_return.items()}

    # ==========================================================================
    # STEP 4: REMOVE ADAPTER NAME FROM KEYS
    # ==========================================================================
    # WHAT: Strip adapter name from state_dict keys
    # WHY: Makes adapters portable - can load with different adapter names
    # HOW: Replace f".{adapter_name}." with "."
    #
    # EXAMPLE TRANSFORMATION:
    # BEFORE: 'transformer.h.0.attn.c_attn.lora_A.default.weight'
    # AFTER:  'transformer.h.0.attn.c_attn.lora_A.weight'
    # ==========================================================================

    # This happens automatically through the filtering above
    # The actual name removal is handled in PeftModel.save_pretrained()

    return to_return


# =============================================================================
# FUNCTION 2: PeftModel.save_pretrained() - Save Adapters to Disk
# =============================================================================

class PeftModel:
    """Annotated excerpt from PeftModel.save_pretrained()"""

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        selected_adapters: Optional[list[str]] = None,
        save_embedding_layers: Union[str, bool] = "auto",
        is_main_process: bool = True,
        **kwargs,
    ) -> None:
        """
        Save adapter model and configuration files to a directory.

        =======================================================================
        WHAT GETS SAVED:
        =======================================================================

        For each adapter, two files are created:

        1. adapter_config.json (Configuration)
        {
            "peft_type": "LORA",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["c_attn", "c_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "base_model_name_or_path": "gpt2",
            ...
        }

        2. adapter_model.safetensors (Weights) OR adapter_model.bin
        {
            # Only adapter parameters, base model NOT included!
            "transformer.h.0.attn.c_attn.lora_A.weight": Tensor(...),
            "transformer.h.0.attn.c_attn.lora_B.weight": Tensor(...),
            ...
        }

        =======================================================================
        """

        # ======================================================================
        # STEP 1: CREATE SAVE DIRECTORY
        # ======================================================================
        if is_main_process:
            os.makedirs(save_directory, exist_ok=True)
            self.create_or_update_model_card(save_directory)

        # ======================================================================
        # STEP 2: DETERMINE WHICH ADAPTERS TO SAVE
        # ======================================================================
        if selected_adapters is None:
            # Save all adapters
            selected_adapters = list(self.peft_config.keys())

        # ======================================================================
        # STEP 3: SAVE EACH ADAPTER
        # ======================================================================
        for adapter_name in selected_adapters:
            peft_config = self.peft_config[adapter_name]

            # ==================================================================
            # STEP 3A: EXTRACT ADAPTER WEIGHTS ONLY
            # ==================================================================
            # THIS IS THE KEY STEP!
            # get_peft_model_state_dict() returns ONLY adapter parameters
            # ==================================================================
            output_state_dict = get_peft_model_state_dict(
                self,
                state_dict=kwargs.get("state_dict", None),
                adapter_name=adapter_name,
                save_embedding_layers=save_embedding_layers,
            )
            # RESULT: output_state_dict contains ~3MB of adapter weights
            #         instead of ~500MB of full model weights!

            # ==================================================================
            # STEP 3B: DETERMINE SAVE PATH
            # ==================================================================
            # If adapter_name is "default", save directly to save_directory
            # Otherwise, create subdirectory with adapter name
            # ==================================================================
            output_dir = (
                os.path.join(save_directory, adapter_name)
                if adapter_name != "default"
                else save_directory
            )
            os.makedirs(output_dir, exist_ok=True)

            # ==================================================================
            # STEP 3C: HANDLE TENSOR ALIASING FOR SAFETENSORS
            # ==================================================================
            # WHAT: Safetensors doesn't support tensor aliasing (shared storage)
            # HOW: Find shared tensors and clone them
            # WHY: Prevents errors when saving
            # ==================================================================
            if is_main_process and safe_serialization:
                import collections

                # Find all tensors that share storage
                ptrs = collections.defaultdict(list)
                for name, tensor in output_state_dict.items():
                    if isinstance(tensor, torch.Tensor):
                        ptrs[id_tensor_storage(tensor)].append(name)
                    else:
                        ptrs[id(tensor)].append(name)

                # Identify shared tensors
                shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

                # Clone shared tensors to avoid aliasing
                for _, names in shared_ptrs.items():
                    for shared_tensor_name in names[1:]:
                        output_state_dict[shared_tensor_name] = (
                            output_state_dict[shared_tensor_name].clone()
                        )

                # ==============================================================
                # STEP 3D: SAVE TO SAFETENSORS FORMAT
                # ==============================================================
                # WHAT: Save adapter weights in safetensors format
                # WHY: Safer, faster, and more portable than .bin
                # ==============================================================
                safe_save_file(
                    output_state_dict,
                    os.path.join(output_dir, "adapter_model.safetensors"),
                    metadata={"format": "pt"},
                )

            elif is_main_process:
                # ==============================================================
                # ALTERNATIVE: SAVE TO PYTORCH FORMAT (.bin)
                # ==============================================================
                torch.save(
                    output_state_dict,
                    os.path.join(output_dir, "adapter_model.bin")
                )

            # ==================================================================
            # STEP 3E: SAVE ADAPTER CONFIGURATION
            # ==================================================================
            # WHAT: Save the PeftConfig as JSON
            # WHY: Needed to reconstruct adapter when loading
            # ==================================================================
            if is_main_process:
                # Update base model path if needed
                if peft_config.base_model_name_or_path is None:
                    peft_config.base_model_name_or_path = (
                        self.base_model.__dict__.get("name_or_path", None)
                        if peft_config.is_prompt_learning
                        else self.base_model.model.__dict__.get("name_or_path", None)
                    )

                # Set inference mode to True for saved config
                inference_mode = peft_config.inference_mode
                peft_config.inference_mode = True

                # Save config as adapter_config.json
                peft_config.save_pretrained(output_dir)

                # Restore original inference mode
                peft_config.inference_mode = inference_mode


# =============================================================================
# KEY TAKEAWAYS - ADAPTER SAVING:
# =============================================================================
#
# 1. **SIZE REDUCTION**:
#    - Full model: ~500 MB (all parameters)
#    - LoRA adapter: ~3 MB (only adapter parameters)
#    - Reduction: 167x smaller!
#
# 2. **FILTERING MECHANISM**:
#    - Iterate through full state_dict
#    - Keep only keys containing adapter markers ("lora_", "ia3_", etc.)
#    - Filter by adapter_name for multi-adapter models
#    - Remove adapter_name from keys for portability
#
# 3. **SAVED FILES**:
#    - adapter_config.json: Configuration (r, alpha, target_modules, etc.)
#    - adapter_model.safetensors (or .bin): Adapter weights only
#    - No base model weights saved!
#
# 4. **PORTABILITY**:
#    - Adapter can be loaded with any adapter_name
#    - Can be shared independently of base model
#    - Multiple users can share adapters for same base model
#
# 5. **MULTI-ADAPTER SUPPORT**:
#    - Each adapter saved in separate subdirectory
#    - Default adapter saved in root of save_directory
#    - Can selectively save subset of adapters
#
# 6. **SAFETENSORS FORMAT**:
#    - Preferred over .bin for safety and speed
#    - Handles tensor aliasing automatically
#    - More portable across platforms
#
# =============================================================================
# LOADING ADAPTERS (Reverse Process):
# =============================================================================
#
# To load saved adapters:
#
# 1. Load base model:
#    model = AutoModelForCausalLM.from_pretrained("gpt2")
#
# 2. Load adapter:
#    model = PeftModel.from_pretrained(model, "path/to/adapter")
#
# What happens internally:
#    a) Load adapter_config.json
#    b) Initialize adapter layers (A and B matrices)
#    c) Load adapter_model.safetensors into adapter layers
#    d) Base model weights remain unchanged!
#
# This allows:
#    - Swapping adapters without reloading base model
#    - Loading multiple adapters simultaneously
#    - Minimal disk space and memory usage
#
# =============================================================================
