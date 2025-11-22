"""
=============================================================================
HEAVILY ANNOTATED: BaseTuner.inject_adapter() - The Main Injection Loop
=============================================================================

File Location: src/peft/tuners/tuners_utils.py:665-870

WHAT: This is the MAIN INJECTION LOOP that iterates through all model layers
      and replaces target modules with adapter-enhanced versions.

HOW: Three-stage process:
     1. Preparation: Validate config, prepare model
     2. Iteration: Loop through all modules, check if they match target_modules
     3. Replacement: Call _create_and_replace() for matched modules

WHY: This is the core mechanism that transforms a base model into a PEFT model.
     Understanding this function is CRITICAL to understanding PEFT internals.

CALL CHAIN:
    get_peft_model()
        → PeftModel.__init__()
            → LoraModel.__init__() [inherits BaseTuner]
                → BaseTuner.__init__()
                    → inject_adapter() ← YOU ARE HERE
                        → _create_and_replace() [for each matched module]

=============================================================================
"""

from typing import Optional
import torch
from torch import nn


class BaseTuner(nn.Module):
    """Base class for all PEFT tuners (LoRA, IA3, AdaLoRA, etc.)"""

    def inject_adapter(
        self,
        model: nn.Module,
        adapter_name: str,
        autocast_adapter_dtype: bool = True,
        low_cpu_mem_usage: bool = False,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        Creates adapter layers and replaces target modules with adapter layers.

        =======================================================================
        HIGH-LEVEL OVERVIEW:
        =======================================================================

        This function performs the "injection" of adapters into a model.

        INPUT:
            model: Base model (e.g., GPT-2 with 124M params)

        OUTPUT:
            model: Same model, but with adapted layers
                   (e.g., nn.Linear → LoraLinear)

        EXAMPLE TRANSFORMATION:
            Before: model.transformer.h[0].attn.c_attn (nn.Linear)
            After:  model.transformer.h[0].attn.c_attn (LoraLinear wrapping nn.Linear)

        =======================================================================
        """

        # ===================================================================
        # STAGE 1: PREPARATION OF MODEL AND CONFIG
        # ===================================================================
        # WHAT: Set up everything needed before injection
        # HOW: Validate config, prepare model, infer target modules
        # WHY: Catch errors early before partial model modification
        # ===================================================================

        # -------------------------------------------------------------------
        # STEP 1.1: Get PEFT Configuration
        # -------------------------------------------------------------------
        # WHERE: self.peft_config is a dict {adapter_name: PeftConfig}
        # EXAMPLE: {"default": LoraConfig(r=8, target_modules=["q_proj"])}
        # -------------------------------------------------------------------
        peft_config = self.peft_config[adapter_name]

        # Initialize tracking lists
        excluded_modules = []  # Modules explicitly excluded
        unmatched_modules = []  # Modules that didn't match target_modules
        targeted_modules_from_peft_config: list[str] = []

        # -------------------------------------------------------------------
        # STEP 1.2: Validate New Adapter Configuration
        # -------------------------------------------------------------------
        # WHAT: Check if new adapter config is compatible with existing ones
        # HOW: Calls _check_new_adapter_config() which validates:
        #      - Only one adapter can use bias != "none"
        #      - No conflicting configurations
        # WHY: Prevent incompatible adapters on same model
        # -------------------------------------------------------------------
        self._check_new_adapter_config(peft_config)

        # -------------------------------------------------------------------
        # STEP 1.3: Check for Tied Weight Modules
        # -------------------------------------------------------------------
        # WHAT: Identify modules with tied/shared weights
        # WHY: Some models share weights (e.g., input/output embeddings)
        #      Need special handling to maintain weight tying
        # EXAMPLE: GPT-2 ties word embeddings between input and output
        # -------------------------------------------------------------------
        self._check_tied_modules(model, peft_config)

        # -------------------------------------------------------------------
        # STEP 1.4: Get Model Configuration
        # -------------------------------------------------------------------
        # WHAT: Extract model config dict (e.g., {"model_type": "gpt2"})
        # HOW: Handles transformers models and custom models
        # WHY: Need model_type to infer default target_modules
        # -------------------------------------------------------------------
        model_config = self.get_model_config(model)
        # EXAMPLE OUTPUT: {"model_type": "gpt2", "vocab_size": 50257, ...}

        # -------------------------------------------------------------------
        # STEP 1.5: Prepare Adapter Configuration
        # -------------------------------------------------------------------
        # WHAT: Finalize peft_config (infer target_modules if needed)
        # HOW: Calls _prepare_adapter_config() which:
        #      - If target_modules is None, infer from model_type
        #      - Uses TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
        # WHY: User convenience - don't require specifying target_modules
        # -------------------------------------------------------------------
        peft_config = self._prepare_adapter_config(peft_config, model_config)
        # EXAMPLE: For GPT-2, if target_modules=None, sets to ["c_attn", "c_proj"]

        # -------------------------------------------------------------------
        # STEP 1.6: Prepare Model Structure
        # -------------------------------------------------------------------
        # WHAT: Modify model structure before injection (if needed)
        # HOW: Calls _prepare_model() which can:
        #      - Replicate layers (layer_replication config)
        #      - Add auxiliary modules
        # WHY: Some PEFT methods need structural changes first
        # EXAMPLE: Layer replication expands model depth
        # -------------------------------------------------------------------
        self._prepare_model(peft_config, model)

        # -------------------------------------------------------------------
        # STEP 1.7: Validate state_dict Usage
        # -------------------------------------------------------------------
        # WHAT: Ensure state_dict and target_parameters aren't both used
        # WHY: target_parameters uses different injection mechanism
        # -------------------------------------------------------------------
        if getattr(peft_config, "target_parameters", []) and state_dict:
            raise ValueError(
                "Trying to inject a PEFT adapter from a state_dict but the PEFT config uses `target_parameters`. "
                "This is not supported -- when using `target_parameters`, please inject the adapter without the "
                "state_dict."
            )

        # -------------------------------------------------------------------
        # STEP 1.8: Get All Module Names
        # -------------------------------------------------------------------
        # WHAT: List all modules in the model with their keys
        # HOW: model.named_modules() returns [(key, module), ...]
        # WHY: Need to iterate through all modules to find matches
        # -------------------------------------------------------------------
        named_modules = list(model.named_modules())
        key_list = [key for key, _ in named_modules]
        # EXAMPLE:
        # named_modules = [
        #     ("", GPT2LMHeadModel),
        #     ("transformer", GPT2Model),
        #     ("transformer.wte", Embedding),
        #     ("transformer.h.0", GPT2Block),
        #     ("transformer.h.0.attn", GPT2Attention),
        #     ("transformer.h.0.attn.c_attn", Conv1D),  ← TARGET!
        #     ("transformer.h.0.attn.c_proj", Conv1D),  ← TARGET!
        #     ...
        # ]

        # -------------------------------------------------------------------
        # STEP 1.9: Handle Dummy Target Modules
        # -------------------------------------------------------------------
        # WHAT: Special case for dummy adapters (testing/debugging)
        # WHY: Allow creating adapter without matching any modules
        # -------------------------------------------------------------------
        uses_dummy_target_modules = (
            getattr(peft_config, "target_modules", None) == DUMMY_TARGET_MODULES
        )
        if uses_dummy_target_modules:
            named_modules = []
            key_list = []

        # -------------------------------------------------------------------
        # STEP 1.10: Expand "all-linear" Target Modules
        # -------------------------------------------------------------------
        # WHAT: If target_modules="all-linear", expand to all Linear layers
        # HOW: _maybe_include_all_linear_layers() finds all nn.Linear
        # WHY: Convenience for users who want to adapt all linear layers
        # -------------------------------------------------------------------
        peft_config = _maybe_include_all_linear_layers(peft_config, model)
        # EXAMPLE: "all-linear" → ["c_attn", "c_proj", "c_fc", "lm_head", ...]

        # -------------------------------------------------------------------
        # STEP 1.11: Optimize Target Modules List (Performance)
        # -------------------------------------------------------------------
        # WHAT: Condense target_modules list if it's very large
        # HOW: Find minimal set of patterns that cover all targets
        # WHY: Optimization for large target_modules lists (hundreds of entries)
        # WHEN: Happens when loading non-PEFT checkpoints (e.g., diffusers LoRAs)
        # -------------------------------------------------------------------
        if (
            isinstance(peft_config.target_modules, (list, set))
            and (len(peft_config.target_modules) >= MIN_TARGET_MODULES_FOR_OPTIMIZATION)
            and (peft_config.peft_type != PeftType.IA3)
        ):
            # Find minimal set of suffixes that match all target modules
            suffixes = tuple("." + suffix for suffix in peft_config.target_modules)
            names_no_target = [
                name for name in key_list
                if (name not in peft_config.target_modules) and not name.endswith(suffixes)
            ]
            new_target_modules = _find_minimal_target_modules(
                peft_config.target_modules, names_no_target
            )
            if len(new_target_modules) < len(peft_config.target_modules):
                peft_config.target_modules = new_target_modules

        # ===================================================================
        # STAGE 2: MATCHING & CREATING MODULES
        # ===================================================================
        # WHAT: The MAIN LOOP - iterate through modules and inject adapters
        # HOW: For each module, check if it matches target_modules
        #      If yes, call _create_and_replace()
        # WHY: This is where the actual injection happens!
        # ===================================================================

        # -------------------------------------------------------------------
        # STEP 2.1: Find Existing Adapter Prefixes
        # -------------------------------------------------------------------
        # WHAT: If adding another adapter, identify already-adapted modules
        # HOW: Look for modules that are already BaseTunerLayer instances
        # WHY: Skip adapter internals when adding additional adapters
        # -------------------------------------------------------------------
        existing_adapter_prefixes = []
        for key, module in named_modules:
            if isinstance(module, BaseTunerLayer):
                existing_adapter_prefixes.append(key + ".")
        # EXAMPLE: If adapter "task_A" exists, might have:
        # ["transformer.h.0.attn.c_attn.", "transformer.h.0.attn.c_proj.", ...]

        # -------------------------------------------------------------------
        # STEP 2.2: Extract Module Names from State Dict (if provided)
        # -------------------------------------------------------------------
        # WHAT: If state_dict provided, use it to determine which modules to adapt
        # HOW: Extract module names from state_dict keys
        # WHY: Allows loading adapters without knowing exact target_modules
        # -------------------------------------------------------------------
        module_names: set[str] = set()
        if state_dict is not None:
            prefix = PEFT_TYPE_TO_PREFIX_MAPPING[peft_config.peft_type]
            # EXAMPLE: prefix = "lora_" for LoRA
            module_names = {k.rsplit("." + prefix, 1)[0] for k in state_dict}
            # EXAMPLE: "transformer.h.0.attn.c_attn.lora_A.weight"
            #       → "transformer.h.0.attn.c_attn"

        # ===================================================================
        # THE MAIN INJECTION LOOP
        # ===================================================================
        # WHAT: Iterate through EVERY module in the model
        # HOW: For each module, check if it should be adapted
        # WHY: This is where nn.Linear becomes LoraLinear!
        # ===================================================================

        for key, module in named_modules:
            # ---------------------------------------------------------------
            # SKIP: Empty keys
            # ---------------------------------------------------------------
            if not key:
                continue

            # ---------------------------------------------------------------
            # SKIP: Adapter Internals (for multi-adapter)
            # ---------------------------------------------------------------
            # WHAT: If this module is inside an existing adapter, skip it
            # WHY: Don't want to adapt adapter internals
            # EXAMPLE: Skip "transformer.h.0.attn.c_attn.lora_A" if
            #          "transformer.h.0.attn.c_attn." is already adapted
            # ---------------------------------------------------------------
            for adapter_key in existing_adapter_prefixes:
                if key.startswith(adapter_key):
                    excluded_modules.append(key)
                    break

            if excluded_modules and excluded_modules[-1] == key:
                continue

            # ===============================================================
            # DECISION POINT: Match Using Config or State Dict?
            # ===============================================================

            if state_dict is None:
                # -----------------------------------------------------------
                # PATH A: NORMAL MATCHING (using peft_config)
                # -----------------------------------------------------------
                # WHAT: Check if this module matches target_modules pattern
                # HOW: Call _check_target_module_exists()
                # RETURNS:
                #   - True/Match: Module should be adapted
                #   - False/None: Module doesn't match
                #   - _ExcludedModule: Module explicitly excluded
                # -----------------------------------------------------------
                result = self._check_target_module_exists(peft_config, key)

                if isinstance(result, _ExcludedModule):
                    # Module explicitly excluded via exclude_modules
                    excluded_modules.append(key)

                elif not result:
                    # Module doesn't match target_modules
                    unmatched_modules.append(key)

                else:
                    # *** MODULE MATCHES! INJECT ADAPTER! ***
                    self.targeted_module_names.append(key)

                    # Get parent module and target module name
                    parent, target, target_name = _get_submodules(model, key)
                    # EXAMPLE:
                    # key = "transformer.h.0.attn.c_attn"
                    # parent = GPT2Attention object
                    # target = Conv1D object (the actual layer)
                    # target_name = "c_attn"

                    # Check compatibility (e.g., Mamba-specific checks)
                    self._check_target_module_compatiblity(peft_config, model, target_name)

                    # Use low CPU memory if requested
                    ctx = init_empty_weights if low_cpu_mem_usage else nullcontext

                    with ctx():
                        # ===================================================
                        # *** THE REPLACEMENT HAPPENS HERE ***
                        # ===================================================
                        # WHAT: Replace target module with adapter version
                        # HOW: Calls subclass-specific _create_and_replace()
                        # RESULT: parent.c_attn is now LoraLinear, not Conv1D
                        # ===================================================
                        self._create_and_replace(
                            peft_config,
                            adapter_name,
                            target,        # The module to replace
                            target_name,   # Its name in parent
                            parent,        # The parent module
                            current_key=key,  # Full path
                        )
                        # AFTER THIS CALL:
                        # parent.c_attn is now a LoraLinear object
                        # that wraps the original Conv1D layer

            else:
                # -----------------------------------------------------------
                # PATH B: STATE DICT MATCHING
                # -----------------------------------------------------------
                # WHAT: Match modules based on state_dict keys
                # WHY: When loading checkpoint without knowing exact config
                # -----------------------------------------------------------
                if key not in module_names:
                    unmatched_modules.append(key)
                else:
                    # Module found in state_dict, inject adapter
                    self.targeted_module_names.append(key)
                    parent, target, target_name = _get_submodules(model, key)
                    self._check_target_module_compatiblity(peft_config, model, target_name)

                    ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
                    with ctx():
                        self._create_and_replace(
                            peft_config, adapter_name, target, target_name, parent, current_key=key
                        )

                # Still record what config would have matched (for comparison)
                if self._check_target_module_exists(peft_config, key):
                    targeted_modules_from_peft_config.append(key)

        # -------------------------------------------------------------------
        # STEP 2.3: Inject Parameters (if using target_parameters)
        # -------------------------------------------------------------------
        # WHAT: Alternative injection for nn.Parameter targets
        # WHY: Some models use nn.Parameter directly (e.g., MoE experts)
        # HOW: Different mechanism using ParamWrapper
        # -------------------------------------------------------------------
        if getattr(peft_config, "target_parameters", []):
            self._inject_parameters(
                peft_config=peft_config,
                model=model,
                adapter_name=adapter_name,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )

        # ===================================================================
        # STAGE 3: ERROR CHECKING AND VALIDATION
        # ===================================================================
        # WHAT: Verify injection was successful and warn about issues
        # HOW: Check for mismatches, missing targets, etc.
        # WHY: Help users debug configuration problems
        # ===================================================================

        # -------------------------------------------------------------------
        # STEP 3.1: Warn About State Dict Mismatches
        # -------------------------------------------------------------------
        # WHAT: If state_dict was used, compare with config-based matching
        # WHY: User might have wrong target_modules in config
        # -------------------------------------------------------------------
        if state_dict is not None:
            targeted_set_from_peft_config = set(targeted_modules_from_peft_config)
            targeted_set_from_state_dict = set(self.targeted_module_names)

            diff_peft_config = targeted_set_from_peft_config - targeted_set_from_state_dict
            diff_state_dict = targeted_set_from_state_dict - targeted_set_from_peft_config

            if diff_peft_config or diff_state_dict:
                warning_msg = (
                    "While injecting the PEFT adapters, an inconsistency was discovered between the PEFT config and "
                    "the provided state_dict. This is not necessarily an issue and can be ignored if this was the "
                    "intent. "
                )
                if diff_peft_config:
                    warning_msg += (
                        f"The PEFT config contained these additional target modules: {sorted(diff_peft_config)}. "
                    )
                if diff_state_dict:
                    warning_msg += (
                        f"The state_dict contained these additional target modules: {sorted(diff_state_dict)}."
                    )
                warnings.warn(warning_msg)

        # -------------------------------------------------------------------
        # STEP 3.2: Warn About No Matches
        # -------------------------------------------------------------------
        # WHAT: If no modules were matched, warn user
        # WHY: Likely a configuration error
        # -------------------------------------------------------------------
        if not uses_dummy_target_modules and not self.targeted_module_names:
            # Additional checks for common errors...
            raise ValueError(
                f"Target modules {peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

        # -------------------------------------------------------------------
        # STEP 3.3: Mark Only Adapters as Trainable
        # -------------------------------------------------------------------
        # WHAT: Freeze base model, unfreeze adapter parameters
        # HOW: Set requires_grad=False for non-adapter params
        # WHY: PEFT's whole point - only train adapters!
        # -------------------------------------------------------------------
        self._mark_only_adapters_as_trainable(model)
        # AFTER THIS:
        # - Base model weights: requires_grad=False
        # - LoRA A/B matrices: requires_grad=True


# =============================================================================
# SUMMARY: What inject_adapter() Does
# =============================================================================
#
# INPUT:
#   model: Base model (e.g., GPT-2)
#   adapter_name: "default" (or custom name)
#   peft_config: LoraConfig(r=8, target_modules=["c_attn", "c_proj"])
#
# PROCESS:
#   1. Prepare config (infer target_modules if needed)
#   2. Iterate through ALL modules in model
#   3. For each module matching target_modules:
#      - Call _create_and_replace()
#      - Replaces nn.Linear with LoraLinear
#   4. Mark only adapters as trainable
#
# OUTPUT:
#   Same model object, but with modified layers:
#   - transformer.h[0].attn.c_attn: Conv1D → LoraLinear(wrapping Conv1D)
#   - transformer.h[0].attn.c_proj: Conv1D → LoraLinear(wrapping Conv1D)
#   - ... (for all 48 layers)
#
# RESULT:
#   - Base weights: 124M params, frozen
#   - Adapter weights: ~1.2M params, trainable (0.97%)
#
# =============================================================================
# KEY INSIGHTS:
# =============================================================================
#
# 1. **In-Place Modification**: The original model is modified directly
#    - No copying of base weights
#    - Layers replaced using setattr() in _create_and_replace()
#
# 2. **Selective Targeting**: Not all layers are adapted
#    - Only those matching target_modules pattern
#    - Can use regex, exact match, or "all-linear"
#
# 3. **Multi-Adapter Support**: Can add multiple adapters to same model
#    - existing_adapter_prefixes tracks already-adapted modules
#    - Each adapter has independent A/B matrices
#
# 4. **Error Handling**: Extensive validation
#    - Catches missing targets early
#    - Warns about config/state_dict mismatches
#    - Validates compatibility (e.g., Mamba models)
#
# 5. **State Dict Loading**: Alternative to config-based matching
#    - Useful when exact target_modules unknown
#    - Infers targets from checkpoint keys
#
# =============================================================================
# NEXT STEPS:
# =============================================================================
#
# After inject_adapter() completes, the model has:
# - Adapter layers injected at target locations
# - Base weights frozen, adapter weights trainable
# - Ready for training or inference
#
# To understand what happens INSIDE _create_and_replace():
# See: annotated_code/05_create_and_replace_annotated.py
#
# To understand the actual replacement mechanism:
# See: guides/01_model_injection_deep_dive.md
#
# =============================================================================
