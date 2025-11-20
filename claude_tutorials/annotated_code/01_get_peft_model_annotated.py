"""
=============================================================================
HEAVILY ANNOTATED: get_peft_model() - The Primary Entry Point for PEFT
=============================================================================

File Location: src/peft/mapping_func.py

WHAT: This is the main user-facing function that transforms a regular PyTorch
model into a PEFT-enabled model by injecting adapter layers.

HOW: It wraps the base model in a PeftModel (or PeftMixedModel) class, which
internally delegates to a tuner class (e.g., LoraModel) that performs the
actual layer replacement/injection.

WHY: This abstraction allows users to add parameter-efficient adapters to any
model with a single function call, without manually modifying the model
architecture.

USAGE PATTERN:
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig

    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    config = LoraConfig(r=8, lora_alpha=32, target_modules=["c_attn"])
    peft_model = get_peft_model(base_model, config)

    # Now peft_model has LoRA adapters injected into c_attn layers
"""

from __future__ import annotations
import warnings
from typing import Optional
from transformers import PreTrainedModel

# Core imports for model type routing and configuration
from .auto import MODEL_TYPE_TO_PEFT_MODEL_MAPPING
from .config import PeftConfig
from .mapping import PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_PREFIX_MAPPING
from .mixed_model import PeftMixedModel
from .peft_model import PeftModel
from .tuners.tuners_utils import BaseTuner, BaseTunerLayer


def get_peft_model(
    model: PreTrainedModel,
    peft_config: PeftConfig,
    adapter_name: str = "default",
    mixed: bool = False,
    autocast_adapter_dtype: bool = True,
    revision: Optional[str] = None,
    low_cpu_mem_usage: bool = False,
) -> PeftModel | PeftMixedModel:
    """
    Returns a Peft model object from a model and a config, where the model will be modified in-place.

    =============================================================================
    PARAMETER DEEP DIVE:
    =============================================================================

    model ([`transformers.PreTrainedModel`]):
        *** WHAT: The base model to be adapted (e.g., GPT-2, BERT, LLaMA)
        *** HOW: This model's layers will be wrapped/replaced with adapter layers
        *** WHY: PEFT modifies existing models rather than creating new architectures
        *** IMPORTANT: The model is modified IN-PLACE during this process

    peft_config ([`PeftConfig`]):
        *** WHAT: Configuration object specifying adapter parameters
        *** HOW: Contains settings like rank (r), alpha, target_modules, etc.
        *** WHY: Defines which layers to adapt and how to configure the adapters
        *** EXAMPLES:
            - LoraConfig: Specifies rank, target modules for LoRA
            - PrefixTuningConfig: Specifies virtual tokens for prefix tuning

    adapter_name (`str`, defaults to "default"):
        *** WHAT: A unique identifier for this adapter
        *** HOW: Used as a key in internal dictionaries (e.g., lora_A, lora_B)
        *** WHY: Enables multi-adapter support - you can have multiple adapters
                 on the same model and switch between them
        *** EXAMPLE: "task_A", "task_B", "domain_specific"

    mixed (`bool`, defaults to False):
        *** WHAT: Whether to allow mixing different (compatible) adapter types
        *** HOW: Uses PeftMixedModel instead of PeftModel
        *** WHY: Allows combining LoRA + IA3, or other compatible combinations
        *** WHEN TO USE: Advanced scenarios requiring multiple adapter methods

    autocast_adapter_dtype (`bool`, defaults to True):
        *** WHAT: Whether to automatically cast adapter weights to float32
        *** HOW: Upcasts float16/bfloat16 adapters to float32 during training
        *** WHY: Ensures training stability - mixed precision can cause issues
                 with small adapter weights
        *** NOTE: Only affects select tuner types (LoRA, AdaLoRA)

    revision (`str`, optional):
        *** WHAT: Git revision of the base model
        *** HOW: Stored in config, used when loading saved adapters
        *** WHY: Ensures adapter compatibility with specific model versions
        *** EXAMPLE: "main", "v1.0", commit SHA

    low_cpu_mem_usage (`bool`, defaults to False):
        *** WHAT: Create empty adapter weights on meta device
        *** HOW: Uses init_empty_weights() context manager
        *** WHY: Speeds up loading when you'll immediately load pretrained weights
        *** WARNING: Don't use for training new adapters from scratch!
                     Only use when loading pretrained adapter weights

    =============================================================================
    """

    # =========================================================================
    # STEP 1: MODEL CONFIGURATION EXTRACTION
    # =========================================================================
    # WHAT: Get the configuration dict from the base model
    # HOW: Uses BaseTuner.get_model_config() which handles various model types
    # WHY: Need model_type (e.g., "llama", "gpt2") to infer default target_modules
    # =========================================================================
    model_config = BaseTuner.get_model_config(model)

    # =========================================================================
    # STEP 2: BASE MODEL PATH MANAGEMENT
    # =========================================================================
    # WHAT: Update peft_config with the actual base model path
    # HOW: Replace config's path with model's actual name_or_path
    # WHY: When saving adapters, we need to know which base model they work with
    # =========================================================================
    old_name = peft_config.base_model_name_or_path
    new_name = model.__dict__.get("name_or_path", None)
    peft_config.base_model_name_or_path = new_name

    # =========================================================================
    # STEP 3: DUPLICATE PEFT APPLICATION WARNING
    # =========================================================================
    # WHAT: Check if model already has PEFT adapters
    # HOW: Iterate through modules looking for BaseTunerLayer instances
    # WHY: Applying PEFT twice can cause unexpected behavior
    # BEST PRACTICE: Call model.unload() before applying new adapters
    # =========================================================================
    if any(isinstance(module, BaseTunerLayer) for module in model.modules()):
        warnings.warn(
            "You are trying to modify a model with PEFT for a second time. If you want to reload the model with a "
            "different config, make sure to call `.unload()` before."
        )

    # =========================================================================
    # STEP 4: BASE MODEL NAME CHANGE WARNING
    # =========================================================================
    # WHAT: Warn if base model name changed
    # HOW: Compare old and new base_model_name_or_path
    # WHY: When loading saved adapters, mismatched names can cause confusion
    # =========================================================================
    if (old_name is not None) and (old_name != new_name):
        warnings.warn(
            f"The PEFT config's `base_model_name_or_path` was renamed from '{old_name}' to '{new_name}'. "
            "Please ensure that the correct base model is loaded when loading this checkpoint."
        )

    # =========================================================================
    # STEP 5: REVISION MANAGEMENT
    # =========================================================================
    # WHAT: Set or override the base model revision
    # HOW: Update peft_config.revision if provided
    # WHY: Ensures saved adapters reference the correct model version
    # =========================================================================
    if revision is not None:
        if peft_config.revision is not None and peft_config.revision != revision:
            warnings.warn(
                f"peft config has already set base model revision to {peft_config.revision}, "
                f"overwriting with revision {revision}"
            )
        peft_config.revision = revision

    # =========================================================================
    # STEP 6: EVA INITIALIZATION OPTIMIZATION SUGGESTION
    # =========================================================================
    # WHAT: Suggest using low_cpu_mem_usage=True for EVA initialization
    # HOW: Check if using LoRA with EVA init and low_cpu_mem_usage=False
    # WHY: EVA can handle larger batches with low_cpu_mem_usage=True
    # CONTEXT: EVA (Explained Variance Adaptation) is a data-driven LoRA init
    # =========================================================================
    if (
        (isinstance(peft_config, PEFT_TYPE_TO_CONFIG_MAPPING["LORA"]))
        and (peft_config.init_lora_weights == "eva")
        and not low_cpu_mem_usage
    ):
        warnings.warn(
            "lora with eva initialization used with low_cpu_mem_usage=False. "
            "Setting low_cpu_mem_usage=True can improve the maximum batch size possible for eva initialization."
        )

    # =========================================================================
    # STEP 7: ADAPTER NAME PREFIX COLLISION CHECK
    # =========================================================================
    # WHAT: Ensure adapter name doesn't contain the PEFT method prefix
    # HOW: Check if adapter_name is substring of the prefix (e.g., "lora_")
    # WHY: Prevents naming collisions during state_dict operations
    # EXAMPLE: Don't name an adapter "lora_default" when using LoRA
    #          (prefix is "lora_", so this would cause issues)
    # =========================================================================
    prefix = PEFT_TYPE_TO_PREFIX_MAPPING.get(peft_config.peft_type)
    if prefix and adapter_name in prefix:
        warnings.warn(
            f"Adapter name '{adapter_name}' should not be contained in the prefix '{prefix}'. "
            "This may lead to reinitialization of the adapter weights during loading."
        )

    # =========================================================================
    # STEP 8: MIXED ADAPTER MODEL ROUTING
    # =========================================================================
    # WHAT: Create PeftMixedModel if mixed adapters requested
    # HOW: Return early with PeftMixedModel instead of continuing
    # WHY: Mixed adapters require special handling to combine different types
    # NOTE: autocast_adapter_dtype not supported for mixed models
    # =========================================================================
    if mixed:
        # PeftMixedModel doesn't support autocast_adapter_dtype
        return PeftMixedModel(model, peft_config, adapter_name=adapter_name)

    # =========================================================================
    # STEP 9: TASK-TYPE-SPECIFIC MODEL SELECTION
    # =========================================================================
    # WHAT: Choose the appropriate PeftModel subclass based on task type
    # HOW: Look up task_type in MODEL_TYPE_TO_PEFT_MODEL_MAPPING
    # WHY: Different tasks (causal LM, seq2seq, classification) need different
    #      forward methods and output handling
    #
    # TASK TYPES AND THEIR MODELS:
    # - CAUSAL_LM → PeftModelForCausalLM
    #   * Autoregressive generation (GPT-style)
    #   * Modified forward for language modeling head
    #
    # - SEQ_2_SEQ_LM → PeftModelForSeq2SeqLM
    #   * Encoder-decoder models (T5, BART)
    #   * Handles both encoder and decoder adaptation
    #
    # - SEQ_CLS → PeftModelForSequenceClassification
    #   * Classification tasks
    #   * Modified forward for classification head
    #
    # - TOKEN_CLS → PeftModelForTokenClassification
    #   * Token-level tasks (NER, POS tagging)
    #
    # - QUESTION_ANS → PeftModelForQuestionAnswering
    #   * Span-based QA (SQuAD-style)
    #
    # - FEATURE_EXTRACTION → PeftModelForFeatureExtraction
    #   * Embedding extraction without task head
    #
    # PROMPT LEARNING EXCLUSION:
    # Prompt learning methods (prefix tuning, prompt tuning, p-tuning)
    # are excluded because they modify the INPUT rather than layer weights,
    # requiring special forward pass handling in PeftModel
    # =========================================================================
    if peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys() and not peft_config.is_prompt_learning:
        # GENERIC PEFT MODEL (No task-specific modifications)
        # Used when task_type is None or not in the mapping
        return PeftModel(
            model,
            peft_config,
            adapter_name=adapter_name,
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

    # TASK-SPECIFIC PEFT MODEL
    # Look up and instantiate the appropriate PeftModel subclass
    # This provides task-specific forward methods and output handling
    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](
        model,
        peft_config,
        adapter_name=adapter_name,
        autocast_adapter_dtype=autocast_adapter_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )


# =============================================================================
# KEY TAKEAWAYS:
# =============================================================================
#
# 1. **In-Place Modification**: The original model is modified in-place, not copied
#    - Memory efficient: no duplication of base weights
#    - Direct modification: layers are replaced with adapter-wrapped versions
#
# 2. **Routing Logic**: Function routes to different model types:
#    - Mixed adapters → PeftMixedModel
#    - Task-specific → PeftModelForCausalLM, PeftModelForSeq2SeqLM, etc.
#    - Generic → PeftModel
#
# 3. **Configuration Management**: Extensive validation and warning system:
#    - Checks for duplicate PEFT application
#    - Validates adapter naming conventions
#    - Manages base model path tracking
#
# 4. **Multi-Adapter Support**: adapter_name parameter enables:
#    - Multiple adapters on same model
#    - Runtime adapter switching
#    - Adapter merging and composition
#
# 5. **Model Type Inference**: Uses MODEL_TYPE_TO_PEFT_MODEL_MAPPING:
#    - Maps task types to appropriate PeftModel subclasses
#    - Ensures correct forward pass for each task
#    - Handles special cases (prompt learning, mixed adapters)
#
# =============================================================================
# NEXT STEPS IN THE PIPELINE:
# =============================================================================
#
# After get_peft_model() returns, the following has occurred:
#
# 1. PeftModel.__init__() was called, which:
#    - Created a tuner instance (e.g., LoraModel)
#    - The tuner's __init__() called inject_adapter()
#
# 2. inject_adapter() (in BaseTuner):
#    - Iterated through all model layers
#    - Identified target layers matching target_modules
#    - Replaced them with adapter-wrapped versions
#
# 3. Result: A model where:
#    - Base weights are unchanged and frozen
#    - Adapter layers are added and trainable
#    - Forward pass now includes adapter computations
#
# See annotated_code/02_lora_config_annotated.py for LoraConfig details
# See annotated_code/03_peft_model_init_annotated.py for PeftModel initialization
# See annotated_code/04_inject_adapter_annotated.py for injection mechanics
#
# =============================================================================
