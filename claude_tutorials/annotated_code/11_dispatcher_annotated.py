"""
=============================================================================
HEAVILY ANNOTATED: _create_new_module() - Layer Type Dispatcher
=============================================================================

File Location: src/peft/tuners/lora/model.py:284-348

WHAT: Dispatches to correct LoRA layer implementation based on:
      - Layer type (Linear, Conv2d, Embedding, etc.)
      - Quantization method (bitsandbytes, GPTQ, AWQ, etc.)
      - Custom user modules

WHY: Different layer types and quantization methods need specialized
     LoRA implementations with different forward passes

DISPATCHER CHAIN (priority order):
1. Custom modules (user-provided)
2. bitsandbytes 8-bit
3. bitsandbytes 4-bit
4. EETQ
5. AQLM
6. AWQ
7. GPTQ
8. HQQ
9. Intel Neural Compressor
10. TorchAO
11. Megatron-LM
12. Default (nn.Linear, nn.Conv2d, etc.)

CRITICAL: First match wins! Order matters.

CALL CHAIN:
    _create_and_replace()
        → _create_new_module() ← YOU ARE HERE
            → dispatch_bnb_8bit() or dispatch_default(), etc.
                → Returns LoraLinear, Linear8bitLt, etc.
"""

from typing import Optional
import torch
from torch import nn


class LoraModel(BaseTuner):
    """LoRA model - handles LoRA injection"""

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        """
        Create appropriate LoRA layer for target module.

        PARAMETERS:
            lora_config: LoraConfig with r, alpha, etc.
            adapter_name: Name of adapter being added
            target: Original module to wrap (nn.Linear, etc.)
            **kwargs: r, lora_alpha, loaded_in_4bit, etc.

        RETURNS:
            New LoRA-enhanced module (e.g., LoraLinear)

        EXAMPLE:
            Input: target = nn.Linear(768, 2304)
            Output: LoraLinear(base_layer=nn.Linear(...), lora_A=..., lora_B=...)
        """

        # ===================================================================
        # BUILD DISPATCHER LIST
        # ===================================================================
        # WHAT: Collect all dispatch functions in priority order
        # WHY: Try each dispatcher until one returns a module
        # ===================================================================
        dispatchers = []

        # ===================================================================
        # 1. CUSTOM MODULES (Highest Priority)
        # ===================================================================
        # WHAT: User-provided custom LoRA implementations
        # WHY: Allow extending PEFT with custom layer types
        # WHEN: lora_config._custom_modules is set
        # ===================================================================
        if lora_config._custom_modules:
            def dynamic_dispatch_func(target, adapter_name, lora_config, **kwargs):
                new_module = None

                # Get base layer (unwrap if already LoRA)
                if isinstance(target, BaseTunerLayer):
                    target_base_layer = target.get_base_layer()
                else:
                    target_base_layer = target

                # Check if any custom class matches
                for key, custom_cls in lora_config._custom_modules.items():
                    if isinstance(target_base_layer, key):
                        new_module = custom_cls(target, adapter_name, **kwargs)
                        break

                return new_module

            dispatchers.append(dynamic_dispatch_func)

        # ===================================================================
        # 2. BITSANDBYTES 8-BIT
        # ===================================================================
        # WHAT: LoRA for 8-bit quantized layers (bitsandbytes)
        # WHY: 8-bit quantization saves memory
        # RETURNS: Linear8bitLt (specialized LoRA)
        # ===================================================================
        if is_bnb_available():
            from .bnb import dispatch_bnb_8bit
            dispatchers.append(dispatch_bnb_8bit)

        # ===================================================================
        # 3. BITSANDBYTES 4-BIT
        # ===================================================================
        # WHAT: LoRA for 4-bit quantized layers (bitsandbytes)
        # WHY: 4-bit quantization saves even more memory
        # RETURNS: Linear4bit (specialized LoRA)
        # ===================================================================
        if is_bnb_4bit_available():
            from .bnb import dispatch_bnb_4bit
            dispatchers.append(dispatch_bnb_4bit)

        # ===================================================================
        # 4-11. OTHER QUANTIZATION METHODS
        # ===================================================================
        # Each quantization method has specialized LoRA implementation:
        # - EETQ: Efficient quantization
        # - AQLM: Additive quantization
        # - AWQ: Activation-aware weight quantization
        # - GPTQ: Post-training quantization
        # - HQQ: Half-quadratic quantization
        # - Intel NC: Intel Neural Compressor
        # - TorchAO: TorchAO quantization
        # - Megatron: Megatron-LM tensor parallel
        # ===================================================================
        dispatchers.extend([
            dispatch_eetq,
            dispatch_aqlm,
            dispatch_awq,
            dispatch_gptq,
            dispatch_hqq,
            dispatch_inc,
            dispatch_torchao,
            dispatch_megatron,
            dispatch_default,  # Catch-all
        ])

        # ===================================================================
        # TRY EACH DISPATCHER
        # ===================================================================
        # WHAT: Call each dispatcher until one returns a module
        # HOW: First match wins!
        # ===================================================================
        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)
            if new_module is not None:
                # Found a match!
                break

        # ===================================================================
        # ERROR IF NO MATCH
        # ===================================================================
        if new_module is None:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following "
                "modules are supported: `torch.nn.Linear`, `torch.nn.Embedding`, "
                "`torch.nn.Conv1d`, `torch.nn.Conv2d`, `torch.nn.Conv3d`, "
                "`transformers.pytorch_utils.Conv1D`, `torch.nn.MultiheadAttention`."
            )

        return new_module


# =============================================================================
# DISPATCHER EXAMPLES
# =============================================================================

def dispatch_default(target, adapter_name, lora_config, **kwargs):
    """
    Default dispatcher - handles standard PyTorch layers.

    Supports:
    - nn.Linear → LoraLinear
    - nn.Embedding → LoraEmbedding
    - nn.Conv1d → LoraConv1d
    - nn.Conv2d → LoraConv2d
    - nn.Conv3d → LoraConv3d
    - Conv1D (transformers) → LoraLinear
    - nn.MultiheadAttention → LoraMultiheadAttention
    """

    from .layer import Linear, Embedding, Conv2d
    from transformers.pytorch_utils import Conv1D

    # Unwrap if already LoRA layer
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    # Match layer type
    if isinstance(target_base_layer, torch.nn.Linear):
        return Linear(target, adapter_name, **kwargs)

    elif isinstance(target_base_layer, Conv1D):
        # transformers Conv1D (used in GPT-2)
        return Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    elif isinstance(target_base_layer, torch.nn.Embedding):
        return Embedding(target, adapter_name, **kwargs)

    elif isinstance(target_base_layer, torch.nn.Conv2d):
        return Conv2d(target, adapter_name, **kwargs)

    # ... (Conv1d, Conv3d, MultiheadAttention, etc.)

    return None  # No match


def dispatch_bnb_4bit(target, adapter_name, lora_config, **kwargs):
    """
    Dispatcher for 4-bit quantized layers (bitsandbytes).

    Checks if target is Linear4bit, returns specialized LoRA if yes.
    """

    from .bnb import Linear4bit

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    # Check if 4-bit quantized
    if isinstance(target_base_layer, bnb.nn.Linear4bit):
        return Linear4bit(target, adapter_name, **kwargs)

    return None  # Not a match


# =============================================================================
# SUMMARY: What _create_new_module() Does
# =============================================================================
#
# INPUT:
#   target = nn.Linear(768, 2304) (or quantized variant)
#   adapter_name = "default"
#   lora_config = LoraConfig(r=8, ...)
#   kwargs = {r=8, lora_alpha=16, loaded_in_4bit=False, ...}
#
# PROCESS:
#   1. Build list of dispatcher functions
#   2. Try each dispatcher in order:
#      - Custom modules
#      - bitsandbytes 8-bit/4-bit
#      - GPTQ, AWQ, AQLM, etc.
#      - Default (nn.Linear, etc.)
#   3. First dispatcher that returns non-None wins
#   4. Return the created LoRA module
#
# OUTPUT:
#   LoraLinear(
#       base_layer=nn.Linear(768, 2304),
#       lora_A={"default": nn.Linear(768, 8)},
#       lora_B={"default": nn.Linear(8, 2304)},
#       ...
#   )
#
# =============================================================================
# KEY INSIGHTS:
# =============================================================================
#
# 1. **Priority Order Matters**: First match wins
#    - Custom modules checked first
#    - Quantization-specific before default
#    - Default catches standard layers
#
# 2. **Quantization Support**: Each method has specialized implementation
#    - Different forward passes for quantized weights
#    - Specialized gradient computation
#    - Memory-efficient adaptations
#
# 3. **Extensibility**: Users can add custom modules
#    - Set lora_config._custom_modules
#    - Highest priority - checked first
#    - Allows supporting new layer types
#
# 4. **Type Detection**: Checks isinstance() for each type
#    - Unwraps BaseTunerLayer first
#    - Checks actual layer type
#    - Returns None if no match
#
# 5. **Error Handling**: Raises ValueError if no dispatcher matches
#    - Provides clear error message
#    - Lists supported layer types
#    - Helps user fix configuration
#
# =============================================================================
# SUPPORTED LAYER TYPES:
# =============================================================================
#
# Standard:
# - torch.nn.Linear
# - torch.nn.Embedding
# - torch.nn.Conv1d, Conv2d, Conv3d
# - transformers.pytorch_utils.Conv1D
# - torch.nn.MultiheadAttention
#
# Quantized (bitsandbytes):
# - bnb.nn.Linear8bitLt
# - bnb.nn.Linear4bit
#
# Quantized (other):
# - GPTQ quantized layers
# - AWQ quantized layers
# - AQLM quantized layers
# - HQQ quantized layers
# - TorchAO quantized layers
#
# Parallel:
# - Megatron-LM ColumnParallelLinear
# - Megatron-LM RowParallelLinear
#
# =============================================================================
# See: annotated_code/05_create_and_replace_annotated.py for context
# See: annotated_code/02_lora_forward_pass_annotated.py for forward logic
# =============================================================================
