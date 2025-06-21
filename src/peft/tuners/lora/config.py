# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from torch import nn

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class LoraRuntimeConfig:
    """
    This is the sub-configuration class to store the runtime configurations for the model.

    Args:
        ephemeral_gpu_offload (`bool`):
            Whether to use ephemeral GPU offloading for models partially kept in CPU memory.
    """

    ephemeral_gpu_offload: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use ephemeral GPU offloading for models partially kept in CPU memory. Ephemeral GPU offloading result in "
                "the data involved in intense operations being momentarily copied over to the GPU, and the results copied "
                "back to CPU. There is a momentary VRAM overhead, but operations are generally orders of magnitude faster "
                "compared to performing them on the CPU. This is useful when parts of the model and/or components (such "
                "as adapters) are kept in CPU memory until they are needed. Rather than perform expensive operations on "
                "small data, the data is transferred to the GPU on-demand, the operation(s) performed, and the results "
                "moved back to CPU memory. Currently only affects DoRA initialization."
            )
        },
    )


@dataclass
class LoftQConfig:
    """
    This is the sub-configuration class to store the configuration of a [`LoraModel`].

    Args:
        bits_pattern (`dict`): The mapping from layer names or regexp expression to bits which are different from the
            default bits specified by `bits`. For example, `{model.decoder.layers.0.encoder_attn.k_proj: 2`}.
        bits (`int`): Quantization bits for LoftQ.
        iter (`int`): Alternating iterations for LoftQ.
        fake (`bool`): True: use fp16/fp32; used for first time to save weights. False: use bitsandbytes 4bit linear
            models. weights can't be saved. Recommend to set to True, save the weights and load the saved weights in 4
            bits.
    """

    loftq_bits: int = field(default=4, metadata={"help": "Quantization bits for LoftQ"})
    loftq_iter: int = field(default=1, metadata={"help": "Alternating iterations for LoftQ"})


@dataclass
class EvaConfig:
    """
    This is the sub-configuration class to store the configuration for a data-driven initialization via EVA. EVA was
    introduced in <a href='https://huggingface.co/papers/2410.07170'>Explained Variance Adaptation</a>.

    Args:
        rho (`float`):
            Rho value for EVA redistribution (>= 1.0). The maximum rank for a layer is lora_r * rho. Default is 2.0,
            meaning the maximum rank allowed for a layer is 2r. Increasing rho will allow for a higher degree of
            redistribution of ranks across layers. Some pre-trained models might be more sensitive to a rank
            redistribution. It can therefore be beneficial to try rho=1.0 (no redistribution) if the performance is
            lower than expected.
        tau (`float`):
            Cosine similarity threshold for early stopping. Compares the cosine similarity of right-singular vectors
            between two consecutive SVD steps. If the cosine similarity is above this threshold, the SVD iteration is
            stopped. Default is 0.99.
        use_label_mask (`bool`):
            Use label mask for EVA initialization. This means that positions where labels=label_mask_value are ignored
            for the SVD computation. Setting use_label_mask=True is preferred in most cases and can be especially
            beneficial for multi-turn conversations. The default value is True. Filtering out items based on the label
            mask can sometimes lead to a small batch size and as a result instabilities in the SVD computation. For
            cases where a large share of batch items would be filtered out, set use_label_mask=False.
        label_mask_value (`int`):
            If use_label_mask=True the value to look for to mask out ignored tokens. Default is -100.
        whiten (`bool`): Apply whitening to singular vectors. Default is False.
            Whitening has been shown to be beneficial for EVA in the vision domain.
        adjust_scaling_factors (`bool`):
            Adjust LoRA scaling factors after the rank redistribution. Setting this to True means the scaling factors
            are adjusted so that all LoRA gradients have the same scale regardless of their rank. Default is True.
    """

    rho: float = field(default=2.0, metadata={"help": "Rho value for EVA redistribution"})
    tau: float = field(default=0.99, metadata={"help": "Cosine similarity threshold for early stopping"})
    use_label_mask: bool = field(default=True, metadata={"help": "Use label mask for EVA initialization"})
    label_mask_value: int = field(
        default=-100, metadata={"help": "if use_label_mask=True the value to look for to mask out ignored tokens"}
    )
    whiten: bool = field(default=False, metadata={"help": "Apply whitening to singular vectors"})
    adjust_scaling_factors: bool = field(
        default=True,
        metadata={"help": "Adjust LoRA scaling factors after the rank redistribution"},
    )

    def __post_init__(self):
        if self.rho < 1.0:
            raise ValueError("`rho` must be >= 1.0")
        if self.tau < 0.0 or self.tau > 1.0:
            raise ValueError("`tau` must be between 0.0 and 1.0.")


@dataclass
class CordaConfig:
    """
    This is the sub-configuration class to store the configuration of a [`LoraModel`].

    Args:
        cache_file (`Optional[str]`):
            File to store the SVD cache. The SVD cache is much smaller than the residual model (for example, residual
            model of Llama-3-8b is 15GB, while SVD cache is 1.4GB), but with SVD cache and original model weights,
            residual model weights can be built quickly. If you need to reuse residual model weights with limited
            storage, you can store the SVD cache instead.
        covariance_file (`Optional[str]`):
            File to store the covariance matrix. If you wish to train multiple models with different ranks, but they
            sample from the same dataset, you can store the covariance matrix and reuse it for different ranks. Note
            that covariance file is usually large (comparable to model size), so you will need sufficient storage.
        corda_method (`Literal["ipm", "kpm"]`):
            Method to build adapter. The KPM (Knowledge-Preserved Mode) not only achieves better performance than LoRA
            on fine-tuning tasks, but also mitigates the catastrophic forgetting of pre-trained world knowledge. When
            preserving pre-trained knowledge is not a concern, the IPM (Instruction-Previewed Mode) is favored because
            it can further accelerate convergence and enhance the fine-tuning performance. Defaults to `'ipm'`.
        verbose (`bool`):
            If true, prints the progress of CorDA initialization. Defaults to `False`.
        use_float16_for_covariance (`bool`):
            If true, uses float16 for the covariance matrix. This can reduce the memory usage of the covariance matrix
            by half, but may lead to numerical instability. Defaults to `False`.
        prune_temporary_fields (`bool`):
            If true, temporary fields generated in CorDA preprocessing will be pruned. Defaults to `True`.
    """

    cache_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "File to store the SVD cache. The SVD cache is much smaller than the residual model (for example, "
                "residual model of Llama-3-8b is 15GB, while SVD cache is 1.4GB), but with SVD cache and original model "
                "weights, residual model weights can be built quickly. If you need to reuse residual model weights with "
                "limited storage, you can store the SVD cache instead."
            )
        },
    )
    covariance_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "File to store the covariance matrix. If you wish to train multiple models with different ranks, but "
                "they sample from the same dataset, you can store the covariance matrix and reuse it for different ranks. "
                "Note that covariance file is usually large (comparable to model size), so you will need sufficient storage."
            )
        },
    )
    corda_method: Literal["ipm", "kpm"] = field(
        default="ipm",
        metadata={
            "help": (
                "Method to build adapter. The KPM not only achieves better performance than LoRA on fine-tuning tasks, but "
                "also mitigates the catastrophic forgetting of pre-trained world knowledge. When preserving pre-trained "
                "knowledge is not a concern, the IPM is favored because it can further accelerate convergence and enhance "
                "the fine-tuning performance."
            )
        },
    )
    verbose: bool = field(default=False, metadata={"help": "If true, prints the progress of CorDA initialization."})
    use_float16_for_covariance: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, uses float16 for the covariance matrix. This can reduce the memory usage of the covariance matrix "
                "by half, but may lead to numerical instability."
            )
        },
    )
    prune_temporary_fields: bool = field(
        default=True, metadata={"help": "If true, temporary fields generated in CorDA preprocessing will be pruned."}
    )


@dataclass
class LoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`LoraModel`].

    Args:
        r (`int`):
            Lora attention dimension (the "rank").
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen (if
            the model is a PreTrainedModel, the output layer excluded). If this is not specified, modules will be
            chosen according to the model architecture. If the architecture is not known, an error will be raised -- in
            this case, you should specify the target modules manually.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be performed.
            When passing a list of strings, either an exact match will be performed or it is checked if the name of the
            module ends with any of the passed strings.
        lora_alpha (`int`):
            The alpha parameter for Lora scaling.
        lora_dropout (`float`):
            The dropout probability for Lora layers.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        bias (`str`):
            Bias type for LoRA. Can be 'none', 'all' or 'lora_only'. If 'all' or 'lora_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
        use_rslora (`bool`):
            When set to True, uses [Rank-Stabilized LoRA](https://huggingface.co/papers/2312.03732) which sets the
            adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it was proven to work better. Otherwise, it will
            use the original default value of `lora_alpha/r`.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
        init_lora_weights (`bool` | `Literal["gaussian", "eva", "olora", "pissa", "pissa_niter_[number of iters]", "corda", "loftq", "orthogonal"]`):
            How to initialize the weights of the adapter layers. Passing True (default) results in the default
            initialization from the reference implementation from Microsoft, with the LoRA B weight being set to 0.
            This means that without further training, the LoRA adapter will be a no-op. Setting the initialization to
            False leads to random initialization of LoRA A and B, meaning that LoRA is not a no-op before training;
            this setting is intended for debugging purposes. Passing 'gaussian' results in Gaussian initialization
            scaled by the LoRA rank for linear and layers. Pass `'loftq'` to use LoftQ initialization. Passing `'eva'`
            results in a data-driven initialization of <a href='https://huggingface.co/papers/2410.07170' >Explained
            Variance Adaptation</a>. EVA initializes LoRA based on the SVD of layer input activations and achieves SOTA
            performance due to its ability to adapt to the finetuning data. Pass `'olora'` to use OLoRA initialization.
            Passing `'pissa'` results in the initialization of <a href='https://huggingface.co/papers/2404.02948'
            >Principal Singular values and Singular vectors Adaptation (PiSSA)</a>, which converges more rapidly than
            LoRA and ultimately achieves superior performance. Moreover, PiSSA reduces the quantization error compared
            to QLoRA, leading to further enhancements. Passing `'pissa_niter_[number of iters]'` initiates
            Fast-SVD-based PiSSA initialization, where `[number of iters]` indicates the number of subspace iterations
            to perform FSVD, and must be a nonnegative integer. When `[number of iters]` is set to 16, it can complete
            the initialization of a 7B model within seconds, and the training effect is approximately equivalent to
            using SVD. Passing `'corda'` results in the initialization of <a
            href='https://huggingface.co/papers/2406.05223' >Context-Oriented Decomposition Adaptation</a>, which
            converges even more rapidly than PiSSA in Instruction-Previewed Mode, and preserves world knowledge better
            than LoRA in Knowledge-Preserved Mode. Passing `"orthogonal"` results in LoRA A and B being intialized
            orthogonally; in this, it resembles `"olora"`, but the base weights are left untouched (requires `r` to be
            even, only supported for linear layers for now).
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`. This should target the
            `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`.
        rank_pattern (`dict`):
            The mapping from layer names or regexp expression to ranks which are different from the default rank
            specified by `r`. For example, `{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`.
        alpha_pattern (`dict`):
            The mapping from layer names or regexp expression to alphas which are different from the default alpha
            specified by `lora_alpha`. For example, `{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`.
        megatron_config (`Optional[dict]`):
            The TransformerConfig arguments for Megatron. It is used to create LoRA's parallel linear layer. You can
            get it like this, `core_transformer_config_from_args(get_args())`, these two functions being from Megatron.
            The arguments will be used to initialize the TransformerConfig of Megatron. You need to specify this
            parameter when you want to apply LoRA to the ColumnParallelLinear and RowParallelLinear layers of megatron.
        megatron_core (`Optional[str]`):
            The core module from Megatron to use, defaults to `"megatron.core"`.
        trainable_token_indices (`Optional[Union[List[int], dict[str, List[int]]]]`)
            Lets you specify which token indices to selectively fine-tune without requiring to re-train the whole
            embedding matrix using the `peft.TrainableTokensModel` method. You can specify token indices in two ways.
            Either you specify a list of indices which will then target the model's input embedding layer (or, if not
            found, `embed_tokens`). Alternatively, you can specify a dictionary where the key is the name of the
            embedding module and the values are the list of token indices, e.g. `{'embed_tokens': [0, 1, ...]}`. Note
            that training with FSDP/DeepSpeed might not yet be fully supported with this option enabled.
        loftq_config (`Optional[LoftQConfig]`):
            The configuration of LoftQ. If this is not None, then LoftQ will be used to quantize the backbone weights
            and initialize Lora layers. Also pass `init_lora_weights='loftq'`. Note that you should not pass a
            quantized model in this case, as LoftQ will quantize the model itself.
        eva_config (`Optional[EvaConfig]`):
            The configuration of EVA. At a minimum the dataset argument needs to be set (use the same dataset as for
            finetuning).
        corda_config (`Optional[CordaConfig]`):
            The configuration of CorDA. If this is not None, then CorDA will be used to build the adapter layers. Also
            pass `init_lora_weights='corda'`.
        use_dora (`bool`):
            Enable 'Weight-Decomposed Low-Rank Adaptation' (DoRA). This technique decomposes the updates of the weights
            into two parts, magnitude and direction. Direction is handled by normal LoRA, whereas the magnitude is
            handled by a separate learnable parameter. This can improve the performance of LoRA especially at low
            ranks. Right now, DoRA only supports linear and Conv2D layers. DoRA introduces a bigger overhead than pure
            LoRA, so it is recommended to merge weights for inference. For more information, see
            https://huggingface.co/papers/2402.09353.
        layer_replication (`List[Tuple[int, int]]`):
            Build a new stack of layers by stacking the original model layers according to the ranges specified. This
            allows expanding (or shrinking) the model without duplicating the base model weights. The new layers will
            all have separate LoRA adapters attached to them.
        runtime_config (`LoraRuntimeConfig`):
            Runtime configurations (which are not saved or restored).
        lora_bias (`bool`):
            Defaults to `False`. Whether to enable the bias term for the LoRA B parameter. Typically, this should be
            disabled. The main use case for this is when the LoRA weights were extracted from fully fine-tuned
            parameters so the bias of those parameters can be taken into account.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D "
                "(if the model is a PreTrainedModel, the output layer excluded)."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            ),
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from Lora."},
    )
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: Literal["none", "all", "lora_only"] = field(
        default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"}
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": (
                "When set to True, uses [Rank-Stabilized LoRA](https://huggingface.co/papers/2312.03732)"
                " which sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it"
                " was proven to work better. Otherwise, it will use the original default"
                " value of `lora_alpha/r`."
            )
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: (
        bool
        | Literal["gaussian", "eva", "olora", "pissa", "pissa_niter_[number of iters]", "corda", "loftq", "orthogonal"]
    ) = field(
        default=True,
        metadata={
            "help": (
                "How to initialize the weights of the LoRA layers. "
                "Passing True (default) results in the default initialization from the reference implementation from "
                "Microsoft, with the LoRA B weight being set to 0. This means that without further training, the LoRA "
                "adapter will be a no-op. "
                "Setting the initialization to False leads to random initialization of LoRA A and B, meaning that LoRA "
                "is not a no-op before training; this setting is intended for debugging purposes. "
                "Passing `'gaussian'` results in Gaussian initialization scaled by the LoRA rank for linear and layers. "
                "Passing `'eva'` results in a data-driven initialization of Explained Variance Adaptation. "
                "Passing `'olora'` results in OLoRA initialization. "
                "Passing `'pissa'` results in PiSSA initialization. "
                "Passing `'pissa_niter_[number of iters]'` initiates Fast-SVD-based PiSSA initialization, where "
                "[number of iters] indicates the number of subspace iterations to perform fsvd, and must be a "
                "nonnegative integer. "
                "Passing `'corda'` results in CorDA initialization. "
                "Pass `'loftq'` to use LoftQ initialization. "
                "Pass `'orthogonal'` for orthogonal initialization of LoRA A and B."
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index. "
            "This only works when target_modules is a list of str."
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
            "This only works when target_modules is a list of str. This should target the `nn.ModuleList` of the "
            "model, which is often called `'layers'` or `'h'`."
        },
    )
    rank_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. "
                "For example, `{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`."
            )
        },
    )
    alpha_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `lora_alpha`. "
                "For example, `{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`."
            )
        },
    )
    megatron_config: Optional[dict] = field(
        default=None,
        metadata={
            "help": (
                "The TransformerConfig from Megatron. It is used to create LoRA's parallel linear layer."
                "You can get it like this, `core_transformer_config_from_args(get_args())`, "
                "these two functions being from Megatron."
                "You need to specify this parameter when you want to apply LoRA to the ColumnParallelLinear and "
                "RowParallelLinear layers of megatron."
                "It should be noted that we may not be able to use the `save_pretrained` and `from_pretrained` "
                "functions, because TransformerConfig may not necessarily be serialized."
                "But when using megatron, we can use `get_peft_model_state_dict` function and "
                "megatron's framework, they can also save and load models and configurations."
            )
        },
    )
    megatron_core: Optional[str] = field(
        default="megatron.core",
        metadata={
            "help": (
                "The core module from Megatron, it is used to create LoRA's parallel linear layer. "
                "It only needs to be passed in when you need to use your own modified megatron core module. "
                "Otherwise, it will use the default value `megatron.core`. "
            )
        },
    )
    trainable_token_indices: Optional[Union[list[int], dict[str, list[int]]]] = field(
        default=None,
        metadata={
            "help": (
                "Lets you specify which token indices to selectively fine-tune without requiring to re-train the "
                "whole embedding matrix using the `peft.TrainableTokensModel` method. You can specify token indices "
                "in two ways. Either you specify a list of indices which will then target the model's input embedding "
                "layer (or, if not found, `embed_tokens`). Alternatively, you can specify a dictionary where the key "
                "is the name of the embedding module and the values are the list of token indices, e.g. "
                "`{'embed_tokens': [0, 1, ...]}`. "
                "Note that training with FSDP/DeepSpeed might not yet be fully supported with this option enabled. "
                "Also note that models using weight-tying are currently not supported."
            )
        },
    )
    # dict type is used when loading config.json
    loftq_config: Union[LoftQConfig, dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The configuration of LoftQ. If this is passed, then LoftQ will be used to quantize the backbone "
                "weights and initialize Lora layers. Also set `init_lora_weights='loftq'` in this case."
            )
        },
    )
    eva_config: Optional[EvaConfig] = field(
        default=None,
        metadata={
            "help": (
                "The configuration of EVA. If this is passed, then EVA will be used to initialize the LoRA layers. "
                "Also set `init_lora_weights='eva'` in this case. "
            )
        },
    )
    corda_config: Optional[CordaConfig] = field(
        default=None,
        metadata={
            "help": (
                "The configuration of CorDA. If this is passed, then CorDA will be used to build the adapter layers. "
                "Also set `init_lora_weights='corda'` in this case."
            )
        },
    )
    use_dora: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable <a href='https://huggingface.co/papers/2402.09353'>'Weight-Decomposed Low-Rank Adaptation' (DoRA)</a>. This technique decomposes the updates of the "
                "weights into two parts, magnitude and direction. Direction is handled by normal LoRA, whereas the "
                "magnitude is handled by a separate learnable parameter. This can improve the performance of LoRA, "
                "especially at low ranks. Right now, DoRA only supports linear and Conv2D layers. DoRA introduces a bigger"
                "overhead than pure LoRA, so it is recommended to merge weights for inference."
            )
        },
    )
    use_qalora: bool = field(
        default=False,
        metadata={
            "help": (
                "It is only implemented in GPTQ for now. Enable <a href='https://huggingface.co/papers/2309.14717'>Quantization-Aware Low-Rank Adaptation (QALoRA)</a>."
                "This technique combines quantization-aware training "
                "with LoRA to improve performance for quantized models. This can improve the performance of LoRA, "
                "especially at low ranks. Right now, QALoRA only supports linear layers."
            )
        },
    )
    qalora_group_size: int = field(
        default=16,
        metadata={
            "help": (
                "Group size parameter for QALoRA pooling, controlling the dimension reduction factor. "
                "Input dimensions are pooled into groups of this size, reducing the computational cost. "
                "Higher values provide more compression but may reduce model quality. "
                "This parameter determines how many original features are averaged together to create "
                "one pooled feature. Only used when `use_qalora=True`."
            )
        },
    )
    # Enables replicating layers in a model to expand it to a larger model.
    layer_replication: Optional[list[tuple[int, int]]] = field(
        default=None,
        metadata={
            "help": (
                "This enables using LoRA to effectively expand a transformer model to a larger size by repeating some layers. "
                "The transformation handles models (currently Llama, Bert or Falcon compatible architectures) with "
                "a module list in the model which it modifies to expand the number of modules. "
                "Base weights are shared so the memory usage is close to the original model. The intended use is these base weights "
                "remain fixed during finetuning but each layer has a separate LoRA adapter so the layers can be specialed via "
                "the adapter layers fit during fine tuning."
                "The format is a list of [start, end) pairs which specify the layer ranges to stack. For example:\n"
                "   Original model has 5 layers labelled by their position in the model: `[0, 1, 2, 3, 4]`\n"
                "   layer_replication: `[[0, 4], [2, 5]]`\n"
                "   Final model will have this arrangement of original layers: `[0, 1, 2, 3, 2, 3, 4]`\n"
                "This format is based on what is used for pass-through merges in mergekit. It makes it simple to select sequential "
                "ranges of a model and stack them while reusing layers at either end of each sequence."
            )
        },
    )
    runtime_config: LoraRuntimeConfig = field(
        default_factory=LoraRuntimeConfig, metadata={"help": "Runtime configurations"}
    )
    lora_bias: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable the bias term for the LoRA B parameter. Typically, this should be disabled. The "
                "main use case for this is when the LoRA weights were extracted from fully fine-tuned parameters so "
                "the bias of those parameters can be taken into account."
            )
        },
    )

    def to_dict(self):
        """
        Returns the configuration for your adapter model as a dictionary. Removes runtime configurations.
        """
        rv = super().to_dict()
        rv.pop("runtime_config")
        return rv

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.LORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )

        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")

        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")

        if self.use_dora and self.megatron_config:
            raise ValueError("DoRA does not support megatron_core, please set `use_dora=False`.")

        # handle init_lora_weights and loftq_config
        if self.init_lora_weights == "loftq":
            import importlib

            if not importlib.util.find_spec("scipy"):
                raise ImportError("The required package 'scipy' is not installed. Please install it to continue.")
            if not self.loftq_config:
                raise ValueError("`loftq_config` must be specified when `init_lora_weights` is 'loftq'.")
            if not isinstance(self.loftq_config, dict):
                # convert loftq_config to dict
                self.loftq_config = vars(self.loftq_config)
        elif self.loftq_config:
            self.loftq_config = {}
            warnings.warn("`loftq_config` specified but will be ignored when `init_lora_weights` is not 'loftq'.")

        elif self.init_lora_weights == "eva" and self.eva_config is None:
            warnings.warn("`init_lora_weights` is 'eva' but `eva_config` is not specified. Using default EVA config.")
            self.eva_config = EvaConfig()
        elif self.init_lora_weights != "eva" and self.eva_config is not None:
            warnings.warn("`eva_config` specified but will be ignored when `init_lora_weights` is not 'eva'.")

        elif self.init_lora_weights == "corda" and self.corda_config is None:
            warnings.warn(
                "`init_lora_weights` is 'corda' but `corda_config` is not specified. Using default CorDA config."
            )
            self.corda_config = CordaConfig()
        elif self.init_lora_weights != "corda" and self.corda_config is not None:
            warnings.warn("`corda_config` specified but will be ignored when `init_lora_weights` is not 'corda'.")

        if self.lora_bias:
            if self.init_lora_weights not in (True, False):
                raise ValueError(
                    f"The argument lora_bias=True is only supported with init_lora_weights=True or False, got "
                    f"init_lora_weights={self.init_lora_weights} instead."
                )
            if self.use_dora:
                raise ValueError("The argument lora_bias=True is not supported for DoRA, please pass use_dora=False")

        # Using post training conversion of modified base weights to restore their initial values PiSSA/CorDA/OLoRA cannot
        # be correctly done when using rslora + rank_pattern/alpha_pattern. We can't really know if the user intends
        # this when they'll eventually call save_pretrained (i.e. if they'll pass
        # path_initial_model_for_weight_conversionl). Therefore, we only warn but don't raise an error here.
        if (
            self.use_rslora
            and (self.rank_pattern or self.alpha_pattern)
            and (
                (isinstance(self.init_lora_weights, str) and (self.init_lora_weights.startswith("pissa")))
                or (self.init_lora_weights == "olora")
                or (self.init_lora_weights == "corda")
            )
        ):
            msg = (
                "Using Rank-Stabilized LoRA with rank_pattern/alpha_pattern and post-training conversion of modified "
                "base weights PiSSA/CorDA/OLoRA means that you won't be able to pass "
                "`path_initial_model_for_weight_conversion` to `save_pretrained` to restore the initial values of the "
                "base weights; if you intend to do this, please ensure not to use rslora or rank_pattern/alpha_pattern."
            )
            warnings.warn(msg)

        self._custom_modules: Optional[dict[type[nn.Module], type[nn.Module]]] = None

    def _register_custom_module(self, mapping: dict[type[nn.Module], type[nn.Module]]) -> None:
        """
        Experimental API to support providing custom LoRA layers.

        This API is subject to change, you should carefully read the docs before deciding to use it:

        https://huggingface.co/docs/peft/developer_guides/custom_models

        To register custom LoRA module types, call this method with a `mapping` argument that is a dict that maps from
        the target layer type to the custom LoRA layer type. The dict can contain multiple items if you wish to target
        multiple layer types. The target layer type can be any nn.Module that we currently don't support in PEFT,
        whether that is an official PyTorch layer type or a custom layer type. The custom LoRA module class has to be
        implemented by the user and follow the PEFT conventions for LoRA layers.

        """
        if self._custom_modules is None:
            self._custom_modules = {}
        self._custom_modules.update(mapping)
