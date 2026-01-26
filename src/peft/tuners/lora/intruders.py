# Copyright 2025-present the HuggingFace Inc. team.
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

import torch

from .layer import LoraLayer


def reduce_intruder_dimension(
    peft_model,
    old_adapter_name="default",
    new_adapter_name="intruder_reduced",
    top_k=10,
    threshold_epsilon=0.5,
    mitigation_lambda=0.75,
    logging_sink=print,
):
    """
    Intruder dimension mitigation based on https://huggingface.co/papers/2410.21228 ("LoRA vs Full Fine-tuning: An
    Illusion of Equivalence").

    This method can recover previous knowledge (i.e. mitigate forgetting) by post-processing already trained low-rank
    adapters. This comes at a cost of task accuracy - tuning the `migration_lambda` value can be used to trade between
    these two factors.

    After mitigation is done there will be a new adapter with the name set in `new_adapter_name` which is also set to
    be the currently active adapter. Inference on the mitigated model will therefore use the modified adapter. To
    switch back to the original adapter you can use `peft_model.set_adapter(<old_adapter_name>)`.

    Currently only LoRA is supported as it is not clear whether this method generalizes to other delta-weight methods.

    Parameters:
        peft_model:
            The PEFT model with a loaded LoRA adapter with the name provided in `old_adapter_name`. Currently mixed
            models are not supported.

        top_k (default: 10)
            Consider the top-k dimensions for intruder detection. The larger the value, the more dimensions will be
            considered for intruder detection analysis (and the more false-postiives there can be). Operates on the
            cosine similarity between base weights and adapter weights roughly sorted by influence of dimension
            (determined by singular value decomposition), so a top-k of 10 will look at the 10 most 'important'
            dimensions.

        threshold_epsilon (default: 0.5)
            Threshold value when to consider a cosine similarity between base weight and adapter weight as intruder.
            According to the paper, intruder dimensions show near-zero absolute cosine similarity with pre-trained
            singular vectors. The lower this value, the less potential intruder dimensions are identified. The higher
            the value, the more potential false-positives are considered as intruders.

        mitigation_lambda (default: 0.75)
            The relative portion of the intruder dimensions that is subtracted from the adapter's delta weight. The
            higher the value the more of the intruder dimension is subtracted but the more information is lost. Refer
            to Figure 8 in the paper for a trade-off analysis.

        logging_sink (default: print)
            Function that prints information about the mitigation process. Set to None if you don't want any output.
    """
    # Note that this function currently doesn't support `compile_kwargs` similar to the LoRA conversion tooling
    # since there's was no clear way how `torch.compile` can be used to improve performance at the time of
    # implementation. See discussion: https://github.com/huggingface/peft/pull/2999#discussion_r2717989613

    def no_logging_sink(*args, **kwargs):
        pass

    if logging_sink is None:
        logging_sink = no_logging_sink

    if peft_model.peft_type != "LORA":
        raise ValueError("The provided model is not using LoRA and is therefore not supported.")

    peft_model.add_adapter(new_adapter_name, peft_model.peft_config[old_adapter_name])

    # apply mitigation on the old adapter's weights and move them to the new adapter's weights
    for layer_name, layer in peft_model.named_modules():
        if not isinstance(layer, LoraLayer):
            continue

        W = layer.get_base_layer().weight.data
        dW = layer.get_delta_weight(old_adapter_name)
        W_merged = W + dW
        is_embedding = old_adapter_name not in layer.lora_B

        cast_to_fp32 = W.dtype in (torch.float16, torch.bfloat16)

        if cast_to_fp32:
            W_dtype = W.dtype
            W = W.float()

        # compare base weights and adapter weights using cosine similarity.
        # based on this similarity we can find intruder dimensions using threshold_epsilon
        # on the top_k dimensions
        U_base, _S_base, _V_base = torch.linalg.svd(W, full_matrices=False)
        U_merged, S_merged, V_merged = torch.linalg.svd(W_merged, full_matrices=False)

        cos_sim = (U_merged.T @ U_base).abs().max(dim=1).values
        intruder_idcs = torch.where(cos_sim[:top_k] < threshold_epsilon)[0].tolist()

        if not intruder_idcs:
            logging_sink(f"{layer_name}: No intruders")

            # we're not modifying the weights since there are no intruders but we make sure to copy the
            # adapter weights unmodified to the new adapter, otherwise these weights will be
            # initialized randomly
            if is_embedding:
                layer.lora_embedding_B[new_adapter_name].data = layer.lora_embedding_B[old_adapter_name].data.clone()
                layer.lora_embedding_A[new_adapter_name].data = layer.lora_embedding_A[old_adapter_name].data.clone()
            else:
                layer.lora_B[new_adapter_name].weight.data = layer.lora_B[old_adapter_name].weight.data.clone()
                layer.lora_A[new_adapter_name].weight.data = layer.lora_A[old_adapter_name].weight.data.clone()
            continue
        else:
            logging_sink(f"{layer_name}: Intruders: {len(intruder_idcs)}")

        # the paper computes the intruder dimensions that are subtracted on (W + dW), so we do the same. experiments
        # showed that this achieves better knowledge recovery than on dW alone.
        B_intruder = U_merged[:, intruder_idcs] @ torch.diag(S_merged)[intruder_idcs, :].sqrt()
        A_intruder = (torch.diag(S_merged)[:, intruder_idcs]).sqrt() @ V_merged[intruder_idcs, :]

        # apply mitigation and recover dW = (B@A).
        # (W+dW+mitigation)-W = dW+mitigation, so we can convert dW back to A/B using SVD
        # since we know the effective rank from the adapter config.
        #
        # note that we also remove the scaling from dW which is applied in get_delta_weight() since
        # it impacts mitigation performance both in task accuracy and forgetting.
        W_mitigated = W_merged + (mitigation_lambda - 1) * (B_intruder @ A_intruder)
        dW_mitigated = W_mitigated - W
        dW_mitigated /= layer.scaling[old_adapter_name]

        U_dW, S_dW, V_dW = torch.linalg.svd(dW_mitigated, full_matrices=False)

        # Note: share scaling by S equally between B and A to avoid one matrix having a significantly
        # different norm and avoid possibly weird training dynamics.
        effective_rank = layer.r[old_adapter_name]
        B_new = U_dW[:, :effective_rank] @ torch.diag(S_dW[:effective_rank]).sqrt()
        A_new = torch.diag(S_dW[:effective_rank]).sqrt() @ V_dW[:effective_rank]

        if is_embedding:
            layer.lora_embedding_B[new_adapter_name].data = B_new
            layer.lora_embedding_A[new_adapter_name].data = A_new
        else:
            layer.lora_B[new_adapter_name].weight.data = B_new
            layer.lora_A[new_adapter_name].weight.data = A_new

        # cast W back from float32 to whatever it was before to save memory in the long run
        if cast_to_fp32:
            W = W.to(W_dtype)

    logging_sink(f"Enabling new adapter {new_adapter_name}")
    peft_model.set_adapter(new_adapter_name)
