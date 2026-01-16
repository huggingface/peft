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
):
    """
    Intruder dimension mitigation based on https://huggingface.co/papers/2410.21228
    ("LoRA vs Full Fine-tuning: An Illusion of Equivalence").

    This method can recover previous knowledge (i.e. mitigate forgetting) by post-processing already trained
    low-rank adapters.

    Parameters:
        top_k (default: 10)
            Consider the top-k dimensions for intruder detection. The larger the value, the more dimensions will
            be considered for intruder detection analysis (and the more false-postiives there can be).
            Operates on the cosine similarity between base weights and adapter weights roughly sorted by influence
            of dimension (determined by singular value decomposition), so a top-k of 10 will look at the 10 most
            'important' dimensions.

        threshold_epsilon (default: 0.5)
            Threshold value when to consider a cosine similarity between base weight and adapter weight as intruder.
            According to the paper, intruder dimensions show near-zero absolute cosine similarity with pre-trained
            singular vectors. The lower this value, the less potential intruder dimensions are identified. The
            higher the value, the more potential false-positives are considered as intruders.

        mitigation_lambda (default: 0.75)
            The relative portion of the intruder dimensions that is subtracted from the adapter's delta weight.
            The higher the value the more of the intruder dimension is subtracted but the more information is
            lost. Refer to Figure 8 in the paper for a trade-off analysis.
    """
    # TODO check if peft method is supported

    peft_model.add_adapter(new_adapter_name, peft_model.peft_config[old_adapter_name])

    # apply mitigation on the old adapter's weights and move them to the new adapter's weights
    for _name, layer in peft_model.named_modules():
        if not isinstance(layer, LoraLayer):
            continue

        W = layer.get_base_layer().weight.data
        dW = layer.get_delta_weight(old_adapter_name)
        W_merged = W + dW

        cast_to_fp32 = W.device.type == "cpu" and W.dtype in (torch.float16, torch.bfloat16)

        if cast_to_fp32:
            W_dtype = W.dtype
            W = W.float()

        # compare base weights and adapter weights using cosine similarity.
        # based on this similarit we can find intruder dimensions using threshold_epsilon
        # on the top_k dimensions
        U_base, _S_base, _V_base = torch.linalg.svd(W, full_matrices=False)
        U_merged, S_merged, V_merged = torch.linalg.svd(W_merged, full_matrices=False)

        cos_sim = (U_merged.T @ U_base).abs().max(dim=1).values
        intruder_idcs = torch.where(cos_sim[:top_k] < threshold_epsilon)[0].tolist()

        if not intruder_idcs:
            continue

        # the paper computes the intruder dimensions that are subtracted on (W + dW)
        # so we do the same. experiments showed that this achieves better knowledge
        # recovery than on dW alone.
        B_intruder = (U_merged[:, intruder_idcs] @ torch.diag(S_merged)[intruder_idcs, :].sqrt())
        A_intruder = (torch.diag(S_merged)[:, intruder_idcs]).sqrt() @ V_merged[intruder_idcs, :]

        # apply mitigation and recover dW = (B@A).
        # (W+dW+mitigation)-W = dW+mitigation, so we can convert dW back to A/B using SVD
        # since we know the effective rank from the adapter config.
        #
        # note that we also remove the scaling from dW which is applied in get_delta_weight() since
        # it impacts mitigation performance both in task accuracy and forgetting.
        W_merged += (mitigation_lambda - 1) * (B_intruder @ A_intruder)
        W_merged -= W
        W_merged /= layer.scaling[old_adapter_name]

        U_dW, S_dW, V_dW = torch.linalg.svd(W_merged, full_matrices=False)

        effective_rank = layer.lora_A[old_adapter_name].out_features
        B_new = U_dW[:, :effective_rank] * S_dW[:effective_rank]
        A_new = V_dW[:effective_rank]

        layer.lora_B[new_adapter_name].weight.data = B_new
        layer.lora_A[new_adapter_name].weight.data = A_new

        # cast W back from float32 to whatever it was before to save memory in the long run
        if cast_to_fp32:
            W = W.to(W_dtype)

    peft_model.set_adapter(new_adapter_name)
