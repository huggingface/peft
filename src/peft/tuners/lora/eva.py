# Copyright 2024-present the HuggingFace Inc. team.
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

import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping
from itertools import cycle
from typing import Dict, Optional, Union

import torch
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import check_target_module_exists
from peft.utils.incremental_pca import IncrementalPCA

from .config import EvaConfig, LoraConfig
from .layer import Embedding, LoraLayer, _ConvNd


UNSUPPORTED_LORA_MODULES = [Embedding, _ConvNd]


class SVDHook:
    """
    A forward hook for calculating incremental SVD/PCA on layer input activations. The hook is designed to be
    registered to a PyTorch module using the `register_forward_hook` method.

    This hook performs a step of incremental Singular Value Decomposition (SVD) on the input activations of a specified
    layer during the forward pass of a neural network. The hook also tracks convergence of the computed components
    using cosine similarity between the current and previous components.

    Args:
        name (str): Name of the layer to which this hook is attached.
        n_components (int): Number of principal components to compute.
        sim_thresh (Union[float, torch.Tensor]): Similarity threshold for convergence.
    """

    def __init__(
        self,
        name: str,
        n_components: int,
        sim_thresh: Union[float, torch.Tensor],
        prepare_activations_fn: Optional[callable] = None,
    ):
        self.name = name
        self.n_components = n_components
        self.sim_thresh = sim_thresh
        if prepare_activations_fn is None:
            self.prepare_activations_fn = self._prepare_activations_fn_default
        else:
            self.prepare_activations_fn = prepare_activations_fn

        if isinstance(sim_thresh, torch.Tensor) and len(sim_thresh.shape) > 0:
            check1 = sim_thresh.size(0) == n_components or sim_thresh.size(0) == 1
            check2 = len(sim_thresh.shape) == 1
            if not (check1 and check2):
                raise ValueError(
                    "if sim_thresh is a tensor with more than 0 dimensions it must have shape (n_components,) or (1,)"
                )

        self.svd = IncrementalPCA(n_components=n_components, copy=True, lowrank=True)

        self.indices = None
        self.converged = torch.zeros((n_components,), dtype=torch.bool)

    def __call__(self, model, input, output):
        previous_components = None
        if hasattr(self.svd, "components_"):
            previous_components = self.svd.components_.clone().detach()

        states = self.prepare_activations_fn(input).detach()

        # merging all but last dimension
        if self.indices is not None:
            states = states[self.indices]

        # check if batch sizes is more than the number of components
        if states.size(0) < self.n_components:
            print(f"skipping SVD for {self.name} because there are less than {self.n_components} examples")
            return

        self.svd.partial_fit(states.to(torch.float32))

        # add if statement to check if we are in the first step where previous_components is None
        if previous_components is None:
            return
        components = self.svd.components_
        if len(components.shape) == 1:
            components = components.reshape(1, -1)
            previous_components = previous_components.reshape(1, -1)
        # consider as converged if enough components have converged via cossim
        sim = torch.nn.functional.cosine_similarity(components, previous_components)
        self.converged = sim >= self.sim_thresh

    @staticmethod
    def _prepare_activations_fn_default(activations) -> torch.Tensor:
        if isinstance(activations, torch.Tensor):
            return activations
        elif isinstance(activations, (tuple, list)):
            return activations[0]
        else:
            raise ValueError(
                f"unsupported input type for prepare_activations_fn: {type(activations)}, "
                "please provide a custom prepare_activations_fn"
            )


# This is used to determine if two input activations are equal. For such cases, SVD
# needs to be done for only for one of the equal inputs.
class HashHook:
    """
    A forward hook for hashing layer input activations. The hook is designed to be registered to a PyTorch module using
    the `register_forward_hook` method.

    This hook hashes the input activations of a specified layer during the forward pass of a neural network and stores
    the hash values for later analysis or comparison.

    Args:
        name (str): Name of the layer to which this hook is attached. hashed_inputs (list): List of hashed input
            activations.
    """

    def __init__(self, name: str):
        self.name = name
        self.hashed_inputs = []

    @staticmethod
    def hash_fn(tensor):
        return hash(tuple(tensor.view(-1).tolist()))

    def __call__(self, model, input, output):
        try:
            x = input.detach().cpu()
        except AttributeError:
            x = input[0].detach().cpu()
        self.hashed_inputs.append(self.hash_fn(x))


def find_equal_values(dictionary: dict) -> dict:
    """
    Find keys in a dictionary that have the same value.

    This function takes a dictionary and returns a new dictionary containing keys that have the same value. The keys in
    the output dictionary are the values from the input dictionary, and the values are lists of keys that share the
    same value.
    """
    value_dict = defaultdict(list)
    for k, v in dictionary.items():
        value_dict[v].append(k)
    return {k: v for k, v in value_dict.items() if len(v) > 1}


def recursive_apply(module: torch.nn.Module, device: Union[str, torch.device]):
    """
    Recursively apply a function to the module and its submodules.

    If the module is a LoraLayer, the base layer is moved to the specified device. If the module has any LoraLayer
    submodules, the function is applied recursively to each submodule. Otherwise, the module is moved to the specified
    device.
    """
    if isinstance(module, LoraLayer):
        module.base_layer.to(device)
        return
    if any(isinstance(submodule, LoraLayer) for submodule in module.modules()):
        for child in module.children():
            recursive_apply(child, device)
    else:
        module.to(device)


def get_device_with_meta_params(model: torch.nn.Module) -> torch.device:
    """
    Get the device of the model's parameters. Useful if some parameters are on meta device.
    """
    devices = list({p.device for p in model.parameters() if p.device.type != "meta"})
    if len(devices) > 1:
        raise ValueError("model has multiple devices")
    return devices[0]


def move_inputs_to_device(inputs, device: Union[str, torch.device]):
    """
    Move the inputs to the specified device. Adapted from hf.Trainer.
    """
    if isinstance(inputs, Mapping):
        return type(inputs)({k: move_inputs_to_device(v, device) for k, v in inputs.items()})
    elif isinstance(inputs, (tuple, list)):
        return type(inputs)(move_inputs_to_device(v, device) for v in inputs)
    elif isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    else:
        raise ValueError(f"unsupported input type: {type(inputs)}")


def get_indices_fn_causal_lm(inputs: dict, config: LoraConfig):
    """
    if not all items in the input should be used for SVD, this function can be used to get the indices of the items
    that should be used
    """
    mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])).bool()
    if config.eva_config.use_label_mask and hasattr(inputs, "labels"):
        mask = torch.logical_and(mask, inputs["labels"] != config.eva_config.label_mask_value)
    return mask.nonzero()


def forward_fn_dict(model, inputs):
    return model(**inputs)


@torch.no_grad()
def get_eva_state_dict(
    model: torch.nn.Module,
    peft_config: LoraConfig,
    dataloader: torch.utils.data.DataLoader,
    forward_fn: Optional[callable] = forward_fn_dict,
    get_indices_fn: Optional[callable] = get_indices_fn_causal_lm,
    prepare_activations_fn: Union[callable, Dict[str, callable], None] = None,
    show_progress_bar: bool = True,
) -> dict:
    """
    Compute the SVD for each layer in the model.

    This function computes the Singular Value Decomposition (SVD) for each layer in the model. It uses the incremental
    PCA method to compute the SVD components. The function also checks for convergence of the computed components using
    cosine similarity. The rank distribution for each layer is determined based on the explained variance ratio.

    Args:
        model (torch.nn.Module): The model to compute the SVD for.
        peft_config (LoraConfig): The configuration for the LoRA layers.
        dataloader (torch.utils.data.DataLoader): The dataloader to use for the forward pass.
        forward_fn (callable):
            The forward function to use for the forward pass. `model(**inputs)` is used if forward_fn is not provided.
        get_indices_fn (Optional[callable]):
            The function to use if not all positions in the input tensor should be used for SVD (e.g. for causal
            language modeling). Can be set to None if all positions should be used.
            `peft.tuners.lora.eva.get_indices_fn_causal_lm` is used by default. Should always return a tensor of shape
            (n_indices, layer_input.ndim-1).
        prepare_activations_fn: (Union[callable, Dict[str, callable], None])
            If a layer receives multiple inputs as a list, per default the first input is used. This function can be
            used to modify this behaviour. Accepts a dictionary with layer names as keys. Default logic:
            ```
            try:
                svd_input = activations.detach()
            except AttributeError:
                svd_input = activations[0].detach()
            ```
        show_progress_bar (bool): Whether to show a progress bar. Default is True.

    Returns:
        eva_state_dict (dict): The state dictionary containing the SVD components for each layer.
    """

    # Computes the rank distribution for each layer based on the explained variance ratio.
    def _get_rank_distribution(hooks, layer_hook_map, equal_inputs_map, rank_budget, max_components):
        exp_vars = {k: h[0].svd.explained_variance_ratio_[:max_components] for k, h in hooks.items()}
        keys, values = zip(*[(k, c) for k, name in layer_hook_map.items() for c in exp_vars[name]])
        idx = torch.stack(values).argsort(descending=True)
        counts = Counter([keys[i] for i in idx[:rank_budget]])
        counts = {k: counts.get(k, 0) for k in layer_hook_map.keys()}  # add layers with 0 rank
        for k, k_hook in equal_inputs_map.items():
            # ensure hook layers have the highest rank if they are equal to another layer
            rank, rank_hook = counts[k], counts[k_hook]
            if rank_hook >= rank:
                continue
            counts[k_hook], counts[k] = rank, rank_hook
        return counts

    # set function to check if modules should be added to hooks
    if hasattr(model, "peft_config"):

        def _check_fn(name, module):
            return hasattr(module, "base_layer") and module not in UNSUPPORTED_LORA_MODULES

    else:

        def _check_fn(name, module):
            is_target_module = True
            if peft_config.target_modules is not None:
                is_target_module = check_target_module_exists(peft_config, name)
            # Conv1D for GPT2 support
            return isinstance(module, (torch.nn.Linear, Conv1D)) and is_target_module

    training = model.training
    device = get_device_with_meta_params(model)
    model.eval()

    hooks = {}
    for name, module in model.named_modules():
        # currently only linear layers are supported
        if not _check_fn(name, module):
            continue
        hook = HashHook(name)
        handle = module.register_forward_hook(hook)
        hooks[name] = (hook, handle)
    rank_budget = peft_config.r * len(hooks)
    max_components = round(peft_config.r * peft_config.eva_config.rho)

    # forward for one batch to check which layer inputs are equal to avoid unneeded svd calculations
    inputs = next(iter(dataloader))
    forward_fn(model, move_inputs_to_device(inputs, device))
    hash_dict = {k: h[0].hashed_inputs[0] for k, h in hooks.items()}
    # equal input maps groups layers which recieve the same input. One layer is defined as the key and recieves an svd hook. For the remaining layers the svd results can be skipped.
    equal_inputs_map = {vv: v[0] for v in find_equal_values(hash_dict).values() for vv in v[1:]}

    # initialize svd hooks
    unmapped = []
    for name in list(hooks.keys()):
        _, handle = hooks.pop(name)
        handle.remove()
        if name in equal_inputs_map:
            continue
        if isinstance(prepare_activations_fn, Mapping):
            try:
                fn = prepare_activations_fn[name]
            except KeyError:
                unmapped.append(name)
                fn = None
        else:
            fn = prepare_activations_fn
        hook = SVDHook(name, max_components, peft_config.eva_config.tau, fn)
        module = model.get_submodule(name)
        handle = module.register_forward_hook(hook)
        hooks[name] = (hook, handle)  # adding the old handle here so we dont get errors in the first forward pass
    layer_hook_map = {**dict(zip(hooks.keys(), hooks.keys())), **equal_inputs_map}

    if unmapped:
        raise ValueError(f"if prepare_activations_fn is a mapping it must contain module names {unmapped}")

    # start svd calculation
    if show_progress_bar:
        pbar = tqdm(iter(cycle(dataloader)), position=0, leave=False)
    else:
        pbar = iter(cycle(dataloader))
    convergence_dict = {k: False for k in hooks.keys()}
    rank_dist = {k: max_components for k in layer_hook_map.keys()}
    for inputs in pbar:
        indices = None
        if get_indices_fn is not None:
            indices = get_indices_fn(inputs, peft_config)

        for name in list(hooks.keys()):
            hook, handle = hooks[name]
            # check if all components that are needed for the rank distribution have converged
            converged = torch.all(hook.converged[: rank_dist[name]])
            # if a layer has switched from not converged to converged in the current step
            if (not convergence_dict[name]) and converged and handle:
                handle.remove()
                handle = None
                convergence_dict[name] = True
                continue
            # if a layer has switched from converged to not converged in the current step
            elif convergence_dict[name] and not converged:
                module = model.get_submodule(name)
                handle = module.register_forward_hook(hook)
                convergence_dict[name] = False
            hook.indices = indices.T.unbind()
            hooks[name] = (hook, handle)

        if show_progress_bar:
            layer_converged = list(convergence_dict.values()) + [
                convergence_dict[v] for v in equal_inputs_map.values()
            ]
            pbar.set_description(f"{sum(layer_converged)}/{len(layer_converged)} layers have converged")

        if all(convergence_dict.values()):
            break

        forward_fn(model, move_inputs_to_device(inputs, device))

        # in case some hooks have to skip the svd calculation because the number of tokens is less than the number of components
        if not all(hasattr(h[0].svd, "components_") for h in hooks.values()):
            continue

        rank_dist = _get_rank_distribution(hooks, layer_hook_map, equal_inputs_map, rank_budget, max_components)

    # check all svd hooks have been removed
    remaining_svd_hooks = {
        n for n, m in model.named_modules() for v in m._forward_hooks.values() if isinstance(v, SVDHook)
    }
    if len(remaining_svd_hooks) > 0:
        raise ValueError(
            f"Found active SVD hooks that weren't properly removed: {remaining_svd_hooks}. "
            "Please report this issue at https://github.com/huggingface/peft/issues"
        )

    eva_state_dict = {}
    for name, rank in rank_dist.items():
        if rank == 0:
            continue
        hook = hooks[layer_hook_map[name]][0]
        if not torch.all(hook.converged[:rank]):
            raise ValueError(
                f"Layer {name} has not converged but was assigned rank {rank}. "
                "Please report this issue at https://github.com/huggingface/peft/issues"
            )
        u = hook.svd.components_[:rank]
        if peft_config.eva_config.whiten:
            u /= hook.svd.singular_values_[:rank].sqrt().reshape(-1, 1)
        eva_state_dict[name] = u

    # objects are torch tensors on the model device
    eva_state_dict = {k: v.to(device) for k, v in eva_state_dict.items()}

    # restore model state
    model.train(training)

    return eva_state_dict


@torch.no_grad()
def initialize_lora_eva_weights(
    model,
    peft_config,
    dataloader,
    forward_fn: Optional[callable] = forward_fn_dict,
    get_indices_fn: Optional[callable] = get_indices_fn_causal_lm,
    prepare_activations_fn: Union[callable, Dict[str, callable], None] = None,
    show_progress_bar: bool = True,
    adapter_name: str = "default",
):
    """
    Initialize the weights of the LoRA layers using the EVA method.

    This function initializes the weights of the LoRA layers using the EVA method. It computes the SVD for each adapter
    layer and updates the weights accordingly.

    Args:
        model (torch.nn.Module): The model to compute the SVD for.
        peft_config (LoraConfig): The configuration for the LoRA layers.
        dataloader (torch.utils.data.DataLoader): The dataloader to use for the forward pass.
        forward_fn (callable):
            The forward function to use for the forward pass. `model(**inputs)` is used if forward_fn is not provided.
        get_indices_fn (Optional[callable]):
            The function to use if not all positions in the input tensor should be used for SVD (e.g. for causal
            language modeling). Can be set to None if all positions should be used.
            `peft.tuners.lora.eva.get_indices_fn_causal_lm` is used by default. Should always return a tensor of shape
            (n_indices, layer_input.ndim-1).
        prepare_activations_fn (Union[callable, Dict[str, callable], None]):
            If a layer receives multiple inputs as a list, per default the first input is used. This function can be
            used to modify this behaviour. Accepts a dictionary with layer names as keys. Default logic:
            ```
            try:
                svd_input = activations.detach()
            except AttributeError:
                svd_input = activations[0].detach()
            ```
        show_progress_bar (bool): Whether to show a progress bar. Default is True.
        adapter_name (str): The name of the adapter to initialize the weights for.

    Returns:
        model (torch.nn.Module): The model with the initialized LoRA weights.
    """
    # assign default eva config
    if peft_config.eva_config is None:
        peft_config.eva_config = EvaConfig()

    # compute svd
    with model.disable_adapter():
        eva_state_dict = get_eva_state_dict(
            model=model,
            peft_config=peft_config,
            dataloader=dataloader,
            forward_fn=forward_fn,
            get_indices_fn=get_indices_fn,
            prepare_activations_fn=prepare_activations_fn,
            show_progress_bar=show_progress_bar,
        )

    # assert all lora layers are contained in eva_state_dict
    update_layer_kwargs = {
        "adapter_name": adapter_name,
        "lora_alpha": peft_config.lora_alpha,
        "lora_dropout": peft_config.lora_dropout,
        "use_rslora": peft_config.use_rslora,
        "use_dora": peft_config.use_dora,
    }
    missing_eva_inits = []
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        if name in eva_state_dict:
            w = eva_state_dict.pop(name)
            module.update_layer(r=w.size(0), init_lora_weights=peft_config.init_lora_weights, **update_layer_kwargs)
            module.lora_A[adapter_name].weight.copy_(w)
        else:
            module.update_layer(r=peft_config.r, init_lora_weights=True, **update_layer_kwargs)
            missing_eva_inits.append(name)
            print(name, type(module))

    if missing_eva_inits:
        warnings.warn(
            f"the following layers were initialized with init_lora_weights=True because they were not found in the eva state_dict: {missing_eva_inits}"
            "currently only the following lora modules are supported: {SUPPORTED_LORA_MODULES}"
        )
    return model
