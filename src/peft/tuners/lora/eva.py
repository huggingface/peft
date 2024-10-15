import torch
from tqdm import tqdm
from functools import reduce, partial
from itertools import cycle
from collections import Counter, defaultdict

from peft.utils.incremental_pca import IncrementalPCA
from peft.tuners import lora
from peft.tuners.lora.config import EvaConfig, LoraConfig

from typing import Optional, Union, Tuple



class SVDHook:
    """
    A forward hook for calculating incremental SVD/PCA on layer input activations.

    This hook performs a step of incremental Singular Value Decomposition (SVD)
    on the input activations of a specified layer during the forward pass of a neural network.

    The hook also tracks convergence of the computed components using cosine similarity
    between the current and previous components.

    Attributes:
        name (str): Name of the layer to which this hook is attached.
        n_components (int): Number of principal components to compute.
        sim_thresh (Union[float, torch.Tensor]): Similarity threshold for convergence.

    The hook is designed to be registered to a PyTorch module using the `register_forward_hook` method.
    """
    def __init__(
        self,
        name: str,
        n_components: int,
        sim_thresh: Union[float, torch.Tensor]
    ):
        self.name = name
        self.n_components = n_components
        self.sim_thresh = sim_thresh

        if isinstance(sim_thresh, torch.Tensor) and len(sim_thresh.shape) > 0:
            check1 = sim_thresh.size(0) == n_components or sim_thresh.size(0) == 1
            check2 = len(sim_thresh.shape) == 1
            assert check1 and check2, "if sim_thresh is a tensor with more than 0 dimensions it must have shape (n_components,) or (1,)"

        self.svd = IncrementalPCA(n_components=n_components, copy=True, lowrank=True)

        self.indices = None
        self.converged = torch.zeros((n_components,), dtype=torch.bool)

    def __call__(self, model, input, output):
        previous_components = None
        if hasattr(self.svd, "components_"):
            previous_components = self.svd.components_.clone().detach()

        try:
            states = input.detach()
        except AttributeError:
            states = input[0].detach()
        states = states[self.indices[:, 0], self.indices[:, 1], :] # TODO might not work for all use cases

        if states.size(0) < self.n_components:
            print(f"skipping SVD for {self.name} because there are less than {self.n_components} tokens")
            return

        self.svd.partial_fit(states.to(torch.float32))

        if previous_components is not None:
            components = self.svd.components_
            if len(components.shape) == 1:
                components = components.reshape(1, -1)
                previous_components = previous_components.reshape(1, -1)
            # consider as converged if enough components have converged via cossim
            sim = torch.nn.functional.cosine_similarity(components, previous_components)
            self.converged = (sim >= self.sim_thresh)


# This is used to determine if two input activations are equal. For such cases, SVD
# needs to be done for only for one of the equal inputs.
class HashHook:
    """
    A forward hook for hashing layer input activations.

    This hook hashes the input activations of a specified layer during the forward pass
    of a neural network and stores the hash values for later analysis or comparison.

    Attributes:
        name (str): Name of the layer to which this hook is attached.
        hashed_inputs (list): List of hashed input activations.
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


def find_equal_values(dictionary):
    """
    Find keys in a dictionary that have the same value.

    This function takes a dictionary and returns a new dictionary containing keys that have the same value.
    The keys in the output dictionary are the values from the input dictionary, and the values are lists of keys
    that share the same value.
    """
    value_dict = defaultdict(list)
    for k,v in dictionary.items():
        value_dict[v].append(k)
    return {k: v for k, v in value_dict.items() if len(v) > 1}


def recursive_apply(module, device):
    """
    Recursively apply a function to the module and its submodules.

    If the module is a LoraLinear, the base layer is moved to the specified device.
    If the module has any LoraLinear submodules, the function is applied recursively to each submodule.
    Otherwise, the module is moved to the specified device.
    """
    if isinstance(module, lora.LoraLayer):
        module.base_layer.to(device)
        return
    if any(isinstance(submodule, lora.LoraLayer) for submodule in module.modules()):
        for child in module.children():
            recursive_apply(child, device)
    else:
        module.to(device)


def get_device_with_meta_params(model):
    """
    Get the device of the model's parameters. Useful if some parameters are on meta device.
    """
    devices = list(set([p.device for p in model.parameters() if str(p.device) != "meta"]))
    assert len(devices) == 1, "model has multiple devices"
    return devices[0]


@torch.no_grad()
def compute_svd(
    model: torch.nn.Module,
    peft_config: LoraConfig,
    dataloader: torch.utils.data.DataLoader,
    collate_fn: callable,
    forward_fn: callable
) -> dict:
    """
    Compute the SVD for each layer in the model.

    This function computes the Singular Value Decomposition (SVD) for each layer in the model.
    It uses the incremental PCA method to compute the SVD components.
    The function also checks for convergence of the computed components using cosine similarity.
    The rank distribution for each layer is determined based on the explained variance ratio.
    """
    
    # Computes the rank distribution for each layer based on the explained variance ratio.
    def _get_rank_distribution(hooks, hook_layer_map, equal_inputs_map, rank_budget, max_components):
        exp_vars = {k: h.svd.explained_variance_ratio_[:max_components] for k, h in hooks.items()}
        keys, values = zip(*[(k, c) for k, name in hook_layer_map.items() for c in exp_vars[name]])
        idx = torch.stack(values).argsort(descending=True)
        counts = Counter([keys[i] for i in idx[:rank_budget]])
        counts = {k: counts.get(k, 0) for k in hook_layer_map.keys()} # add layers with 0 rank
        for k, k_hook in equal_inputs_map.items():
            # ensure hook layers have the highest rank if they are equal to another layer
            rank, rank_hook = counts[k], counts[k_hook]
            if rank_hook >= rank:
                continue
            counts[k_hook], counts[k] = rank, rank_hook
        return counts

    training = model.training
    device = get_device_with_meta_params(model)
    model.eval()

    hooks = {}
    for name, module in model.named_modules():
        if isinstance(module, lora.LoraLayer):
            hook = HashHook(name)
            module.register_forward_hook(hook)
            hooks[name] = hook
    rank_budget = peft_config.r * len(hooks)
    max_components = round(peft_config.r * peft_config.eva_config.rho)

    # forward for one batch to check which layer inputs are equal to avoid unneeded svd calculations
    inputs, _ = collate_fn(next(iter(dataloader)), peft_config, device)
    forward_fn(model, inputs)
    hash_dict = {k: h.hashed_inputs[0] for k, h in hooks.items()}
    equal_inputs_map = {vv: v[0] for v in find_equal_values(hash_dict).values() for vv in v[1:]}
    hooks = {k: SVDHook(k, max_components, peft_config.eva_config.tau) for k in hooks.keys() if k not in equal_inputs_map}
    layer_hook_map = {**dict(zip(hooks.keys(), hooks.keys())), **equal_inputs_map}
    for name in layer_hook_map.keys():
        module = reduce(getattr, name.split("."), model) # TODO: replace with model.get_submodule(name)
        module._forward_hooks.clear()
    
    # start svd calculation
    pbar = tqdm(iter(cycle(dataloader)), position=0, leave=False)
    convergence_dict = {k: False for k in hooks.keys()}
    rank_dist = {k: max_components for k in layer_hook_map.keys()}
    for inputs in pbar:

        inputs, indices = collate_fn(inputs, peft_config, device)

        for name, hook in hooks.items():
            module = reduce(getattr, name.split("."), model) # TODO: replace with model.get_submodule(name)
            module._forward_hooks.clear()
            # check if all components that are needed for the rank distribution have converged
            if torch.all(hook.converged[:rank_dist[name]]):
                convergence_dict[name] = True
                continue
            convergence_dict[name] = False
            hook.indices = indices
            module.register_forward_hook(hook)

        if all(convergence_dict.values()):
            print("exiting - all SVD components have converged.")
            break

        forward_fn(model, inputs)

        # in case some hooks have to skip the svd calculation because the number of tokens is less than the number of components
        if not all([hasattr(h.svd, "components_") for h in hooks.values()]):
            continue

        rank_dist = _get_rank_distribution(hooks, layer_hook_map, equal_inputs_map, rank_budget, max_components)

        layer_converged = list(convergence_dict.values()) + [convergence_dict[v] for v in equal_inputs_map.values()]
        pbar.set_description(f"{sum(layer_converged)}/{len(layer_converged)} layers have converged")

    svd_dict = {}
    for name, rank in rank_dist.items():
        if rank == 0:
            continue
        hook = hooks[layer_hook_map[name]]
        assert torch.all(hook.converged[:rank]) # this should never happen because we check for convergence
        u = hook.svd.components_[:rank]
        if peft_config.eva_config.whiten:
            u /= hook.svd.singular_values_[:rank].sqrt().reshape(-1, 1)
        svd_dict[name] = u

    # objects are torch tensors on the model device
    svd_dict = {k: v.to(device) for k, v in svd_dict.items()}

    # restore model state
    model.train(training)

    return svd_dict


def collate_fn_language_modeling(inputs: dict, config: LoraConfig, device: Union[str, torch.device]):
    "default collate_fn for autregressive language model"
    mask = torch.ones_like(inputs["input_ids"], dtype=torch.bool)
    if hasattr(inputs, "attention_mask"):
        mask = inputs["attention_mask"].bool()
    if config.eva_config.use_label_mask and hasattr(inputs, "labels"):
        mask = torch.logical_and(mask, inputs["labels"] != config.eva_config.label_mask_value)
    indices = torch.nonzero(mask)
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "labels"}
    return inputs, indices


def forward_fn_language_modeling(model, inputs: dict):
    "default forward_fn for autregressive language model"
    return model(**inputs)


@torch.no_grad()
def initialize_lora_eva_weights(
    model,
    config,
    dataloader,
    collate_fn = None,
    forward_fn = None,
    device = None,
    adapter_name = "default", # TODO
):
    orig_device = get_device_with_meta_params(model)
    if device is not None:
        recursive_apply(model, device)
    model.disable_adapter()
    if collate_fn is None:
        collate_fn = collate_fn_language_modeling
    if forward_fn is None:
        forward_fn = forward_fn_language_modeling
    with model.disable_adapter():
        svd_dict = compute_svd(model, config, dataloader, collate_fn, forward_fn)
    for k in list(svd_dict.keys()):
        w = svd_dict.pop(k)
        layer = model.get_submodule(k)
        layer.update_layer(
            adapter_name,
            w.size(0),
            config.lora_alpha,
            config.lora_dropout,
            config.init_lora_weights,
            config.use_rslora,
            config.use_dora
        )
        layer.lora_A[adapter_name].weight.copy_(w)
    return model.to(orig_device)
