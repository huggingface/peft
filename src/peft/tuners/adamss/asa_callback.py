# Copyright 2026-present the HuggingFace Inc. team.
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

"""
ASA (Adaptive Subspace Allocation) Callback of Adamss for Transformers Trainer.

This callback implements dynamic subspace selection during training based on
gradient-based importance scoring.

Note:
    This callback provides the same functionality as manually calling
    `model.base_model.update_and_allocate(global_step)` in a custom training loop.
    When using this callback with Trainer, DO NOT call update_and_allocate() manually
    as it would result in duplicate ASA updates.
    
    Use this callback if you're using HuggingFace Trainer, or call update_and_allocate()
    manually if you have a custom training loop.
"""

from typing import Optional
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
import torch
from .layer import AdamssLayer


class AdamssASACallback(TrainerCallback):
    """
    Callback for Adaptive Subspace Allocation (ASA).
    
    This callback periodically updates importance scores and masks less important
    subspaces during training to achieve adaptive parameter selection.
    
    Important:
        This callback implements the same functionality as `AdamssModel.update_and_allocate()`.
        When using this callback with HuggingFace Trainer, DO NOT manually call
        `model.base_model.update_and_allocate(global_step)` as it would result in
        duplicate ASA updates.
        
        - Use AdamssASACallback: for HuggingFace Trainer
        - Use update_and_allocate(): for custom training loops
    
    Note:
        ASA parameters (asa_target_subspaces, init_warmup, final_warmup, mask_interval,
        asa_importance_beta, asa_uncertainty_beta, asa_schedule_exponent) are read from
        the AdaMSSConfig. This ensures a single source of truth for these parameters.
    
    Args:
        total_steps (Optional[int]): Total training steps. If None, computed from training args.
        verbose (bool): Enable verbose debug output (default: False).
    
    Example:
        ```python
        from peft import AdamssConfig, get_peft_model, AdamssASACallback
        from transformers import Trainer
        
        # Configure Adamss with ASA - all ASA params are on the config
        config = AdamssConfig(
            r=100,
            num_subspaces=10,
            subspace_rank=3,
            use_asa=True,
            asa_target_subspaces=5,
            init_warmup=50,
            final_warmup=1000,
            mask_interval=100,
        )
        model = get_peft_model(model, config)
        
        # Create ASA callback - no need to repeat params
        asa_callback = AdamssASACallback()
        
        # Use with Trainer
        trainer = Trainer(
            model=model,
            callbacks=[asa_callback],
            ...
        )
        trainer.train()
        ```
    """
    
    def __init__(
        self,
        total_steps: Optional[int] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.total_steps = total_steps
        self.verbose = verbose
        
        # ASA parameters - will be read from config in on_train_begin
        self.asa_target_subspaces = None
        self.init_warmup = None
        self.final_warmup = None
        self.mask_interval = None
        self.asa_importance_beta = None
        self.asa_uncertainty_beta = None
        self.asa_schedule_exponent = None
        
        self.current_step = 0
        self.asa_total_subspaces = None
        self.trainer = None  # Store trainer instance
        self._config_loaded = False
        self._collected_asa_total_subspaces = False
    
    def _schedule_threshold(self, step: int) -> tuple[int, bool]:
        """
        Calculate current target subspaces based on warmup schedule.
        
        Returns:
            (current_active_subspaces, should_mask): Current target number of subspaces and whether to apply masking.
        """
        if step < self.init_warmup:
            # Initial warmup: use all subspaces
            return self.asa_total_subspaces, False
        
        elif step <= self.final_warmup:
            # Gradual decrease following adamss_pkg schedule
            # mul_coeff = 1 - (step - init) / (final - init)
            # current_active_subspaces = asa_target_subspaces + (asa_total_subspaces - asa_target_subspaces) * (mul_coeff ** asa_schedule_exponent)
            mul_coeff = 1.0 - (step - self.init_warmup) / (self.final_warmup - self.init_warmup)
            mul_coeff = max(0.0, min(1.0, mul_coeff))
            current_active_subspaces = int(self.asa_target_subspaces + (self.asa_total_subspaces - self.asa_target_subspaces) * (mul_coeff ** self.asa_schedule_exponent))
            return current_active_subspaces, True
        
        else:
            # After final warmup: fix asa_target_subspaces; no further masking needed
            return self.asa_target_subspaces, False
    
    def _collect_asa_total_subspaces(self, model):
        """Collect total number of subspaces from the model (called once)."""
        if self.asa_total_subspaces is not None:
            return

        # Find Adamss layers and collect per-layer num_subspaces, then sum to global total
        adapters_info: list[tuple[object, str, int]] = []  # (module, adapter_name, num_subspaces)
        total = 0
        for module in model.modules():
            if isinstance(module, AdamssLayer) and module.num_subspaces:
                for adapter_name, n_subspaces in module.num_subspaces.items():
                    n_subspaces_int = int(n_subspaces)
                    adapters_info.append((module, adapter_name, n_subspaces_int))
                    total += n_subspaces_int
        else:
            if not adapters_info:
                raise RuntimeError("ASA: Could not find Adamss layers or num_subspaces information in model")

        self._adapters_info = adapters_info
        self.asa_total_subspaces = total
        self._collected_asa_total_subspaces = True
    
    def _update_model_importance(self, model):
        """
        Update (accumulate) importance scores for all Adamss layers.
        
        This should be called every step during the warmup period. Importance
        is accumulated using exponential moving average (EMA) over multiple
        calls, building up meaningful statistics for masking decisions.
        """
        for module in model.modules():
            if isinstance(module, AdamssLayer):
                for adapter_name in model.active_adapters:
                    if adapter_name in module.exp_avg_ipt:
                        module.update_importance(adapter_name, self.asa_importance_beta, self.asa_uncertainty_beta)
    
    def _mask_model_to_target(self, model, asa_target_subspaces: int):
        """
        Apply global top-K masking across all layers.
        
        This implements the exact behavior of adamss_pkg: collect importance scores
        from all subspaces across all layers, rank them globally, and select the
        top asa_target_subspaces subspaces to keep active.
        """
        
        if self.verbose:
            print(f"[DEBUG][_mask_model_to_target] Starting global top-{asa_target_subspaces} masking")
        
        # Ensure we have adapters info collected
        if not getattr(self, "_collected_asa_total_subspaces", False):
            self._collect_asa_total_subspaces(model)

        adapters = getattr(self, "_adapters_info", [])
        if not adapters:
            if self.verbose:
                print("[DEBUG][_mask_model_to_target] No adapters collected, skipping")
            return

        # Step 1: Collect importance scores for all subspaces across all layers
        # Format: list of (module, adapter_name, subspace_idx, importance_score)
        subspace_scores = []
        
        for module, adapter_name, n_subspaces in adapters:
            if adapter_name not in module.exp_avg_ipt:
                if self.verbose:
                    print(f"[DEBUG] Adapter {adapter_name} has no importance tracking, skipping")
                continue
            
            exp_avg_ipt = module.exp_avg_ipt[adapter_name]
            exp_avg_unc = module.exp_avg_unc[adapter_name]
            
            # Calculate score for each subspace in this adapter
            # Note: layer.py uses keys like "A_{i}" and "B_{i}" (without adapter_name prefix)
            for i in range(n_subspaces):
                key_A = f"A_{i}"
                key_B = f"B_{i}"
                
                if key_A not in exp_avg_ipt or key_B not in exp_avg_ipt:
                    continue
                
                # Score = (ipt * unc).mean() for both A and B
                score_A = (exp_avg_ipt[key_A] * exp_avg_unc[key_A]).mean()
                score_B = (exp_avg_ipt[key_B] * exp_avg_unc[key_B]).mean()
                total_score = score_A + score_B
                
                subspace_scores.append((module, adapter_name, i, total_score))
        
        if not subspace_scores:
            if self.verbose:
                print("[DEBUG][_mask_model_to_target] No importance scores available, skipping")
            return
        
        # Step 2: Use kthvalue to find threshold (matches adamss_pkg exactly)
        # Collect all scores into a tensor
        all_scores = torch.stack([s[3] for s in subspace_scores])
        
        if self.verbose:
            print(f"[DEBUG][_mask_model_to_target] Collected {len(subspace_scores)} subspaces globally")
            print(f"[DEBUG][_mask_model_to_target] Top 5 scores: {[float(s) for s in all_scores.topk(min(5, len(all_scores)))[0]]}")
        
        # Step 3: Find kth largest value as threshold
        # If asa_target_subspaces >= total subspaces, keep all active
        if asa_target_subspaces >= len(subspace_scores):
            mask_threshold = float('-inf')  # Keep all
            if self.verbose:
                print(f"[DEBUG][_mask_model_to_target] asa_target_subspaces >= total subspaces, keeping all active")
        else:
            # Use kthvalue: find the asa_target_subspaces-th largest score
            # Note: kthvalue returns kth smallest, so we negate the scores
            mask_threshold = -torch.kthvalue(-all_scores, asa_target_subspaces)[0].item()
            if self.verbose:
                print(f"[DEBUG][_mask_model_to_target] Mask threshold: {mask_threshold}")
        
        # Step 4: Apply masking - subspaces with score > threshold are active
        # This matches adamss_pkg: is_dict[name_mat] > mask_threshold
        for module, adapter_name, n_subspaces in adapters:
            num_active_in_adapter = 0
            
            # Get ParameterLists for this adapter (ModuleDict uses [] access, not .get())
            if adapter_name not in module.adamss_A or adapter_name not in module.adamss_B:
                continue
            
            param_list_A = module.adamss_A[adapter_name]
            param_list_B = module.adamss_B[adapter_name]
            
            for i in range(n_subspaces):
                # Find this subspace's score
                subspace_score = None
                for mod, adp, idx, score in subspace_scores:
                    if id(mod) == id(module) and adp == adapter_name and idx == i:
                        subspace_score = score
                        break
                
                # Active if score > threshold (strict inequality, like adamss_pkg)
                # Convert tensor to Python bool for requires_grad
                if subspace_score is not None:
                    is_active = (float(subspace_score) > mask_threshold)
                else:
                    is_active = False
                
                # Set requires_grad for both A and B parameters using ParameterList access
                if i < len(param_list_A):
                    param_list_A[i].requires_grad = is_active
                    if not is_active:
                        param_list_A[i].grad = None
                        
                if i < len(param_list_B):
                    param_list_B[i].requires_grad = is_active
                    if not is_active:
                        param_list_B[i].grad = None
                
                if is_active:
                    num_active_in_adapter += 1
            
            if self.verbose:
                print(f"[DEBUG][_mask_model_to_target] Adapter {adapter_name}: {num_active_in_adapter}/{n_subspaces} subspaces active")
    
    def _reset_model_importance(self, model):
        """Reset importance stats for all Adamss layers."""
        for module in model.modules():
            if isinstance(module, AdamssLayer):
                for adapter_name in model.active_adapters:
                    if adapter_name in module.exp_avg_ipt:
                        module.reset_importance(adapter_name)

    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called after optimizer step but before zero_grad.
        This is the correct place to inspect gradients for importance calculation.
        
        Importance is accumulated every step during the warmup period.
        At mask intervals, the accumulated importance is used for masking decisions,
        then reset to start fresh accumulation for the next interval.
        """
        model = kwargs.get("model")
        if model is None:
            return control
            
        current_step = state.global_step
        
        # Check if we are in the active ASA range
        is_in_warmup = self.init_warmup <= current_step <= self.final_warmup
        is_mask_interval = (current_step % self.mask_interval == 0)
        
        # Accumulate importance EVERY step during the warmup period
        # This builds up gradient-based importance scores over time
        if is_in_warmup:
            self._update_model_importance(model)
        
        # At mask intervals, use the accumulated importance to make masking decisions
        if is_in_warmup and is_mask_interval:
            # Calculate threshold and apply mask
            current_active_subspaces, should_mask = self._schedule_threshold(current_step)
            
            if self.verbose:
                print(f"[DEBUG][ASA] step={current_step}, init_warmup={self.init_warmup}, final_warmup={self.final_warmup}, mask_interval={self.mask_interval}")
                print(f"[DEBUG][ASA] Masking condition: True")
                print(f"[DEBUG][ASA] Called _schedule_threshold: current_active_subspaces={current_active_subspaces}, should_mask={should_mask}")
            
            if should_mask:
                self._mask_model_to_target(model, current_active_subspaces)

                # Critical: Rebuild optimizer to sync requires_grad changes
                if self.trainer is not None:
                    self.trainer.create_optimizer_and_scheduler(self.trainer.num_training_steps)
                    if self.verbose:
                        print("[DEBUG][ASA] Optimizer param groups after masking:")
                        for i, group in enumerate(self.trainer.optimizer.param_groups):
                            print(f"  Param group {i}:")
                            for p in group['params']:
                                print(f"    shape={tuple(p.shape)}, requires_grad={p.requires_grad}")

                # Print current active AdaMSS trainable parameter count for alignment checks
                active_adamss = sum(
                    p.numel() for n, p in model.named_parameters() if ('adamss' in n.lower() and p.requires_grad)
                )
                total_adamss = sum(
                    p.numel() for n, p in model.named_parameters() if ('adamss' in n.lower())
                )
                
                print(
                    f"ASA step {current_step}: Masked to {current_active_subspaces}/{self.asa_total_subspaces} subspaces | "
                    f"AdaMSS active: {active_adamss:,}/{total_adamss:,}"
                )
            
            # Reset importance AFTER masking to start fresh accumulation for next interval
            self._reset_model_importance(model)
        
        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called at the end of each training step.
        Used only for collecting initial metadata.
        """
        model = kwargs.get("model")
        if model is None:
            return control
        
        # Collect asa_total_subspaces on first step
        if not self._collected_asa_total_subspaces:
            self._collect_asa_total_subspaces(model)
            
            # Set total_steps if not provided
            if self.total_steps is None:
                self.total_steps = state.max_steps
            
            # Validate warmup schedule
            if self.total_steps and self.final_warmup > self.total_steps:
                print(f"Warning: final_warmup ({self.final_warmup}) > total_steps ({self.total_steps}), adjusting...")
                self.final_warmup = int(self.total_steps * 0.8)
        
        self.current_step = state.global_step
        
        return control
    
    def _load_config_from_model(self, model):
        """
        Load ASA parameters from the model's AdaMSSConfig.
        
        This ensures a single source of truth for ASA parameters - they are defined
        on the config, not duplicated on the callback.
        """
        if self._config_loaded:
            return
        
        # Find the AdaMSS config from the model
        config = None
        
        # Try to get config from PeftModel
        if hasattr(model, 'peft_config'):
            # peft_config is a dict of adapter_name -> config
            for adapter_name, adapter_config in model.peft_config.items():
                if hasattr(adapter_config, 'use_asa'):
                    config = adapter_config
                    break
        
        if config is None:
            raise RuntimeError(
                "ASA callback could not find AdaMSSConfig on the model. "
                "Make sure you are using an AdaMSS model with use_asa=True."
            )
        
        if not config.use_asa:
            raise RuntimeError(
                "ASA callback requires use_asa=True in AdaMSSConfig. "
                "Set use_asa=True when creating the config."
            )
        
        # Load ASA parameters from config
        self.asa_target_subspaces = config.asa_target_subspaces
        self.init_warmup = config.init_warmup
        self.final_warmup = config.final_warmup
        self.mask_interval = config.mask_interval
        self.asa_importance_beta = config.asa_importance_beta
        self.asa_uncertainty_beta = config.asa_uncertainty_beta
        self.asa_schedule_exponent = config.asa_schedule_exponent
        
        # Validate parameters
        if not (0 < self.asa_importance_beta < 1):
            raise ValueError(f"asa_importance_beta must be in (0, 1), got {self.asa_importance_beta}")
        if not (0 < self.asa_uncertainty_beta < 1):
            raise ValueError(f"asa_uncertainty_beta must be in (0, 1), got {self.asa_uncertainty_beta}")
        if self.init_warmup >= self.final_warmup:
            raise ValueError(f"init_warmup ({self.init_warmup}) must be < final_warmup ({self.final_warmup})")
        
        self._config_loaded = True

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of training."""
        # Store trainer instance
        self.trainer = kwargs.get('trainer', None)
        
        # Load ASA parameters from model config
        model = kwargs.get("model")
        if model is not None:
            self._load_config_from_model(model)
        
        print("="*80)
        print("ASA (Adaptive Subspace Allocation) Enabled")
        print("="*80)
        print(f"  Target subspaces: {self.asa_target_subspaces}")
        print(f"  Initial warmup: {self.init_warmup} steps")
        print(f"  Final warmup: {self.final_warmup} steps")
        print(f"  Mask interval: {self.mask_interval} steps")
        print(f"  Importance beta: {self.asa_importance_beta}")
        print(f"  Uncertainty beta: {self.asa_uncertainty_beta}")
        print(f"  Schedule exponent: {self.asa_schedule_exponent}")
        print("="*80 + "\n")
        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training."""
        # Break circular reference between callback and trainer
        self.trainer = None
        return control
