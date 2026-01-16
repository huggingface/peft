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
ASA (Adaptive Subspace Allocation) Callback for Adamss.

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


class ASACallback(TrainerCallback):
    """
    Callback for Adaptive Subspace Allocation (ASA).
    
    This callback periodically updates importance scores and masks less important
    subspaces during training to achieve adaptive parameter selection.
    
    Important:
        This callback implements the same functionality as `AdamssModel.update_and_allocate()`.
        When using this callback with HuggingFace Trainer, DO NOT manually call
        `model.base_model.update_and_allocate(global_step)` as it would result in
        duplicate ASA updates.
        
        - Use ASACallback: for HuggingFace Trainer
        - Use update_and_allocate(): for custom training loops
    
    Args:
        target_kk (int): Target number of active subspaces.
        init_warmup (int): Initial warmup steps before starting masking.
        final_warmup (int): Final warmup step when target_kk is reached.
        mask_interval (int): Steps between ASA updates.
        beta1 (float): EMA coefficient for importance averaging (default: 0.85).
        beta2 (float): EMA coefficient for uncertainty averaging (default: 0.85).
        tt (float): Schedule exponent for gradual transition (default: 3.0).
        total_steps (Optional[int]): Total training steps. If None, computed from training args.
        verbose (bool): Enable verbose debug output (default: False).
    
    Example:
        ```python
        from peft import AdamssConfig, get_peft_model, ASACallback
        from transformers import Trainer
        
        # Configure Adamss with ASA
        config = AdamssConfig(
            r=100,
            num_subspaces=10,
            subspace_rank=3,
            use_asa=True,
            target_kk=5,
        )
        model = get_peft_model(model, config)
        
        # Create ASA callback
        asa_callback = ASACallback(
            target_kk=5,
            init_warmup=50,
            final_warmup=1000,
            mask_interval=100,
        )
        
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
        target_kk: int = 50,
        init_warmup: int = 50,
        final_warmup: int = 1000,
        mask_interval: int = 100,
        beta1: float = 0.85,
        beta2: float = 0.85,
        tt: float = 3.0,
        total_steps: Optional[int] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.target_kk = target_kk
        self.init_warmup = init_warmup
        self.final_warmup = final_warmup
        self.mask_interval = mask_interval
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_steps = total_steps
        self.tt = tt
        self.verbose = verbose
        
        # Sanity checks
        assert 0 < beta1 < 1, f"beta1 must be in (0, 1), got {beta1}"
        assert 0 < beta2 < 1, f"beta2 must be in (0, 1), got {beta2}"
        assert init_warmup < final_warmup, f"init_warmup ({init_warmup}) must be < final_warmup ({final_warmup})"
        
        self.current_step = 0
        self._collected_total_kk = False
        self.total_kk = None
        self.trainer = None  # Store trainer instance
    
    def _schedule_threshold(self, step: int) -> tuple[int, bool]:
        """
        Calculate current target KK based on warmup schedule.
        
        Returns:
            (curr_kk, should_mask): Current target number of subspaces and whether to apply masking.
        """
        if step < self.init_warmup:
            # Initial warmup: use all subspaces
            return self.total_kk, False
        
        elif step <= self.final_warmup:
            # Gradual decrease following adamss_pkg schedule
            # mul_coeff = 1 - (step - init) / (final - init)
            # curr_kk = target_kk + (total_kk - target_kk) * (mul_coeff ** tt)
            mul_coeff = 1.0 - (step - self.init_warmup) / (self.final_warmup - self.init_warmup)
            mul_coeff = max(0.0, min(1.0, mul_coeff))
            curr_kk = int(self.target_kk + (self.total_kk - self.target_kk) * (mul_coeff ** self.tt))
            return curr_kk, True
        
        else:
            # After final warmup: fix target_kk; no further masking needed
            return self.target_kk, False
    
    def _collect_total_kk(self, model):
        """Collect total number of subspaces from the model (called once)."""
        if self._collected_total_kk:
            return

        # Find Adamss layers and collect per-layer KK, then sum to global total
        from .layer import AdamssLayer

        adapters_info: list[tuple[object, str, int]] = []  # (module, adapter_name, KK)
        total = 0
        for name, module in model.named_modules():
            if isinstance(module, AdamssLayer):
                # module.KK is a dict mapping adapter_name->KK for that module
                for adapter_name, kk in module.KK.items():
                    try:
                        kk_int = int(kk)
                    except Exception:
                        kk_int = kk
                    adapters_info.append((module, adapter_name, kk_int))
                    total += kk_int

        if total == 0 or not adapters_info:
            raise RuntimeError("ASA: Could not find Adamss layers or KK information in model")

        self._adapters_info = adapters_info
        self.total_kk = total
        self._collected_total_kk = True
        print(f"ASA: Detected total_kk (global) = {self.total_kk} subspaces across {len(adapters_info)} adapters")
    
    def _update_model_importance(self, model):
        """Update importance scores for all Adamss layers."""
        from .layer import AdamssLayer
        
        for name, module in model.named_modules():
            if isinstance(module, AdamssLayer):
                for adapter_name in module.exp_avg_ipt.keys():
                    module.update_importance(adapter_name, self.beta1, self.beta2)
    
    def _mask_model_to_target(self, model, target_kk: int):
        """
        Apply global top-K masking across all layers.
        
        This implements the exact behavior of adamss_pkg: collect importance scores
        from all subspaces across all layers, rank them globally, and select the
        top target_kk subspaces to keep active.
        """
        from .layer import AdamssLayer
        
        if self.verbose:
            print(f"[DEBUG][_mask_model_to_target] Starting global top-{target_kk} masking")
        
        # Ensure we have adapters info collected
        if not getattr(self, "_collected_total_kk", False):
            self._collect_total_kk(model)

        adapters = getattr(self, "_adapters_info", [])
        if not adapters:
            if self.verbose:
                print("[DEBUG][_mask_model_to_target] No adapters collected, skipping")
            return

        # Step 1: Collect importance scores for all subspaces across all layers
        # Format: list of (module, adapter_name, subspace_idx, importance_score)
        subspace_scores = []
        
        for module, adapter_name, kk in adapters:
            if adapter_name not in module.exp_avg_ipt:
                if self.verbose:
                    print(f"[DEBUG] Adapter {adapter_name} has no importance tracking, skipping")
                continue
            
            exp_avg_ipt = module.exp_avg_ipt[adapter_name]
            exp_avg_unc = module.exp_avg_unc[adapter_name]
            
            # Calculate score for each subspace in this adapter
            for i in range(kk):
                key_A = f"{adapter_name}_A_{i}"
                key_B = f"{adapter_name}_B_{i}"
                
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
        # If target_kk >= total subspaces, keep all active
        if target_kk >= len(subspace_scores):
            mask_threshold = float('-inf')  # Keep all
            if self.verbose:
                print(f"[DEBUG][_mask_model_to_target] target_kk >= total subspaces, keeping all active")
        else:
            # Use kthvalue: find the target_kk-th largest score
            # Note: kthvalue returns kth smallest, so we negate the scores
            mask_threshold = -torch.kthvalue(-all_scores, target_kk)[0].item()
            if self.verbose:
                print(f"[DEBUG][_mask_model_to_target] Mask threshold: {mask_threshold}")
        
        # Step 4: Apply masking - subspaces with score > threshold are active
        # This matches adamss_pkg: is_dict[name_mat] > mask_threshold
        for module, adapter_name, kk in adapters:
            num_active_in_adapter = 0
            
            for i in range(kk):
                key_A = f"{adapter_name}_A_{i}"
                key_B = f"{adapter_name}_B_{i}"
                
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
                
                # Set requires_grad for both A and B parameters
                if key_A in module.adamss_A:
                    module.adamss_A[key_A].requires_grad = is_active
                    if not is_active:
                        module.adamss_A[key_A].grad = None
                        
                if key_B in module.adamss_B:
                    module.adamss_B[key_B].requires_grad = is_active
                    if not is_active:
                        module.adamss_B[key_B].grad = None
                
                if is_active:
                    num_active_in_adapter += 1
            
            if self.verbose:
                print(f"[DEBUG][_mask_model_to_target] Adapter {adapter_name}: {num_active_in_adapter}/{kk} subspaces active")
    
    def _reset_model_importance(self, model):
        """Reset importance stats for all Adamss layers."""
        from .layer import AdamssLayer
        for module in model.modules():
            if isinstance(module, AdamssLayer):
                for adapter_name in module.exp_avg_ipt.keys():
                    module.reset_importance(adapter_name)

    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called after optimizer step but before zero_grad.
        This is the correct place to inspect gradients for importance calculation.
        """
        model = kwargs.get("model")
        if model is None:
            return control
            
        current_step = state.global_step
        
        # Check if we are in the active range and at an interval
        is_warmup = self.init_warmup <= current_step <= self.final_warmup
        is_interval = (current_step % self.mask_interval == 0)
        
        if is_warmup and is_interval:
            # 1. Reset importance (snapshot behavior)
            self._reset_model_importance(model)
            
            # 2. Update importance using current accumulated gradients
            self._update_model_importance(model)
            
            # 3. Calculate threshold and apply mask
            curr_kk, should_mask = self._schedule_threshold(current_step)
            
            if self.verbose:
                print(f"[DEBUG][ASA] step={current_step}, init_warmup={self.init_warmup}, final_warmup={self.final_warmup}, mask_interval={self.mask_interval}")
                print(f"[DEBUG][ASA] Masking condition: True")
                print(f"[DEBUG][ASA] Called _schedule_threshold: curr_kk={curr_kk}, should_mask={should_mask}")
            
            if should_mask:
                self._mask_model_to_target(model, curr_kk)

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
                    f"ASA step {current_step}: Masked to {curr_kk}/{self.total_kk} subspaces | "
                    f"AdaMSS active: {active_adamss:,}/{total_adamss:,}"
                )
        
        return control

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called at the beginning of each training step.
        """
        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called at the end of each training step.
        Used only for collecting initial metadata.
        """
        model = kwargs.get("model")
        if model is None:
            return control
        
        # Collect total_kk on first step
        if not self._collected_total_kk:
            self._collect_total_kk(model)
            
            # Set total_steps if not provided
            if self.total_steps is None:
                self.total_steps = state.max_steps
            
            # Validate warmup schedule
            if self.total_steps and self.final_warmup > self.total_steps:
                print(f"Warning: final_warmup ({self.final_warmup}) > total_steps ({self.total_steps}), adjusting...")
                self.final_warmup = int(self.total_steps * 0.8)
        
        self.current_step = state.global_step
        
        return control
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of training."""
        # Store trainer instance
        self.trainer = kwargs.get('trainer', None)
        print("="*80)
        print("ASA (Adaptive Subspace Allocation) Enabled")
        print("="*80)
        print(f"  Target subspaces: {self.target_kk}")
        print(f"  Initial warmup: {self.init_warmup} steps")
        print(f"  Final warmup: {self.final_warmup} steps")
        print(f"  Mask interval: {self.mask_interval} steps")
        print(f"  Beta1 (importance): {self.beta1}")
        print(f"  Beta2 (uncertainty): {self.beta2}")
        print(f"  Schedule exponent tt: {self.tt}")
        print("="*80 + "\n")
        return control
