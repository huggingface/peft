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

"""
ASA (Adaptive Subspace Allocation) Callback for AdaMSS.

This callback implements dynamic subspace selection during training based on
gradient-based importance scoring.
"""

from typing import Optional
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
import torch


class ASACallback(TrainerCallback):
    """
    Callback for Adaptive Subspace Allocation (ASA).
    
    This callback periodically updates importance scores and masks less important
    subspaces during training to achieve adaptive parameter selection.
    
    Args:
        target_kk (int): Target number of active subspaces.
        init_warmup (int): Initial warmup steps before starting masking.
        final_warmup (int): Final warmup step when target_kk is reached.
        mask_interval (int): Steps between ASA updates.
        beta1 (float): EMA coefficient for importance averaging (default: 0.85).
        beta2 (float): EMA coefficient for uncertainty averaging (default: 0.85).
        total_steps (Optional[int]): Total training steps. If None, computed from training args.
    """
    
    def __init__(
        self,
        target_kk: int = 50,
        init_warmup: int = 50,
        final_warmup: int = 1000,
        mask_interval: int = 100,
        beta1: float = 0.85,
        beta2: float = 0.85,
        total_steps: Optional[int] = None,
    ):
        super().__init__()
        self.target_kk = target_kk
        self.init_warmup = init_warmup
        self.final_warmup = final_warmup
        self.mask_interval = mask_interval
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_steps = total_steps
        
        # Sanity checks
        assert 0 < beta1 < 1, f"beta1 must be in (0, 1), got {beta1}"
        assert 0 < beta2 < 1, f"beta2 must be in (0, 1), got {beta2}"
        assert init_warmup < final_warmup, f"init_warmup ({init_warmup}) must be < final_warmup ({final_warmup})"
        
        self.current_step = 0
        self._collected_total_kk = False
        self.total_kk = None
    
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
            # Gradual decrease from total_kk to target_kk
            # Use polynomial decay: curr_kk = total_kk - (total_kk - target_kk) * ((step - init) / (final - init))^3
            progress = (step - self.init_warmup) / (self.final_warmup - self.init_warmup)
            decay_ratio = progress ** 3  # Cubic decay
            curr_kk = int(self.total_kk - (self.total_kk - self.target_kk) * decay_ratio)
            return curr_kk, True
        
        else:
            # After final warmup: fixed target_kk
            return self.target_kk, True
    
    def _collect_total_kk(self, model):
        """Collect total number of subspaces from the model (called once)."""
        if self._collected_total_kk:
            return
        
        # Find AdaMSS layers and get total KK
        from .layer import AdaMSSLayer
        
        for name, module in model.named_modules():
            if isinstance(module, AdaMSSLayer):
                # Get KK from first adapter (assumes all adapters have same KK)
                if module.KK:
                    adapter_name = list(module.KK.keys())[0]
                    self.total_kk = module.KK[adapter_name]
                    self._collected_total_kk = True
                    print(f"ASA: Detected total_kk = {self.total_kk} subspaces")
                    break
        
        if not self._collected_total_kk:
            raise RuntimeError("ASA: Could not find AdaMSS layers in model")
    
    def _update_model_importance(self, model):
        """Update importance scores for all AdaMSS layers."""
        from .layer import AdaMSSLayer
        
        for name, module in model.named_modules():
            if isinstance(module, AdaMSSLayer):
                for adapter_name in module.exp_avg_ipt.keys():
                    module.update_importance(adapter_name, self.beta1, self.beta2)
    
    def _mask_model_to_target(self, model, target_kk: int):
        """Apply masking to all AdaMSS layers."""
        from .layer import AdaMSSLayer
        
        for name, module in model.named_modules():
            if isinstance(module, AdaMSSLayer):
                for adapter_name in module.exp_avg_ipt.keys():
                    module.mask_to_target(adapter_name, target_kk)
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called at the end of each training step.
        Updates importance and applies masking at regular intervals.
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
        
        # Always update importance (needed for EMA smoothing)
        self._update_model_importance(model)
        
        # Apply masking at intervals
        if self.current_step % self.mask_interval == 0:
            curr_kk, should_mask = self._schedule_threshold(self.current_step)
            
            if should_mask:
                self._mask_model_to_target(model, curr_kk)
                print(f"ASA step {self.current_step}: Masked to {curr_kk}/{self.total_kk} subspaces")
        
        return control
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of training."""
        print("="*80)
        print("ASA (Adaptive Subspace Allocation) Enabled")
        print("="*80)
        print(f"  Target subspaces: {self.target_kk}")
        print(f"  Initial warmup: {self.init_warmup} steps")
        print(f"  Final warmup: {self.final_warmup} steps")
        print(f"  Mask interval: {self.mask_interval} steps")
        print(f"  Beta1 (importance): {self.beta1}")
        print(f"  Beta2 (uncertainty): {self.beta2}")
        print("="*80 + "\n")
        return control
