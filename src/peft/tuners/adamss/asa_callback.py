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
        tt: float = 3.0,
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
        self.tt = tt
        
        # Sanity checks
        assert 0 < beta1 < 1, f"beta1 must be in (0, 1), got {beta1}"
        assert 0 < beta2 < 1, f"beta2 must be in (0, 1), got {beta2}"
        assert init_warmup < final_warmup, f"init_warmup ({init_warmup}) must be < final_warmup ({final_warmup})"
        
        self.current_step = 0
        self._collected_total_kk = False
        self.total_kk = None
        self.trainer = None  # 保存trainer实例
    
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
        
        print(f"[DEBUG][_mask_model_to_target] 开始遍历模型，target_kk={target_kk}")
        for name, module in model.named_modules():
            if isinstance(module, AdaMSSLayer):
                print(f"[DEBUG][_mask_model_to_target] 找到 AdaMSSLayer: {name}")
                for adapter_name in module.exp_avg_ipt.keys():
                    print(f"[DEBUG][_mask_model_to_target] 调用 mask_to_target for adapter: {adapter_name}")
                    module.mask_to_target(adapter_name, target_kk)
    
    def _reset_model_importance(self, model):
        """Reset importance stats for all AdaMSS layers."""
        from .layer import AdaMSSLayer
        for module in model.modules():
            if isinstance(module, AdaMSSLayer):
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
            
            print(f"[DEBUG][ASA] step={current_step}, init_warmup={self.init_warmup}, final_warmup={self.final_warmup}, mask_interval={self.mask_interval}")
            print(f"[DEBUG][ASA] masking条件: True")
            print(f"[DEBUG][ASA] 调用_schedule_threshold: curr_kk={curr_kk}, should_mask={should_mask}")
            
            if should_mask:
                self._mask_model_to_target(model, curr_kk)

                # 关键：重建 optimizer 以同步 requires_grad 变化
                if self.trainer is not None:
                    self.trainer.create_optimizer_and_scheduler(self.trainer.num_training_steps)
                    # Debug print: 检查 optimizer param group 中参数的 requires_grad 状态
                    # print("[DEBUG][ASA] Optimizer param groups after masking:")
                    # for i, group in enumerate(self.trainer.optimizer.param_groups):
                    #     print(f"  Param group {i}:")
                    #     for p in group['params']:
                    #         print(f"    shape={tuple(p.shape)}, requires_grad={p.requires_grad}")

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
        # 保存 trainer 实例
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
