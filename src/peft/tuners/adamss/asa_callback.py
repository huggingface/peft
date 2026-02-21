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
ASA (Adaptive Subspace Allocation) Callback for HuggingFace Trainer.

This is a thin wrapper around ``AdamssModel.update_and_allocate()``.  All ASA
logic (importance accumulation, global top-K masking, importance reset) lives
in :meth:`AdamssModel.update_and_allocate` so that users with custom training
loops can call it directly without needing this callback.

Important:
    To avoid circular imports between peft and transformers, this callback is NOT
    exported from the top-level ``peft`` package. Import it directly::

        from peft.tuners.adamss.asa_callback import AdamssAsaCallback
"""


from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class AdamssAsaCallback(TrainerCallback):
    """
    Trainer callback for Adaptive Subspace Allocation (ASA).

    This callback delegates to :meth:`AdamssModel.update_and_allocate` on every
    optimizer step so that ASA "just works" with HuggingFace ``Trainer``.

    All ASA parameters (``asa_target_subspaces``, ``init_warmup``, ``final_warmup``,
    ``mask_interval``, etc.) are read from the :class:`AdamssConfig` that was used
    to create the model – there is nothing to configure on the callback itself.

    For custom training loops **without** Trainer, call
    ``model.base_model.update_and_allocate(global_step)`` directly instead.

    Example::

        from peft import AdamssConfig, get_peft_model
        from peft.tuners.adamss.asa_callback import AdamssAsaCallback
        from transformers import Trainer

        config = AdamssConfig(
            r=100, num_subspaces=10, subspace_rank=3,
            use_asa=True, asa_target_subspaces=5,
            init_warmup=50, final_warmup=1000, mask_interval=100,
        )
        model = get_peft_model(base_model, config)

        trainer = Trainer(
            model=model,
            callbacks=[AdamssAsaCallback()],
            ...,
        )
        trainer.train()
    """

    def on_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called after optimizer.step() – delegates to model.update_and_allocate()."""
        model = kwargs.get("model")
        if model is None:
            return control

        # Resolve to the AdamssModel (works for both PeftModel and raw base_model)
        base_model = getattr(model, "base_model", model)
        if hasattr(base_model, "update_and_allocate"):
            base_model.update_and_allocate(state.global_step)

        return control
