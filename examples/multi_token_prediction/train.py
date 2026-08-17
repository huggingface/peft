import argparse
import math
import os
import time
from dataclasses import asdict, dataclass
from functools import partial
from itertools import cycle

import safetensors.torch
import torch
import torch.nn.functional as F
import trackio
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from peft import LoraConfig, TaskType, get_peft_model


@dataclass
class TrainConfig:
    batch_size: int
    num_mtp: int
    num_steps: int
    use_sampler: bool
    sampler_loss_weight: float
    sampler_detach_hidden_state: bool
    sampler_use_rnn: bool
    sampler_teacher_forcing: bool
    lc_loss_weight: float
    tv_loss_weight: float
    use_lc_loss: bool
    use_tv_loss: bool
    log_step: int
    eval_step: int
    output_dir: str
    checkpoint_dir: str
    checkpoint_step: int
    max_gradient_norm: float


MTP_MASK_TOKENS = None


def augment_mtp(sample: list[int], bs: int, num_mtp: int, use_lc_loss: bool, use_tv_loss: bool):
    """Augment samples with MTP tokens.

    Create `bs` samples based on a single sample, with each generated sample having a different
    starting position relative to the original sample.
    MTP tokens don't overlap. First starting position chosen at random, then offsets move forward
    with num_mtp steps.

    Input sequence
    X X X X X X X X X X

    random idx 3
    X X X X X X X X X X
          ^

    first sample (assuming 2 mask tokens)
    X X X M M

    second sample
    X X X X X M M

    etc.

    Offsets would be:
    [3, 5, 7, ...]

    If latent consistency loss (lc) is used, the last sample contains no mask tokens:
    X X X X X X X

    Labels are -100 except for the mask token positions.

    first sample
    -100 -100 -100 X X

    Returns
        input_ids: list[list[int]]
        labels: list[list[int]]
        offsets: list[int]

    """
    assert bs > 1
    seq_len = len(sample)
    assert seq_len > bs * num_mtp + 2
    # +2 because the first generated token is always NTP
    rand_idx_seq = torch.randint(2, seq_len - bs * num_mtp, (1,))[0]

    input_ids = []
    offsets = []
    labels = []

    indices = range(bs - int(use_lc_loss or use_tv_loss))
    for i in indices:
        # each sequence in the batch is longer by num_mtp tokens instead of just 1
        # to be more sample efficient.
        end_idx = rand_idx_seq + i * num_mtp
        input_ids.append(sample[:end_idx] + MTP_MASK_TOKENS)
        # the offset represents the position of the first MTP token, since python
        # indices are non-inclusive, this means that sample goes to end_idx - 1
        # and therefore the first MTP token is at end_idx.
        offsets.append(end_idx)
        # the labels will always be as long as the input sequence but we need to
        # mask the values not relating to the MTP input tokens.
        labels.append([-100] * (end_idx) + sample[end_idx + 1 : end_idx + 1 + num_mtp])

    if use_lc_loss or use_tv_loss:
        end_idx = rand_idx_seq + (bs - 1) * num_mtp
        input_ids.append(sample[:end_idx])
        offsets.append(end_idx)
        labels.append([-100] * end_idx)

    return input_ids, labels, offsets


class Sampler(nn.Module):
    """Section 2.3"""

    def __init__(self, embedding, unembedding, num_mtp: int, hidden_size: int, skip_last: bool, recurrent: bool = False,
                 teacher_forcing: bool = False):
        super().__init__()
        self.k = num_mtp
        self.hidden_size = hidden_size
        self.skip_last = skip_last
        self.recurrent = recurrent
        self.teacher_forcing = teacher_forcing

        self.embedding = embedding

        if recurrent:
            self.sampler = SamplerModuleRecurrent(unembedding, hidden_size)
        else:
            self.sampler = SamplerModule(unembedding, hidden_size)

    def forward(self, logits, hidden, offsets):
        # when using LC loss, last sequence in batch has no mask tokens to make
        # the reference computation easier, therefore we need to skip that here.
        if self.skip_last:
            logits = logits[:-1]
            hidden = hidden[:-1]
            offsets = offsets[:-1]

        # Next Token Prediction logits are used for the first sampler step.
        #   The sampler's output is used in the following steps.
        # Multi Token Prediction hidden states are used in every sampler step.
        ntp_logits = []
        mtp_hidden = []
        mtp_logits = []
        for batch_idx, offset in enumerate(offsets):
            # logits: [B, T, V]
            # hidden: [B, T, H]
            ntp_logits.append(logits[batch_idx, offset - 1 : offset])  # [K, V]
            mtp_hidden.append(hidden[batch_idx, offset : offset + self.k])  # [K, H]
            if self.teacher_forcing:
                mtp_logits.append(logits[batch_idx, offset : offset + self.k])  # [K, V]
        ntp_logits = torch.stack(ntp_logits, dim=0)  # [B, 1, V]
        mtp_hidden = torch.stack(mtp_hidden, dim=0)  # [B, K, H]
        if self.teacher_forcing:
            mtp_logits = torch.stack(mtp_logits, dim=0)  # [B, K, V]
        prev_token = ntp_logits.argmax(dim=-1)  # [B, 1]
        all_logits = []

        if self.recurrent:
            sampler_hidden = torch.zeros(1, mtp_hidden.shape[0], mtp_hidden.shape[-1]).to(logits)  # [L, B, H]

        for i_k in range(self.k):
            prev_token_emb = self.embedding(prev_token)  # [B, 1, H]
            if self.recurrent:
                sampler_logits, sampler_hidden = self.sampler(
                    mtp_hidden[:, i_k : i_k + 1],  # [b, 1, h]
                    prev_token_emb,
                    sampler_hidden,
                )  # [b, 1, v]
            else:
                sampler_logits = self.sampler(
                    mtp_hidden[:, i_k : i_k + 1],  # [b, 1, h]
                    prev_token_emb,
                )  # [b, 1, v]

            if self.training and self.teacher_forcing:
                prev_token = mtp_logits[:, i_k].argmax(dim=-1)  # [B, 1]
            else:
                prev_token = sampler_logits.argmax(dim=-1)  # [B, 1]
            all_logits.append(sampler_logits)

        return torch.concat(all_logits, dim=1)  # [B, K, V]


class SamplerModule(torch.nn.Module):
    """
    Sampler network: two-layer MLP with residual connection to z_n.

    For each mask position n:
    - Input: [embedding(prev_token); hidden_state]
    - Output: logits for token at position n

    The MLP learns a residual correction to z_n based on prev_token:
        logits = W · (z_n + MLP([E_{y_{n-1}}; z_n]))
    This ensures the sampler starts from z_n (which already encodes the future
    via base-CE training) and the MLP only refines it using prev_token info.
    """

    def __init__(self, unembedding, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
        )
        # W is the unembedding layer (shared with base model)
        self.unembedding = unembedding

        self.reset_parameters()

    def reset_parameters(self):
        # Make the last layer zero so that we still allow gradients to flow but make
        # sure that the contribution is zero at the beginning to not add unnecessary noise.
        torch.nn.init.zeros_(self.mlp[-3].weight)
        if self.mlp[-3].bias is not None:
            torch.nn.init.zeros_(self.mlp[-3].bias)

    def forward(self, hidden_states: torch.Tensor, prev_token_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim] - hidden reps from decoder
            prev_token_embs: [batch, seq_len, hidden_dim] - embeddings of previously sampled tokens
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Concatenate [prev_token_emb; hidden_state]
        combined = torch.cat([prev_token_embs, hidden_states], dim=-1)  # [B, T, 2H]
        transformed = self.mlp(combined)  # [B, T, H]

        # Residual: add z_n so the MLP only learns corrections, not the full mapping
        transformed = transformed + hidden_states  # [B, T, H]

        # Project to vocab using unembedding layer
        logits = F.linear(transformed, self.unembedding.weight)  # [B, T, V]
        return logits


class SamplerModuleRecurrent(torch.nn.Module):
    """Recurrent sampler, similar to SamplerModule but also uses a hidden state
    to model transitions.
    """

    def __init__(self, unembedding, hidden_size):
        super().__init__()
        self.mlp_pre = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
        )
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.mlp_post = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # W is the unembedding layer (shared with base model)
        self.unembedding = unembedding

        self.reset_parameters()

    def reset_parameters(self):
        # Make the last layer zero so that we still allow gradients to flow but make
        # sure that the contribution is zero at the beginning to not add unnecessary noise.
        torch.nn.init.zeros_(self.mlp_post[-1].weight)
        if self.mlp_post[-1].bias is not None:
            torch.nn.init.zeros_(self.mlp_post[-1].bias)

    def forward(self, hidden_states: torch.Tensor, prev_token_embs: torch.Tensor, rnn_hidden) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim] - hidden reps from decoder
            prev_token_embs: [batch, seq_len, hidden_dim] - embeddings of previously sampled tokens
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Concatenate [prev_token_emb; hidden_state]
        combined = torch.cat([prev_token_embs, hidden_states], dim=-1)  # [B, T, 2H]
        transformed = self.mlp_pre(combined)  # [B, T, H]
        transformed, rnn_hidden = self.rnn(transformed, rnn_hidden)
        transformed = self.mlp_post(transformed)

        # Residual: add z_n so the MLP only learns corrections, not the full mapping
        transformed = transformed + hidden_states  # [B, T, H]

        # Project to vocab using unembedding layer
        logits = F.linear(transformed, self.unembedding.weight)  # [B, T, V]
        return logits, rnn_hidden


def calculate_lc_loss(hidden_states, offsets, *, num_mtp: int, lc_loss_weight: float):
    # We assume that we receive the last layer's hidden states.
    loss = 0
    # When training with LC loss the last item in the batch will contain the full sequence
    # without MTP tokens as a target sequence for the hidden states. Therefore we will
    # iterate over all but the last items in the last layer's hidden state to compute the
    # LC loss.
    for hidden_state, offset in zip(hidden_states[:-1], offsets):
        hs = hidden_state[offset : offset + num_mtp]  # [T, H]
        target = hidden_states[-1, offset : offset + num_mtp].detach()
        loss += lc_loss_weight * F.mse_loss(hs, target)
    return loss / (len(hidden_states) - 1)


def calculate_tv_loss(logits, offsets, num_mtp: int, tv_loss_weight: float):
    # The TV loss (from DSpark) is similar to LCM loss but operates on the target distribution
    # instead of the latent space. The goal is to match the mask token's distribution to that
    # of the base model's output distribution.
    #
    # We achieve this similarly to the LCM loss implementation: the last item in the batch
    # contains all logits of the input sequence (i.e. the target distribution for all tokens).
    # This way we can now calculate each MTP token logit distribution and the distance to
    # the reference logits.
    loss = 0
    for sublogits, offset in zip(logits[:-1], offsets[:-1]):
        logits_pred = sublogits[offset : offset + num_mtp]  # [B, K, V]
        logits_true = logits[-1, offset : offset + num_mtp].detach()  # [B, K, V]
        # we use float32 values since the vocabulary can be huge and therefore
        # rounding errors might become more of a problem (e.g., non-discernable entries
        # in the distribution).
        probas_pred = F.softmax(logits_pred.float(), dim=-1)  # [B, K, V]
        probas_true = F.softmax(logits_true.float(), dim=-1)  # [B, K, V]
        # instead of using l1_loss() we're summing the differences so that we're not
        # dividing by K*V but only by K, otherwise the values would vanish since V is huge.
        loss += tv_loss_weight * (probas_pred - probas_true).abs().sum(dim=-1).mean()
    return loss / len(offsets[:-1])


def calculate_mtp_loss(logits, labels, offsets, use_lc_loss: bool, use_tv_loss: bool, num_mtp: int):
    """Calculates the cross-entropy loss exactly on the MTP tokens"""
    loss = 0
    logits = logits if not (use_lc_loss or use_tv_loss) else logits[:-1]
    for logit, label, offset in zip(logits, labels, offsets):
        # since the number of MTP tokens is equal for each sample, we can just sum them
        loss += F.cross_entropy(logit[offset : offset + num_mtp], label[offset : offset + num_mtp], ignore_index=-100)

    return loss / len(logits)


def calculate_sampler_loss(logits, labels, offsets, num_mtp: int):
    """Calculates the cross-entropy loss exactly on the MTP tokens for the sampler.

    This differs to calculate_mtp_loss in that we expect the logits sequence to be
    only as long as num_mtp but the labels sequence is the original labels sequence,
    so we need to get the correct labels from the MTP offsets.

    Note that we DO NOT drop the last batch of logits in case of `use_lc_loss`/`use_tv_loss`
    since this is supposed to be handled in the sampler. The returned logits are already
    skipping the last batch item in that case.
    """
    loss = 0
    for logit, label, offset in zip(logits, labels, offsets):
        # since the number of MTP tokens is equal for each sample, we can just sum them
        loss += F.cross_entropy(logit[0:num_mtp], label[offset : offset + num_mtp], ignore_index=-100)

    return loss / len(logits)


def calculate_mtp_accuracy(logits, labels, logit_offsets, label_offsets, num_mtp, use_lc_loss: bool, use_tv_loss: bool):
    """Calculate the accuracy for each MTP token position individually.

    To support both the base model and the sampler case this function takes two offset vectors.
    They can be the same (the base model case) or different (the sampler case). In the latter case
    the logit offsets are all zero, as the sampler only produces MTP tokens while the base model
    would produce len(inputs) + num_mtp tokens.
    """
    y_pred = []
    y_true = []
    logit_offsets = logit_offsets[:-1] if (use_lc_loss or use_tv_loss) else logit_offsets
    for idx_batch, (logit_offset, label_offset) in enumerate(zip(logit_offsets, label_offsets)):
        y_pred.append(logits[idx_batch, logit_offset : logit_offset + num_mtp].argmax(dim=-1))
        y_true.append(labels[idx_batch, label_offset : label_offset + num_mtp])
    y_pred = torch.stack(y_pred, dim=0)  # [B, K]
    y_true = torch.stack(y_true, dim=0)  # [B, K]
    token_accuracies = (y_pred == y_true).float().mean(dim=0)
    token_accuracies = token_accuracies.detach().cpu().tolist()
    return token_accuracies


def calculate_model_match(model, batch, labels, offsets, model_logits, num_mtp, use_lc_loss: bool, use_tv_loss: bool):
    """Calculate the rate of how many MTP tokens match the base model predictions, i.e. how many tokens would
    be accepted in a speculative decoding scenario.
    """
    match_rates = []
    offsets = offsets[:-1] if (use_lc_loss or use_tv_loss) else offsets

    for idx_batch, (input_ids, offset) in enumerate(zip(batch["input_ids"], offsets)):
        # auto-regressively create the reference tokens from the base model.
        #
        # for input index i the corresponding prediction will be for i+1. so the model logits for
        # <offset> (the first MTP token input idx) will predict position offset+1.
        #
        # 0  1  2  3      input idcs
        # a m1 m2 m3      input tokens
        # 1  2  3  4      label idcs
        # b  c  d  e      label tokens
        #
        # a  b  c  d  e   generated from input[:offset] (offset=1)
        #    1  2  3  4   respective indices in the generated output sequence
        #
        # the abbove shows why we generate num_mtp+1 tokens: we also have to include the
        # NTP prediction.
        #
        in_ref = input_ids[None, :offset].clone()
        y_ref = model.generate(
            input_ids=in_ref,
            attention_mask=torch.ones(*in_ref.shape).to(model.device),
            max_new_tokens=num_mtp + 1,
            do_sample=False,
            num_beams=1,
        )[0, -num_mtp:]

        y_mtp = model_logits[idx_batch, offset : offset + num_mtp].argmax(dim=-1)

        # the first mismatched index is also the number of matched tokens that are usable in a speculative decoding
        # setup (since after the first mismatch, the following tokens cannot be used). example:
        # ref:  [a, b, c, d, e]
        # mtp:  [a, b, d, f, e]
        # idx:  [0, 1, 2, 3, 4]
        #  !=:  [2, 3]
        # => 2 tokens were matched correctly of the 5 MTP candidates.
        match_indices = torch.where(y_ref != y_mtp)[0]

        if not len(match_indices):
            # no mismatch found
            match_rate = 1.0
        else:
            match_rate = match_indices[0].item() / num_mtp
        match_rates.append(match_rate)

    return sum(match_rates) / len(match_rates)


def setup_model_with_masks(model_id: str, num_mtp: int):
    """Load model and extend vocab with k mask tokens."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    # Add mask tokens to tokenizer
    mask_tokens = [f"<mask_{i}>" for i in range(1, num_mtp + 1)]
    special_tokens = {"additional_special_tokens": mask_tokens}
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} mask tokens")

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Resize model embeddings
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))

    # Get mask token IDs for ALoRA
    mask_token_ids = [tokenizer.convert_tokens_to_ids(f"<mask_{i}>") for i in range(1, num_mtp + 1)]
    print(f"Mask token IDs: {mask_token_ids}")

    return model, tokenizer, mask_token_ids


def save_artifacts(model, sampler, tokenizer, train_config, is_checkpoint: bool = False):
    if is_checkpoint:
        directory = train_config.checkpoint_dir
    else:
        directory = train_config.output_dir

    model.save_pretrained(directory)
    sampler_state_dict = sampler.state_dict()
    for key in list(sampler_state_dict.keys()):
        if key.startswith(("embedding.", "sampler.unembedding.")):
            del sampler_state_dict[key]
    safetensors.torch.save_file(
        tensors=sampler_state_dict,
        filename=os.path.join(directory, "sampler_model.safetensors"),
    )
    tokenizer.save_pretrained(directory)
    print(f"Saved the PEFT model and sampler at {directory}")


def train_loop(
    model, tokenizer, sampler, optimizer, iterator_train, iterator_eval, lr_scheduler, train_config: TrainConfig
):
    # assume batch size 1
    for step, sample in tqdm(enumerate(cycle(iterator_train)), desc="train"):
        if step == train_config.num_steps:
            break

        if isinstance(sample, dict):
            sample = [n.item() for n in sample["input_ids"]]  # finewiki dataloader must emit dicts, unpack that
        else:
            sample = sample[0].tolist()  # unpack batch of size 1, batching happens in augment_mtp

        output = train_step(
            model=model,
            tokenizer=tokenizer,
            sampler=sampler,
            optimizer=optimizer,
            iterator_train=iterator_train,
            lr_scheduler=lr_scheduler,
            train_config=train_config,
            sample=sample,
            step=step,
        )

        if step % train_config.log_step == 0:
            trackio.log(
                {
                    "train/loss/total": output["loss"],
                    "train/loss/mtp": output["loss_mtp"],
                    "train/loss/sampler": output["loss_sampler"],
                    "train/loss/lcm": output["loss_lc"],
                    "train/loss/tv": output["loss_tv"],
                    "train/lr/model": lr_scheduler.get_last_lr()[0],
                    "train/lr/sampler": lr_scheduler.get_last_lr()[1],
                    "train/step": step,
                    **{f"train/{k}": v for k, v in output["grad_norm_stats"].items()},
                }
            )

        if (step % train_config.eval_step == 0) and (step > 0):
            eval_output = evaluate(model, tokenizer, sampler, iterator_eval, train_config=train_config)
            trackio.log(
                {
                    "eval/loss/total": eval_output["loss"],
                    "eval/loss/mtp": eval_output["loss_mtp"],
                    "eval/loss/sampler": eval_output["loss_sampler"],
                    "eval/loss/lcm": eval_output["loss_lc"],
                    "eval/loss/tv": eval_output["loss_tv"],
                    "eval/step": step,
                    "eval/mtp_match_rate": eval_output["mtp_match_rate"],
                    **{
                        f"eval/accuracy/mtp_model_{i}": eval_output[f"mtp_model_accuracy_{i}"]
                        for i in range(train_config.num_mtp)
                    },
                    **{
                        f"eval/accuracy/mtp_sampler_{i}": eval_output[f"mtp_sampler_accuracy_{i}"]
                        for i in range(train_config.num_mtp)
                    },
                }
            )

        if step > 0 and train_config.checkpoint_step > 0 and step % train_config.checkpoint_step == 0:
            save_artifacts(model, sampler, tokenizer, train_config, is_checkpoint=True)

    save_artifacts(model, sampler, tokenizer, train_config, is_checkpoint=False)


def train_step(
    model, tokenizer, sampler, optimizer, iterator_train, lr_scheduler, train_config: TrainConfig, sample, step
):
    tic = time.perf_counter()
    # assume batch size 1
    input_ids, labels, offsets = augment_mtp(
        sample,
        bs=train_config.batch_size,
        num_mtp=train_config.num_mtp,
        use_tv_loss=train_config.use_tv_loss,
        use_lc_loss=train_config.use_lc_loss,
    )

    # create the batch
    tokens_per_sample = [len(i) for i in input_ids]
    total_tokens = sum(tokens_per_sample)
    batch = tokenizer.pad({"input_ids": input_ids}, return_tensors="pt").to(model.device)
    actual_batch_size = len(batch["input_ids"])

    # pad targets to padded input ids
    for i, (input_ids, label) in enumerate(zip(batch["input_ids"], labels)):
        labels[i] = labels[i] + [-100] * (input_ids.shape[0] - len(label))  # assume right padding
    labels = torch.tensor(labels).to(model.device)

    # train step
    optimizer.zero_grad()
    outputs = model(**batch, num_items_in_batch=total_tokens, output_hidden_states=True)
    loss_mtp = calculate_mtp_loss(outputs.logits, labels, offsets, train_config.use_lc_loss, train_config.use_tv_loss, train_config.num_mtp)
    if train_config.use_lc_loss:
        loss_lc = calculate_lc_loss(
            outputs.hidden_states[-1],
            offsets,
            num_mtp=train_config.num_mtp,
            lc_loss_weight=train_config.lc_loss_weight,
        )
    else:
        loss_lc = torch.tensor(0.0)
    if train_config.use_tv_loss:
        loss_tv = calculate_tv_loss(
            outputs.logits,
            offsets,
            num_mtp=train_config.num_mtp,
            tv_loss_weight=train_config.tv_loss_weight,
        )
    else:
        loss_tv = torch.tensor(0.0)
    if train_config.use_sampler:
        # Generate a num_mtp long sequence using the sampler and compute the MTP loss
        # for that (paper says: should be better than via the model itself).
        hidden_state = outputs.hidden_states[-1]
        if train_config.sampler_detach_hidden_state:
            hidden_state = hidden_state.detach()
        logits_sampler = sampler(outputs.logits, hidden_state, offsets)
        loss_sampler = (
            calculate_sampler_loss(
                logits=logits_sampler,
                labels=labels,
                offsets=offsets,
                num_mtp=train_config.num_mtp,
            )
            * train_config.sampler_loss_weight
        )
    else:
        loss_sampler = torch.tensor(0.0)

    if step % 100 == 0:
        grad_norm_stats = gradient_norm_step(optimizer, loss_mtp, loss_lc, loss_tv, loss_sampler)
    else:
        grad_norm_stats = {}

    loss = loss_mtp + loss_lc + loss_tv + loss_sampler
    loss.backward()

    if train_config.max_gradient_norm > 0:
        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"], train_config.max_gradient_norm)

    optimizer.step()
    lr_scheduler.step()

    toc = time.perf_counter()
    return {
        "loss": loss.detach().cpu().item(),
        "loss_mtp": loss_mtp.detach().cpu().item(),
        "loss_lc": loss_lc.detach().cpu().item(),
        "loss_tv": loss_tv.detach().cpu().item(),
        "loss_sampler": loss_sampler.detach().cpu().item(),
        "duration": toc - tic,
        "num_samples": actual_batch_size,
        "num_tokens": total_tokens,
        "grad_norm_stats": grad_norm_stats,
    }


@torch.inference_mode
def evaluate(model, tokenizer, sampler, iterator_eval, train_config):
    model_training = model.training
    sampler_training = sampler.training
    model.eval()
    sampler.eval()
    outputs = []
    for step, sample in tqdm(enumerate(iterator_eval), desc="eval"):
        if isinstance(sample, dict):
            sample = [n.item() for n in sample["input_ids"]]  # finewiki dataloader must emit dicts, unpack that
        else:
            sample = sample[0].tolist()  # unpack batch of size 1, batching happens in augment_mtp

        outputs.append(eval_step(model, tokenizer, sampler, train_config, sample, step))

    # we now have a list of metric dicts ([{'loss': ...,}, {'loss': ...,}, ...]) and we
    # want to take the average of every metric.
    outputs_aggregated = {}
    counts_aggregated = {}
    for metrics in outputs:
        for metric_key, value in metrics.items():
            # None values are used to signify skipped values that are not reported in every step
            if value is None:
                continue
            outputs_aggregated[metric_key] = outputs_aggregated.get(metric_key, 0) + value
            counts_aggregated[metric_key] = counts_aggregated.get(metric_key, 0) + 1
    for metric_key in outputs_aggregated:
        outputs_aggregated[metric_key] /= counts_aggregated[metric_key]

    model.train(model_training)
    sampler.train(sampler_training)
    return outputs_aggregated


def eval_step(model, tokenizer, sampler, train_config: TrainConfig, sample, step):
    tic = time.perf_counter()
    # assume batch size 1
    input_ids, labels, offsets = augment_mtp(
        sample,
        bs=train_config.batch_size,
        num_mtp=train_config.num_mtp,
        use_lc_loss=train_config.use_lc_loss,
        use_tv_loss=train_config.use_tv_loss,
    )

    # create the batch
    tokens_per_sample = [len(i) for i in input_ids]
    total_tokens = sum(tokens_per_sample)
    batch = tokenizer.pad({"input_ids": input_ids}, return_tensors="pt").to(model.device)

    # pad targets to padded input ids
    for i, (input_ids, label) in enumerate(zip(batch["input_ids"], labels)):
        labels[i] = labels[i] + [-100] * (input_ids.shape[0] - len(label))  # assume right padding
    labels = torch.tensor(labels).to(model.device)

    # train step
    outputs = model(**batch, num_items_in_batch=total_tokens, output_hidden_states=True)
    loss_mtp = calculate_mtp_loss(outputs.logits, labels, offsets, train_config.use_lc_loss, train_config.use_tv_loss, train_config.num_mtp)
    if train_config.use_lc_loss:
        loss_lc = calculate_lc_loss(
            outputs.hidden_states[-1],
            offsets,
            num_mtp=train_config.num_mtp,
            lc_loss_weight=train_config.lc_loss_weight,
        )
    else:
        loss_lc = torch.tensor(0.0)
    if train_config.use_tv_loss:
        loss_tv = calculate_tv_loss(
            outputs.logits,
            offsets,
            num_mtp=train_config.num_mtp,
            tv_loss_weight=train_config.tv_loss_weight,
        )
    else:
        loss_tv = torch.tensor(0.0)
    if train_config.use_sampler:
        # Generate a num_mtp long sequence using the sampler and compute the MTP loss
        # for that (paper says: should be better than via the model itself).
        logits_sampler = sampler(outputs.logits, outputs.hidden_states[-1], offsets)
        loss_sampler = (
            calculate_sampler_loss(
                logits=logits_sampler,
                labels=labels,
                offsets=offsets,
                num_mtp=train_config.num_mtp,
            )
            * train_config.sampler_loss_weight
        )
    else:
        loss_sampler = torch.tensor(0.0)

    # compute how many tokens would have been accepted on average. this is rather expensive so we
    # limit it to a subset of the samples.
    #
    # we also report NTP accuracy as a baseline for MTP, if we are significantly better than that, we're
    # probably overfitting the fine-tuning dataset. if we're way worse than that, we are not learning enough.
    if step % 50 == 0:
        mtp_match_rate = calculate_model_match(
            model,
            batch,
            labels,
            offsets,
            outputs.logits,
            train_config.num_mtp,
            train_config.use_lc_loss,
            train_config.use_tv_loss,
        )
    else:
        mtp_match_rate = None

    # we gather the top-1 accuracy for each mask token prediction individually
    token_accuracies_model = calculate_mtp_accuracy(
        outputs.logits,
        labels,
        logit_offsets=offsets,
        label_offsets=offsets,
        num_mtp=train_config.num_mtp,
        use_lc_loss=train_config.use_lc_loss,
        use_tv_loss=train_config.use_tv_loss,
    )

    if train_config.use_sampler:
        token_accuracies_sampler = calculate_mtp_accuracy(
            logits_sampler,
            labels,
            logit_offsets=[0] * len(offsets),  # no other tokens in the logits but MTP tokens
            label_offsets=offsets,
            num_mtp=train_config.num_mtp,
            use_lc_loss=train_config.use_lc_loss,
            use_tv_loss=train_config.use_tv_loss,
        )
    else:
        token_accuracies_sampler = [torch.nan] * train_config.num_mtp

    loss = loss_mtp + loss_lc + loss_tv + loss_sampler

    toc = time.perf_counter()
    return {
        "loss": loss.detach().cpu().item(),
        "loss_mtp": loss_mtp.detach().cpu().item(),
        "loss_lc": loss_lc.detach().cpu().item(),
        "loss_tv": loss_tv.detach().cpu().item(),
        "loss_sampler": loss_sampler.detach().cpu().item(),
        "duration": toc - tic,
        "num_tokens": total_tokens,
        "mtp_match_rate": mtp_match_rate,
        **{f"mtp_model_accuracy_{i}": token_accuracies_model[i] for i in range(train_config.num_mtp)},
        **{f"mtp_sampler_accuracy_{i}": token_accuracies_sampler[i] for i in range(train_config.num_mtp)},
    }


def gradient_norm_step(optimizer, loss_mtp, loss_lc, loss_tv, loss_sampler):
    """Compute the average gradient norm for each loss term so that we can see the impact of each
    loss term and the potential need for individual scaling.
    """

    def get_grad_norm(optimizer):
        grad_norms = []
        for group in optimizer.param_groups:
            norms = torch.tensor([p.grad.norm() for p in group["params"] if p.grad is not None])
            if len(norms) > 0:
                grad_norms.append(norms.mean())
        return sum(grad_norms) / len(grad_norms)

    loss_mtp.backward(retain_graph=True)
    grad_norm_mtp = get_grad_norm(optimizer)
    optimizer.zero_grad()

    if loss_lc > 0:
        loss_lc.backward(retain_graph=True)
        grad_norm_lc = get_grad_norm(optimizer)
        optimizer.zero_grad()
    else:
        grad_norm_lc = torch.tensor(0.0)

    if loss_tv > 0:
        loss_tv.backward(retain_graph=True)
        grad_norm_tv = get_grad_norm(optimizer)
        optimizer.zero_grad()
    else:
        grad_norm_tv = torch.tensor(0.0)

    if loss_sampler > 0:
        loss_sampler.backward(retain_graph=True)
        grad_norm_sampler = get_grad_norm(optimizer)
        optimizer.zero_grad()
    else:
        grad_norm_sampler = torch.tensor(0.0)

    gradnorms = {
        "gradnorm/mtp": grad_norm_mtp.cpu().item(),
        "gradnorm/lc": grad_norm_lc.cpu().item(),
        "gradnorm/tv": grad_norm_tv.cpu().item(),
        "gradnorm/sampler": grad_norm_sampler.cpu().item(),
    }

    return gradnorms


class TextDataset(Dataset):
    def __init__(self, text_data, tokenizer, seq_len):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.full_tokens = tokenizer.encode(text_data, add_special_tokens=False)

    def __len__(self):
        return len(self.full_tokens) // self.seq_len + 1

    def __getitem__(self, idx):
        return torch.tensor(
            [self.tokenizer.bos_token_id] + self.full_tokens[idx * self.seq_len : (idx + 1) * self.seq_len],
        )


def tokenize_wiki(examples, tokenizer, chunk_size):
    chunks = []
    for tokens in tokenizer.encode(examples["text"], add_special_tokens=False):
        for i_split in range(0, len(tokens), chunk_size):
            chunk = tokens[i_split : i_split + chunk_size]
            if len(chunk) == chunk_size:
                chunks.append([tokenizer.bos_token_id] + chunk)

    return {"input_ids": chunks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("-k", type=int, default=2)
    parser.add_argument("--text_file", type=str, default="train.txt")
    parser.add_argument("--use_wiki", action="store_true", default=False, help="Use finewiki instead of --text_file")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr_schedule", choices=["wsd", "cosine"], default="cosine")
    parser.add_argument("--num_steps", type=int, default=10_000, help="Training steps.")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Training steps for lr ramp-up")
    parser.add_argument("--decay_steps", type=int, default=None,
        help="Training steps for lr ramp-down (sqrt). Default: 20%% of total steps")
    parser.add_argument("--output_dir", type=str, default="mtp_model")
    parser.add_argument("--checkpoint_dir", type=str, default="mtp_model_checkpoint")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_sampler", action="store_true", default=False)
    parser.add_argument("--sampler_loss_weight", type=float, default=1.0)
    parser.add_argument("--sampler_detach_hidden_state", action="store_true", default=False)
    parser.add_argument("--sampler_use_rnn", action="store_true", default=False)
    parser.add_argument("--sampler_teacher_forcing", action="store_true", default=False)
    parser.add_argument("--sampler_lr", type=float)
    parser.add_argument("--lc_loss_weight", type=float, default=3.0)
    parser.add_argument("--tv_loss_weight", type=float, default=0.9)
    parser.add_argument("--use_lc_loss", action="store_true", default=False)
    parser.add_argument("--use_tv_loss", action="store_true", default=False)
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--eval_step", type=int, default=1000)  # eval is expensive, not too often
    parser.add_argument("--checkpoint_step", type=int, default=1000)
    parser.add_argument("--num_valid", type=int, default=200)
    parser.add_argument("--max_grad_norm", type=float, default=2)

    default_lr = 1e-4

    parser.set_defaults(
        sampler_lr=default_lr,
        lr=default_lr,
    )

    args = parser.parse_args()

    if args.decay_steps is None:
        args.decay_steps = int(args.num_steps * 0.2)

    if args.warmup_steps + args.decay_steps > args.num_steps:
        raise ValueError("There is no stable phase. Increase num_steps to be > warum_steps + decay_steps.")

    trackio.init(project="mtp-training")

    model, tokenizer, mask_token_ids = setup_model_with_masks(args.model_id, args.k)

    global MTP_MASK_TOKENS
    MTP_MASK_TOKENS = mask_token_ids

    # Setup ALoRA configuration
    # ALoRA activates LoRA for tokens after specific invocation tokens
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.r,
        lora_alpha=args.alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        alora_invocation_tokens=mask_token_ids,  # LoRA activates for mask tokens and after
        use_rslora=True,
        trainable_token_indices=mask_token_ids,
        ensure_weight_tying=True,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Setup sampler head
    hidden_size = model.config.hidden_size
    sampler = Sampler(
        hidden_size=hidden_size,
        num_mtp=args.k,
        skip_last=args.use_lc_loss or args.use_tv_loss,
        embedding=model.get_input_embeddings(),
        unembedding=model.get_output_embeddings(),
        recurrent=args.sampler_use_rnn,
    ).to(model.device, dtype=torch.bfloat16)

    # Create dataset
    if args.use_wiki:
        print("Loading text from finewiki")
        ds = load_dataset("HuggingFaceFW/finewiki", split="train", streaming=True)
        num_valid_samples = args.num_valid
        dataset_train = ds.skip(num_valid_samples).map(
            partial(tokenize_wiki, tokenizer=tokenizer, chunk_size=args.seq_len),
            batched=True,
            remove_columns=ds.column_names,
            drop_last_batch=True,
        )
        dataset_valid = ds.take(num_valid_samples).map(
            partial(tokenize_wiki, tokenizer=tokenizer, chunk_size=args.seq_len),
            batched=True,
            remove_columns=ds.column_names,
        )
        dataset_valid = dataset_valid.take(num_valid_samples)
    else:
        print(f"Loading text from: {args.text_file}")
        with open(args.text_file) as f:
            text = f.read()
        split_idx = args.num_valid
        dataset_train = TextDataset(text[split_idx:], tokenizer, args.seq_len)
        dataset_valid = TextDataset(text[:split_idx], tokenizer, args.seq_len)
        print(f"Train dataset size (sequences): {len(dataset_train)}")
        print(f"Valid dataset size (sequences): {len(dataset_valid)}")
    dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=False)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)

    # Optimizer: separate LR for sampler MLP (trains from scratch, may need higher LR)
    # Exclude unembedding from sampler params (it's shared with the model)
    sampler_params = [
        p for n, p in sampler.named_parameters() if not n.startswith(("sampler.unembedding", "embedding"))
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": sampler_params, "lr": args.sampler_lr},
        ]
    )

    # Trapezoid learning rate schedule where we have a linear ramp-up, a stable phase and a
    # square-root decay to 10% of the original LR. Hopefully this let's us observe the model
    # training behavior in a stable setting while also benefiting from a decay at the end.
    def wsd_lambda(step, *, total_steps, warmup_steps, decay_steps, floor_ratio=0.1):
         # 1) warmup: 0 -> 1
         if step < warmup_steps:
             return step / max(1, warmup_steps)

         stable_end = total_steps - decay_steps

         # 2) stable: hold peak
         if step < stable_end:
             return 1.0

         # 3) decay: 1 -> floor_ratio over the last decay_frac of steps
         progress = (step - stable_end) / max(1, decay_steps)   # 0 -> 1
         decay = 1 - math.sqrt(progress)
         return floor_ratio + (1 - floor_ratio) * decay

    if args.lr_schedule == "wsd":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=partial(
                wsd_lambda,
                total_steps=args.num_steps,
                warmup_steps=args.warmup_steps,
                decay_steps=args.decay_steps,
                floor_ratio=0.1,
            ),
        )
    elif args.lr_schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_steps
        )
    else:
        raise NotImplementedError

    print(f"LR schedule: warmup={args.warmup_steps} steps, sqrt decay over {args.decay_steps} steps")
    print(f"  model lr={args.lr}, sampler lr={args.sampler_lr} (x{args.warmup_steps}, x{args.decay_steps})")

    # Training loop
    print(f"Starting training for {args.num_steps} steps...")
    model.train()
    sampler.train()

    model.generation_config.pad_token_id = tokenizer.eos_token_id

    train_config = TrainConfig(
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        num_mtp=args.k,
        use_sampler=args.use_sampler,
        sampler_loss_weight=args.sampler_loss_weight,
        sampler_detach_hidden_state=args.sampler_detach_hidden_state,
        sampler_use_rnn=args.sampler_use_rnn,
        sampler_teacher_forcing=args.sampler_teacher_forcing,
        lc_loss_weight=args.lc_loss_weight,
        tv_loss_weight=args.tv_loss_weight,
        use_lc_loss=args.use_lc_loss,
        use_tv_loss=args.use_tv_loss,
        log_step=args.log_step,
        eval_step=args.eval_step,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_step=args.checkpoint_step,
        max_gradient_norm=args.max_grad_norm,
    )

    trackio.config.update({
        'r': args.r,
        'alpha': args.alpha,
        'lr': args.lr,
        'sampler_lr': args.sampler_lr,
        'seq_len': args.seq_len,
        'warmup_steps': args.warmup_steps,
        'decay_steps': args.decay_steps,
        **asdict(train_config),
    })

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        train_loop(
            model=model,
            tokenizer=tokenizer,
            sampler=sampler,
            optimizer=optimizer,
            iterator_train=dataloader_train,
            iterator_eval=dataloader_valid,
            lr_scheduler=scheduler,
            train_config=train_config,
        )
    except KeyboardInterrupt:
        pass

    trackio.finish()
    print("Training complete!")

    # Save model
    output_dir = args.output_dir
    model.save_pretrained(output_dir, save_embedding_layers=False)
    torch.save(sampler.state_dict(), f"{output_dir}/sampler.pt")
    tokenizer.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
