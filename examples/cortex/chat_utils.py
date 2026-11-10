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

"""Message rendering and response parsing shared by the Cortex examples."""

import re
from collections.abc import Sequence


def render_messages(
    tokenizer,
    messages: list[dict],
    model_type: str,
    *,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
    reasoning_effort: str = "medium",
    include_reasoning: bool = True,
) -> list[int]:
    """Render canonical messages, mapping reasoning to the model's native field.

    ``reasoning_content`` is the portable field in our JSONL. GPT-OSS instead
    expects ``thinking``. The input messages are never modified. Thinking in
    an assistant's final training turn is supported for Qwen3 and GPT-OSS;
    other architectures still support ordinary messages without reasoning.
    """
    if not tokenizer.chat_template:
        raise ValueError("Choose an instruction model with a chat template.")
    if enable_thinking and model_type not in {"qwen3", "gpt_oss"}:
        raise ValueError(f"Thinking message rendering is not supported for {model_type!r}.")
    native = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise TypeError("Each message needs role and string content fields.")
        if message.get("role") not in {"system", "developer", "user", "assistant"}:
            raise ValueError("This example supports system, developer, user, and assistant messages.")
        if "think" in message or "thinking" in message:
            raise ValueError("Use the canonical reasoning_content field in input messages.")
        item = dict(message)
        reasoning = item.pop("reasoning_content", None)
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("reasoning_content must be a string when provided.")
        if include_reasoning and reasoning is not None:
            if item["role"] != "assistant":
                raise ValueError("Only assistant messages may contain reasoning_content.")
            if model_type not in {"qwen3", "gpt_oss"}:
                if reasoning:
                    raise ValueError(f"Reasoning message rendering is not supported for {model_type!r}.")
            else:
                field = "thinking" if model_type == "gpt_oss" else "reasoning_content"
                item[field] = reasoning
        if item["role"] == "assistant" and model_type == "qwen3":
            # An absent field makes Qwen's template search content for </think>;
            # an explicit empty value keeps tags inside final code as literals.
            item.setdefault("reasoning_content", "")
        native.append(item)
    kwargs = {}
    if model_type == "qwen3":
        kwargs["enable_thinking"] = enable_thinking
    elif model_type == "gpt_oss":
        if reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("GPT-OSS reasoning_effort must be low, medium, or high.")
        kwargs["reasoning_effort"] = reasoning_effort
        # Match the research prompts without editing the checkpoint's template.
        # This per-call override must also be used after reloading the tokenizer.
        kwargs["strftime_now"] = lambda _: "xxxx-xx-xx"
    ids = tokenizer.apply_chat_template(
        native, tokenize=True, return_dict=False, add_generation_prompt=add_generation_prompt, **kwargs
    )
    if model_type == "qwen3" and add_generation_prompt and not enable_thinking:
        prefix = tokenizer.decode(ids, skip_special_tokens=False)
        if prefix.rstrip().endswith("<think>"):
            raise ValueError(
                "This tokenizer's chat template always enables thinking; enable_thinking=False cannot disable it. "
                "Use thinking generation with this checkpoint."
            )
    return ids


def tokenize_assistant_turn(
    tokenizer,
    messages: list[dict],
    model_type: str,
    *,
    include_reasoning: bool = True,
    reasoning_effort: str = "medium",
) -> tuple[list[int], int]:
    """Render a complete turn and return its assistant-only supervision boundary.

    An empty Qwen thinking block changes the tokenization of the newline after
    ``<think>``. Keep the full native rendering and mask its common token prefix
    with the generation prompt, checking that every prompt message stays masked.
    """
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("Training messages must contain a prompt followed by an assistant turn.")
    options = {
        "enable_thinking": model_type == "qwen3",
        "include_reasoning": include_reasoning,
        "reasoning_effort": reasoning_effort,
    }
    ids = render_messages(tokenizer, messages, model_type, **options)
    prompt = render_messages(tokenizer, messages[:-1], model_type, **options)
    prefix = render_messages(tokenizer, messages[:-1], model_type, add_generation_prompt=True, **options)
    boundary = 0
    for actual, expected in zip(ids, prefix):
        if actual != expected:
            break
        boundary += 1
    if ids[: len(prompt)] != prompt or boundary < len(prompt):
        raise ValueError("The chat template changes prompt tokens when adding the assistant turn.")
    if boundary == len(ids):
        raise ValueError("The rendered conversation has no assistant tokens to supervise.")
    return ids, boundary


def final_answer_prefix(
    tokenizer,
    messages: list[dict],
    model_type: str,
    *,
    reasoning_effort: str = "medium",
) -> list[int]:
    """Build a reference-answer scoring prefix without an unknown reasoning trace.

    Qwen receives the native empty thinking block, explicitly closed before the
    reference answer. GPT-OSS receives its native final channel. This is a
    teacher-forced scoring condition, not a claim that thinking can be disabled.
    """
    if model_type not in {"qwen3", "gpt_oss"}:
        return render_messages(tokenizer, messages, model_type, add_generation_prompt=True)
    assistant = {"role": "assistant", "content": "", "reasoning_content": ""}
    ids = render_messages(tokenizer, [*messages, assistant], model_type, reasoning_effort=reasoning_effort)
    eos = _eos_ids(tokenizer)
    end = next((index for index in range(len(ids) - 1, -1, -1) if ids[index] in eos), None)
    prompt = render_messages(tokenizer, messages, model_type, reasoning_effort=reasoning_effort)
    if end is None or end <= len(prompt) or tokenizer.decode(ids[end + 1 :], skip_special_tokens=False).strip():
        raise ValueError("The chat template did not render an empty assistant turn with a terminal EOS token.")
    return ids[:end]


def _eos_ids(tokenizer, eos_token_ids: int | Sequence[int] | None = None) -> set[int]:
    eos = tokenizer.eos_token_id if eos_token_ids is None else eos_token_ids
    if eos is None:
        return set()
    return {eos} if isinstance(eos, int) else set(eos)


def _parse_qwen_response(text: str, prefix: str) -> tuple[dict, bool]:
    response = {"role": "assistant", "content": ""}
    text = text.strip()
    opened_in_prefix = prefix.rstrip().endswith("<think>")
    if text.startswith("<think>"):
        text = text[len("<think>") :]
        opened_in_prefix = True
    if opened_in_prefix:
        reasoning, closing, final = text.partition("</think>")
        response["reasoning_content"] = reasoning.strip()
        if not closing:
            return response, False
        final = final.strip()
        if final.startswith(("<think>", "</think>")):
            return response, False
        response["content"] = final
    elif text.startswith("</think>"):
        # A closing delimiter without its opening/prefix is a malformed response.
        return response, False
    else:
        response["content"] = text
    return response, bool(response["content"])


def _parse_harmony_response(text: str, prefix: str) -> tuple[dict, bool]:
    response = {"role": "assistant", "content": ""}
    # Both spellings occur in GPT-OSS checkpoints. Keep the final body intact;
    # only the outer assistant/channel headers are interpreted as protocol.
    for start, channel, message, end in (
        ("<|start|>", "<|channel|>", "<|message|>", "<|end|>"),
        ("<|im_start|>", "<|meta_sep|>", "<|im_sep|>", "<|im_end|>"),
    ):
        header = start + "assistant"
        full = text.lstrip()
        if not full.startswith(header):
            offset = prefix.rfind(header)
            if offset >= 0:
                full = prefix[offset:] + text
            elif full.startswith(channel):
                full = header + full
            else:
                continue
        pattern = re.escape(header + channel) + r"(analysis|final)" + re.escape(message)
        match = re.match(pattern, full)
        if match is None:
            continue
        body = full[match.end() :]
        if match[1] == "final":
            response["content"] = body.strip()
            return response, bool(response["content"])
        transition = end + header + channel + "final" + message
        reasoning, boundary, final = body.partition(transition)
        response["reasoning_content"] = reasoning.strip()
        if not boundary:
            return response, False
        response["content"] = final.strip()
        return response, bool(response["content"])
    return response, False


def parse_response(
    tokenizer,
    response_ids: Sequence[int],
    model_type: str,
    *,
    prefix_ids: Sequence[int] | None = None,
    enable_thinking: bool = False,
    eos_token_ids: int | Sequence[int] | None = None,
) -> tuple[dict, bool]:
    """Return a canonical assistant message and whether a final answer exists.

    This parsed text is for portable SFT records and evaluation. Distillation
    must retain the original response IDs, not re-tokenize these strings.
    Generation truncation is checked separately by ``generation_status``.
    """
    ids = list(response_ids)
    if ids and ids[-1] in _eos_ids(tokenizer, eos_token_ids):
        ids.pop()
    text = tokenizer.decode(ids, skip_special_tokens=False)
    prefix = tokenizer.decode(list(prefix_ids or []), skip_special_tokens=False)
    if model_type == "qwen3":
        return _parse_qwen_response(text, prefix)
    if model_type == "gpt_oss":
        return _parse_harmony_response(text, prefix)
    if enable_thinking:
        raise ValueError(f"Thinking response parsing is not supported for {model_type!r}.")
    # Ordinary tokenizers may produce BOS or other special tokens. Removing
    # those here matches the plain text example's original decoding behavior.
    final = tokenizer.decode(ids, skip_special_tokens=True).strip()
    return {"role": "assistant", "content": final}, bool(final)


def generation_status(
    tokenizer,
    response_ids: Sequence[int],
    max_new_tokens: int,
    *,
    complete: bool,
    eos_token_ids: int | Sequence[int] | None = None,
) -> dict:
    """Describe termination without mistaking an EOS at the budget for truncation."""
    ids = list(response_ids)
    if ids and ids[-1] in _eos_ids(tokenizer, eos_token_ids):
        reason = "stop"
    elif len(ids) >= max_new_tokens:
        reason = "length"
    else:
        reason = "unknown"
    return {"finish_reason": reason, "complete": complete and reason == "stop"}


def generation_kwargs(enable_thinking: bool) -> dict:
    """Use the Qwen3 thinking recipe; ordinary examples retain greedy decoding."""
    if enable_thinking:
        return {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
    return {}
