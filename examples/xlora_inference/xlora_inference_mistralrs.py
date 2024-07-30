"""
Perform inference of an X-LoRA model using the inference engine mistral.rs

This will allow vastly improved throughput, with dual KV cache and non-granular scalings.

Links:

- Installation: https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/README.md
- Runable example: https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/xlora_zephyr.py
- Adapter model docs and making the ordering file: https://github.com/EricLBuehler/mistral.rs/blob/master/docs/ADAPTER_MODELS.md
"""

from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.XLora(
        tok_model_id=None,  # Automatically determine from ordering file
        model_id=..., # Model ID of the base model (local path of HF model ID)
        xlora_model_id=..., # X-LoRA Model ID of the base model (local path of HF model ID)
        order=..., # Ordering file to ensure compatability with PEFT
        tgt_non_granular_index=3, # Only generate scalings for the first 3 decoding tokens, and then use the last generated one
    )
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="mistral",
        messages=[
            {"role": "user", "content": "Tell me a story about 2 low rank matrices."}
        ],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.5,
    )
)
print(res.choices[0].message.content)
print(res.usage)
