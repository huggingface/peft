# X-LoRA examples

## `xlora_inference_mistralrs.py`

Perform inference of an X-LoRA model using the inference engine mistral.rs.

Mistral.rs supports many base models besides Mistral, and can load models directly from saved LoRA checkpoints. Check out [adapter model docs](https://github.com/EricLBuehler/mistral.rs/blob/master/docs/ADAPTER_MODELS.md) and the [models support matrix](https://github.com/EricLBuehler/mistral.rs?tab=readme-ov-file#support-matrix).

Mistral.rs features X-LoRA support and incorporates techniques such as a dual-KV cache, continuous batching, Paged Attention, and optional non granular scalings, will allow vastly improved throughput.    

Links:

- Installation: https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/README.md
- Runnable example: https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/xlora_zephyr.py
- Adapter model docs and making the ordering file: https://github.com/EricLBuehler/mistral.rs/blob/master/docs/ADAPTER_MODELS.md