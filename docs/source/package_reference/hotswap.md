<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Hotswapping adapters

The idea of hotswapping an adapter is the following: We can already load multiple adapters, e.g. two LoRAs, at the same time. But sometimes, we want to load one LoRA and then replace its weights in-place with the LoRA weights of another adapter. This is now possible the hotswap_adapter function.

In general, this should be faster than deleting one adapter and loading the adapter in its place, which would be the how to achieve the same final outcome without hotswapping. Another advantage of hotswapping is that it prevents re-compilation in case the PEFT model is already compiled using `torch.compile`. This can save quite a lot of time.

There are some caveats for hotswapping:

- It only works for the same PEFT method, so no swapping LoRA and LoHa.
- Right now, only LoRA is properly supported.
- The adapters must be compatible (e.g. same LoRA alpha, same target modules).

[[autodoc]] utils.hotswap.hotswap_adapter
    - all

[[autodoc]] utils.hotswap.hotswap_adapter_from_state_dict
    - all
