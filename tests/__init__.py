import os


if os.environ.get("PEFT_DEBUG_WITH_TORCH_COMPILE") == "1":
    # This is a hack purely for debugging purposes. If the environment variable PEFT_DEBUG_WITH_TORCH_COMPILE is set to
    # 1, get_peft_model() will return a compiled model. This way, all unit tests that use peft.get_peft_model() will
    # use a compiled model. See .github/workflows/torch_compile_tests.yml.
    import torch

    import peft
    from peft.mapping import get_peft_model as get_peft_model_original

    def get_peft_model_new(*args, **kwargs):
        """Make get_peft_model() return a compiled model."""
        peft_model = get_peft_model_original(*args, **kwargs)
        peft_model = torch.compile(peft_model)
        return peft_model

    peft.get_peft_model = get_peft_model_new
