import torch
from torch import nn

from peft import LoHaConfig, get_peft_model


class TestDeterministicMerge:
    # Regression test for https://github.com/huggingface/peft/issues/3586:
    # merging LoHa/LoKr while the model is in train mode applied a *random*
    # rank_dropout mask inside get_delta_weight, making merge results
    # non-reproducible and breaking the merge->unmerge inverse property.

    def test_merge_is_deterministic_and_invertible_in_train_mode(self):
        torch.manual_seed(0)

        class Mlp(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin0 = nn.Linear(32, 32)

        cfg = LoHaConfig(r=4, alpha=8, target_modules=["lin0"], rank_dropout=0.5)
        model = get_peft_model(Mlp(), cfg)
        for n, p in model.named_parameters():
            if "hada" in n:  # LoHa zero-inits one factor; make the adapter non-trivial
                torch.nn.init.normal_(p, std=0.1)

        model.train()  # explicitly keep train mode, as after a Trainer run

        w0 = model.base_model.model.lin0.base_layer.weight.detach().clone()

        model.merge_adapter()
        w1 = model.base_model.model.lin0.base_layer.weight.detach().clone()
        model.unmerge_adapter()
        w_back = model.base_model.model.lin0.base_layer.weight.detach().clone()

        model.merge_adapter()
        w2 = model.base_model.model.lin0.base_layer.weight.detach().clone()
        model.unmerge_adapter()

        assert not torch.equal(w0, w1), "merge should still change weights"
        assert torch.equal(w1, w2), "two merges in train mode must produce identical weights"
        assert torch.allclose(w0, w_back), "unmerge must invert merge exactly"
