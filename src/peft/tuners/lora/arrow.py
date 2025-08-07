import torch
from torch import nn


class ArrowLoraLinearLayer(nn.Module):
    def __init__(self, in_features, arrow_config):
        super().__init__()
        # extra parameters needed for arrow
        self._protos_ready = False
        self.top_k = arrow_config.arrow_top_k
        self.temperature = arrow_config.arrow_router_temperature
        self.expert_names = arrow_config.ts_names.copy()
        self.le_names = arrow_config.gen_names
        self.num_experts = len(self.expert_names)
        self.register_buffer("prototypes", torch.empty((self.num_experts, in_features)), persistent=False)
        self.use_gks = arrow_config.use_gks
        # self.precomputed_W = None
        self.gks_done = False
        self.in_features = in_features
        self.cast_input_dtype_enabled = True

    @torch.no_grad()
    def build_prototypes(self, lora_A, lora_B=None):
        if self._protos_ready:
            return
        protos = []
        for name in self.expert_names:
            A = lora_A[name].weight  # (r, in_features)
            # SVD on A (tiny: r≪in_features)
            _, _, Vh = torch.linalg.svd(A.float(), full_matrices=False)
            protos.append(Vh[0].to(A.dtype))  # (in_features,)
        self.prototypes[:] = torch.stack(protos, dim=0)  # (E, in_features)
        self._protos_ready = True

    @torch.no_grad()
    def gen_know_sub(self, lora_A, lora_B):
        if self.gks_done or not self.use_gks:
            return
        else:
            # 1) compute average A/B over le_names
            avg_A = torch.stack([lora_A[n].weight for n in self.le_names], dim=0).mean(0)  # shape (r, in_features)
            avg_B = torch.stack([lora_B[n].weight for n in self.le_names], dim=0).mean(0)  # shape (out_features, r)

            # 2) Subtract the average from task-specific experts
            for name in self.expert_names:
                lora_A[name].weight.data.sub_(avg_A)
                lora_B[name].weight.data.sub_(avg_B)

            # 3) now delete the original LE adapters
            for n in self.le_names:
                del lora_A[n]
                del lora_B[n]

            # # 4) register the avg as a new adapter at position num_experts
            # avg_name = "gks_avg"
            # device = self.prototypes.device
            # A_mod = nn.Linear(avg_A.size(1), avg_A.size(0), bias=False).to(device)
            # B_mod = nn.Linear(avg_B.size(1), avg_B.size(0), bias=False).to(device)

            # A_mod.weight.data.copy_(avg_A.to(device))
            # B_mod.weight.data.copy_(avg_B.to(device))

            # lora_A[avg_name] = A_mod
            # lora_B[avg_name] = B_mod
            # self.expert_names.append(avg_name)

            # # 5) reset caches
            # E = len(self.expert_names)
            # new_protos = torch.empty((E, self.in_features), device=self.prototypes.device, dtype=self.prototypes.dtype)
            # # this replaces the old buffer in-place with newer size
            # self.prototypes = new_protos
            # self._protos_ready = False
            # self.precomputed_W = None

            # print(lora_A)
            # print(lora_B)
            self.gks_done = True

    def _cast_input_dtype(self, x, dtype: torch.dtype):
        """
        Whether to cast the dtype of the input of the forward method.

        Usually, we want to enable this to align the input dtype with the dtype of the weight, but by setting
        layer.cast_input_dtype=False, this can be disabled if necessary.

        Enabling or disabling can be managed via the peft.helpers.disable_lora_input_dtype_casting context manager.
        """
        if x is None:  # useful e.g. if x is the bias, which can be None
            return None

        cast_input_dtype_enabled = getattr(self, "cast_input_dtype_enabled", True)
        if (not cast_input_dtype_enabled) or (x.dtype == dtype):
            return x
        return x.to(dtype=dtype)

    def forward(self, x, lora_A, lora_B, dropout, scaling):
        """
        x : (B, *, in_features) prototypes : (E, in_features) ← built from top-SV of W = B @ A returns : LoRA delta
        with Arrow routing
        """
        # self.prototypes = self.prototypes.to(x.dtype)
        x = self._cast_input_dtype(x, lora_A[self.expert_names[0]].weight.dtype)
        B, *rest, F_in = x.shape
        tok = x.view(-1, F_in)  # (t, F_in)
        t, E = tok.size(0), self.prototypes.size(0)

        # 1) similarity   — sign-agnostic
        sim = torch.abs(tok @ self.prototypes.T)  # (t, E)

        # 2) top-k + softmax over full E (non-top-k = -inf)
        top_v, idx = torch.topk(sim, self.top_k, dim=1)
        full_score = tok.new_full((t, E), float("-inf"))
        full_score.scatter_(1, idx, top_v)
        coeff = torch.softmax(full_score / self.temperature, dim=1)  # (t, E)

        # 3) stack all A and B weights once
        #   A_stack: (E, r, in_features), B_stack: (E, out_features, r)
        A_stack = torch.stack([lora_A[n].weight for n in self.expert_names], dim=0)
        B_stack = torch.stack([lora_B[n].weight for n in self.expert_names], dim=0)

        # 4) project tokens into each expert’s low‑rank space:
        #    z[e] = tok @ A_e.T   → shape (t, E, r)
        z = torch.einsum("tf, erf -> ter", tok, A_stack)

        # 5) lift back each expert’s output:
        #    y[e] = z[e] @ B_e.T  → shape (t, E, out_features)
        y = torch.einsum("ter, eor -> teo", z, B_stack)

        # 6) weighted sum over experts:
        #    delta_flat[t,o] = Σ_e coeff[t,e] * y[t,e,o]
        delta_flat = torch.einsum("te, teo -> to", coeff, y)  # (t, out_features)

        # 7) dropout, scale, and reshape
        delta = scaling * dropout(delta_flat)
        out_dim = delta_flat.size(-1)
        return delta.view(B, *rest, out_dim)
