import torch
from torch import nn


class ArrowLoraLinearLayer(nn.Module):
    def __init__(self, in_features, ts_names, num_experts, top_k, temperature, use_gks, le_names):
        super().__init__()
        # extra parameters needed for arrow
        self.register_buffer("prototypes", torch.empty((num_experts, in_features)), persistent=False)
        self._protos_ready = False
        self.top_k = top_k
        self.temperature = temperature
        self.num_experts = num_experts
        self.use_gks = use_gks
        self.expert_names = ts_names.copy()
        self.le_names = le_names
        self.precomputed_W = None
        self.gks_done = False
        self.in_features = in_features

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

    def forward(self, x, lora_A, lora_B, dropout, scaling):
        """
        x : (B, *, in_features) prototypes : (E, in_features) ← built from top-SV of W = B @ A returns : LoRA delta
        with Arrow routing
        """
        self.prototypes = self.prototypes.to(x.dtype)
        B, *rest, F_in = x.shape
        tok = x.view(-1, F_in)  # (t, F_in)
        t, E = tok.size(0), self.prototypes.size(0)

        # 1) similarity   — sign-agnostic
        sim = torch.abs(tok @ self.prototypes.T)  # (t, E)

        # 2) top-k + softmax over full E (non-top-k = -inf)
        top_v, idx = torch.topk(sim, self.top_k, dim=1)
        full_score = tok.new_full((t, E), float("-inf"))
        full_score.scatter_(1, idx, top_v)
        coeff = torch.softmax(full_score / self.temperature, dim=1).to(x.dtype)  # (t, E)

        # 3) pre-compute W_e  (B @ A) once → gather
        if self.precomputed_W is None:
            Ws = []
            for name in self.expert_names:
                A = lora_A[name].weight  # (r, in)
                B = lora_B[name].weight  # (out, r)
                Ws.append(B @ A)  # (out, in)
            self.precomputed_W = torch.stack(Ws, dim=0)  # (E, out, in)
        W = self.precomputed_W.to(x.dtype)  # alias

        # 4) weighted sum:   Δ = Σ_e α_te · W_e · tok
        # coeff: (t, E)
        # W: (E, f_out, f_in)
        # tok: (t, f_in)
        # Substep 1: Compute all expert outputs (W_e @ tok) -> [E, t, f_out]
        W_tok = torch.einsum("eoi,ti->eto", W, tok)  # [E, t, f_out]

        # Substep 2: Weight by coefficients -> [t, f_out]
        delta = torch.einsum("te,eto->to", coeff, W_tok)  # [t, f_out]

        delta = scaling * dropout(delta).to(x.dtype)
        return delta.view(*x.shape[:-1], delta.size(-1))
