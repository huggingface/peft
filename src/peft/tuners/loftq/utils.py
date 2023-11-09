import os
import torch
from torch import Tensor
import math
import random
from torch import nn
import torch.nn.functional as F
from scipy.stats import norm
from torch import optim


def low_rank_decomposition(weight, reduced_rank=32):
    """
    :param          weight: The matrix to decompose, of shape (H, W)
    :param    reduced_rank: the final rank
    :return:
    """

    """parameter_ratio = rank * (H + W) / (H * W)"""
    """rank_ratio = """
    matrix_dimension = len(weight.size())
    assert matrix_dimension == 2, "Only Support 2D matrix"
    H, W = weight.size()

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    rank = torch.count_nonzero(S)
    is_full_rank = rank == min(H, W)

    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh

    # print(f"W: ({H},{W}) | Rank: {rank} | U:{U.shape} | S:{S.shape} | Vh:{Vh.shape}")
    # print(f"Reduced Rank: {reduced_rank} | Num Parameters: {(H + W) * reduced_rank}")
    # print(f"L: {L.shape} | R: {R.shape}")

    return {"L": L, "R": R, "U": U, "S": S, "Vh": Vh, 'reduced_rank': reduced_rank}


class NFQuantizer:
    def __init__(self, num_bits=2, device='cuda', method='normal', block_size=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bits = num_bits
        self.device = device
        self.method = method
        self.block_size = block_size
        if self.method == 'normal':
            self.norm_lookup_table = self.create_normal_map(num_bits=self.num_bits)
            self.norm_lookup_table = self.norm_lookup_table.to(device)
        elif self.method == 'uniform':
            self.norm_lookup_table = self.create_uniform_map(num_bits=self.num_bits)
            self.norm_lookup_table = self.norm_lookup_table.to(device)
        else:
            raise NotImplementedError("Other quantization methods not supported yet.")

    @staticmethod
    def create_uniform_map(symmetric=False, num_bits=4):
        if symmetric:
            # print("symmetric uniform quantization")
            negative = torch.linspace(-1, 0, 2 ** (num_bits - 1))
            positive = torch.linspace(0, 1, 2 ** (num_bits - 1))
            table = torch.cat([negative, positive[1:]])
        else:
            # print("asymmetric uniform quantization")
            table = torch.linspace(-1, 1, 2 ** num_bits)
        return table

    @staticmethod
    def create_normal_map(offset=0.9677083, symmetric=False, num_bits=2):
        variations = 2 ** num_bits

        if symmetric:
            # print("symmetric NormalFloat")
            v = norm.ppf(torch.linspace(1 - offset, offset, variations + 1)).tolist()
            values = []
            for index in range(len(v) - 1):
                values.append(0.5 * v[index] + 0.5 * v[index + 1])
            v = values
        else:
            # one more positive value, this is an asymmetric type
            # print("asymmetric NormalFloat")
            v1 = norm.ppf(torch.linspace(offset, 0.5, variations // 2 + 1)[:-1]).tolist()
            # print(torch.linspace(offset, 0.5, 9)[:-1])
            # print(v1)
            v2 = [0]
            # v2 = [0]*(256-15) ## we have 15 non-zero values in this data type
            v3 = (-norm.ppf(torch.linspace(offset, 0.5, variations // 2)[:-1])).tolist()
            # print(torch.linspace(offset, 0.5, 8)[:-1])
            # print(v3)
            v = v1 + v2 + v3

        values = torch.Tensor(v)
        values = values.sort().values
        values /= values.max()
        # print(values)
        return values
        # assert values.

    def quantize_tensor(self, weight):
        max_abs = torch.abs(weight).max()
        weight_normed = weight / max_abs

        weight_normed_expanded = weight_normed.unsqueeze(-1)

        # Reshape L to have the same number of dimensions as X_expanded
        L_reshaped = torch.tensor(self.norm_lookup_table).reshape(1, -1)

        # Calculate the absolute difference between X_expanded and L_reshaped
        abs_diff = torch.abs(weight_normed_expanded - L_reshaped)

        # Find the index of the minimum absolute difference for each element
        qweight = torch.argmin(abs_diff, dim=-1)
        # print(min_index)
        return qweight, max_abs

    def dequantize_tensor(self, qweight, max_abs):
        qweight_flatten = qweight.flatten()

        weight_normed = self.norm_lookup_table[qweight_flatten]
        weight = weight_normed * max_abs

        weight = weight.reshape(qweight.shape)

        return weight

    def quantize_block(self, weight):
        assert len(weight.shape) == 2 and weight.shape[0] * weight.shape[1] % self.block_size == 0
        M, N = weight.shape
        device = weight.device

        # Quantization
        weight_flatten = weight.flatten()  # (M*N, )
        weight_block = weight_flatten.reshape(-1, self.block_size)  # (L, B), L = M * N / B
        if self.method == 'normal':
            weight_max = weight_block.abs().max(dim=-1)[0]  # (L, 1)
        elif self.method == 'uniform':
            weight_max = weight_block.mean(dim=-1) + 2.5 * weight_block.std(dim=-1)
        else:
            raise NotImplementedError("Method not supported yet.")
        weight_max = weight_max.unsqueeze(-1)
        weight_divabs = weight_block / weight_max  # (L, B)
        weight_divabs = weight_divabs.unsqueeze(-1)  # (L, B, 1)
        L_reshaped = self.norm_lookup_table.reshape(1, -1)  # (1, 2**K)

        abs_diff = torch.abs(weight_divabs - L_reshaped)  # (L, B, 2**K)
        qweight = torch.argmin(abs_diff, dim=-1)  # (L, B)

        # Pack multiple k-bit into uint8
        qweight = qweight.reshape(-1, 8 // self.num_bits)
        qweight_pack = torch.zeros((M * N // 8 * self.num_bits, 1), dtype=torch.uint8, device=device)

        # data format example:
        # [1, 0, 3, 2] or [01, 00, 11, 10]  -> [10110001], LIFO
        for i in range(8 // self.num_bits):
            qweight[:, i] = qweight[:, i] << i * self.num_bits
            qweight_pack[:, 0] |= qweight[:, i]

        return qweight_pack, weight_max, weight.shape

    def dequantize_block(self, qweight, weight_max, weight_shape):
        # unpack weight
        device = qweight.device
        weight = torch.zeros((qweight.shape[0], 8 // self.num_bits), dtype=torch.float32, device=device)
        for i in range(8 // self.num_bits):
            lookup_table_idx = qweight.to(torch.long) % 2 ** self.num_bits  # get the most right 2 bits
            lookup_table_idx = lookup_table_idx.to(torch.int)
            weight[:, i] = self.norm_lookup_table[lookup_table_idx].squeeze()
            qweight = qweight >> self.num_bits  # right shift 2 bits of the original data

        weight_block = weight.reshape(-1, self.block_size)
        weight = weight_block * weight_max
        weight = weight.reshape(weight_shape)

        return weight


class QLinearLR(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 reduced_rank: int,
                 num_bits: int,
                 block_size=64,
                 enable_lora=True,
                 bias=None,
                 device='cuda',
                 method='normal'
                 ):
        super().__init__()
        self.num_bits = num_bits
        self.enable_lora = enable_lora
        self.method = method
        self.block_size = block_size

        self.quantizer = NFQuantizer(num_bits=num_bits, method=method, device=device, block_size=block_size)

        self.register_buffer('qweight', torch.empty((in_features * out_features // 8 * num_bits, 1), dtype=torch.uint8,
                                                    device=device))
        self.register_buffer('absmax', torch.empty((in_features * out_features // block_size, 1), dtype=torch.float32,
                                                   device=device))
        self.lora_A = nn.Parameter(torch.empty((reduced_rank, in_features), dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.empty((out_features, reduced_rank), dtype=torch.float32, device=device))

        self.bias = bias

        self.weight_size = torch.Size([out_features, in_features])
        self.weight_type = torch.float32

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.quantizer.dequantize_block(self.qweight, self.absmax, self.weight_size)
        dense = input @ weight.T
        lora = (input @ self.lora_A.T) @ self.lora_B.T if self.enable_lora else 0

        return dense + lora + self.bias if self.bias is not None else dense + lora

    def initial_backbone(self, weight):
        self.qweight, self.absmax, _ = self.quantizer.quantize_block(weight)

    def initial_lora(self, lora_A, lora_B):
        self.lora_A.data = lora_A
        self.lora_B.data = lora_B


def replace_module(
        module,
        prename='model',
        allow_name=None,
        block_name=None,
        reduced_rank=32,
        num_bits=4,
        num_iter=5,
        enable_lora=True,
        num_layers=32,
        empty_init=True,
        quant_method='normal',
        fake_quant=True
):
    """
    :param       module: have to inherit nn.Module
    :param      prename: previous name, used to iteratively obtain parameters name
    :param   allow_name: allowed nn.Linear to quantize
    :param   block_name: blocked nn.Linear to quantize
    :param reduced_rank: low-rank rank
    :param     num_bits: low-precision bits. 2,4,8 as expected, float number between (2, 4) enables mixed precision
    :param     num_iter: alternating steps
    :param  enable_lora: whether enable lora part in forward pass
    :param   num_layers: total number of layers. can be obtained by the model config file
    :param   empty_init: True for the first time decomposition, False for loading model from checkpoints
    :param quant_method: choose in ['normal', 'uniform'], other quantization method not supported
    :param   fake_quant: True for fake quantization where values change but memory not saved; False for real quant
    :return:
    """
    # Default allow name and block name lists
    if allow_name is None:
        allow_name = ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h',
                      'q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
    if block_name is None:
        block_name = ['pooler', 'classifier', 'LayerNorm', 'embeddings', 'lora']

    # mixed precision: first k layers use 4-bit weights and the rest use 2-bit weights
    bit4_layer = int((num_layers * (num_bits - 2)) // 2)
    if num_bits not in [2, 4, 8]:
        print("Warning: Only support decoder-only or encoder-only models. "
              "Apply to encoder-decoder models on your own risk.")

    allow_module = [nn.Linear]

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        weight_name = prename + '.' + attr_str

        if any(isinstance(target_attr, module) for module in allow_module) and any(an in attr_str for an in allow_name):
            print("====================================================")
            print(weight_name, target_attr)
            print(dir(target_attr))

            # determine the true bit for this specific matrix
            num_bits_ = 4 if any(f".{i}." in weight_name for i in range(bit4_layer)) else 2
            num_bits_ = 8 if num_bits == 8 else num_bits_

            if not empty_init:  # decomposition mode
                weight, lora_A, lora_B = loftq_init(weight=target_attr.weight,
                                                    num_bits=num_bits_,
                                                    num_iter=num_iter,
                                                    reduced_rank=reduced_rank,
                                                    method=quant_method,
                                                    block_size=64)
                if fake_quant:  # require input model to have peft_lora already
                    target_attr.weight.data = weight
                    target_attr.lora_A['default'].weight.data = lora_A
                    target_attr.lora_B['default'].weight.data = lora_B
                    torch.cuda.empty_cache()
                else:  # require input model not to have peft_lora
                    qlinear_lora = QLinearLR(target_attr.in_features,
                                             target_attr.out_features,
                                             reduced_rank,
                                             num_bits_,
                                             block_size=64,
                                             enable_lora=enable_lora,
                                             bias=target_attr.bias,
                                             device='cuda' if torch.cuda.is_available() else 'cpu',
                                             method=quant_method)
                    qlinear_lora.initial_backbone(weight)
                    qlinear_lora.initial_lora(lora_A, lora_B)

                    delattr(module, attr_str)
                    torch.cuda.empty_cache()
                    setattr(module, attr_str, qlinear_lora)

            else:  # when loading real quantized weights
                qlinear_lora = QLinearLR(target_attr.in_features,
                                         target_attr.out_features,
                                         reduced_rank,
                                         num_bits_,
                                         block_size=64,
                                         enable_lora=enable_lora,
                                         bias=target_attr.bias,
                                         device='meta',  # use meta to avoid out of memory
                                         method=quant_method)

                delattr(module, attr_str)
                torch.cuda.empty_cache()
                setattr(module, attr_str, qlinear_lora)

    for name, immediate_child_module in module.named_children():
        # do not continue to iterate when the module's name is in the block_name
        if not any(name in bn for bn in block_name):
            replace_module(immediate_child_module,
                           prename=prename + '.' + name,
                           allow_name=allow_name,
                           block_name=block_name,
                           reduced_rank=reduced_rank,
                           num_bits=num_bits,
                           num_iter=num_iter,
                           enable_lora=enable_lora,
                           num_layers=num_layers,
                           empty_init=empty_init,
                           quant_method=quant_method,
                           fake_quant=fake_quant,
                           )


def loftq_init(weight, num_bits: int, reduced_rank: int, num_iter: int, method='normal', block_size=64):
    out_feature, in_feature = weight.size()
    device = weight.device
    print(f"Weight: ({out_feature}, {in_feature}) | Rank: {reduced_rank} | Num Iter: {num_iter} | Num Bits: {num_bits}")

    quantizer = NFQuantizer(num_bits=num_bits, device=device, method=method, block_size=block_size)
    res = weight.clone()
    for i in range(num_iter):
        # Quantization
        quantized_weight, max_abs, shape = quantizer.quantize_block(res)
        dequantized_weight = quantizer.dequantize_block(quantized_weight, max_abs, shape)
        res = weight - dequantized_weight

        # Decompose the residual by SVD
        output = low_rank_decomposition(res, reduced_rank=reduced_rank)
        L, R, reduced_rank = output['L'], output['R'], output['reduced_rank']
        res = weight - torch.mm(L, R)

    if num_iter == 0:
        quantized_weight, max_abs, shape = quantizer.quantize_block(res)
        dequantized_weight = quantizer.dequantize_block(quantized_weight, max_abs, shape)
        R = torch.randn((reduced_rank, in_feature), device=device)
        L = torch.zeros((out_feature, reduced_rank), device=device)

    lora_A, lora_B = R, L

    return dequantized_weight, lora_A, lora_B
