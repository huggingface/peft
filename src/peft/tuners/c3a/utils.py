# Copyright 2025-present the HuggingFace Inc. team.
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
import torch
from torch.autograd import Function
from torch.fft import fft, ifft


def get_circulant_fast(w):
    m, n, b = w.shape
    x = torch.eye(n * b, dtype=w.dtype, device=w.device)
    x = x.reshape(*x.shape[:-1], n, b)
    x = torch.einsum("...nb,mnb->...mb", ifft(x), fft(w))
    x = fft(x).real.flatten(start_dim=1).T
    return x


class BlockCircularConvolution(Function):
    @staticmethod
    def forward(ctx, x, w):
        m, n, b = w.shape
        x = x.reshape(*x.shape[:-1], n, b)
        ctx.save_for_backward(x, w)
        x = torch.einsum("...nb,mnb->...mb", ifft(x), fft(w))
        x = fft(x).real
        x = x.reshape(*x.shape[:-2], -1)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        m, n, b = w.shape
        grad_output = grad_output.reshape(*grad_output.shape[:-1], m, b)
        grad_output_fft = fft(grad_output)
        x_grad = fft(torch.einsum("...mb,mnb->...nb", grad_output_fft, ifft(w))).real
        x_grad = x_grad.reshape(*x_grad.shape[:-2], -1)
        w_grad = fft(torch.einsum("...mb,...nb->mnb", grad_output_fft, ifft(x))).real
        return x_grad, w_grad
