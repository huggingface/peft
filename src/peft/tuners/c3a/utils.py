import torch
from torch.autograd import Function
from torch.fft import fft, ifft


def circulant(tensor, dim=-1):
    """get a circulant version of the tensor along the {dim} dimension.
    
    The additional axis is appended as the last dimension.
    E.g. tensor=[0,1,2], dim=0 --> [[0,1,2],[2,0,1],[1,2,0]]"""
    S = tensor.shape[dim]
    tmp = torch.cat([tensor.flip((dim,)), torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=S-1)], dim=dim)
    return tmp.unfold(dim, S, 1).flip((-1,))

def get_circulant_fast(w):
    m, n, b = w.shape
    x = torch.eye(n*b, dtype=w.dtype, device=w.device)
    x = x.reshape(*x.shape[:-1], n, b)
    x = torch.einsum( "...nb,mnb->...mb", ifft(x), fft(w) ) 
    x = fft(x).real.flatten(start_dim=1).T
    return x

class BlockCircularConvolution(Function):
    @staticmethod
    def forward(ctx, x, w):
        m, n, b = w.shape
        x = x.reshape(*x.shape[:-1], n, b)
        ctx.save_for_backward(x, w)
        x = torch.einsum( "...nb,mnb->...mb", ifft(x), fft(w) ) 
        x = fft(x).real
        x = x.reshape(*x.shape[:-2], -1)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        m, n, b = w.shape
        grad_output = grad_output.reshape(*grad_output.shape[:-1], m, b)
        grad_output_fft = fft(grad_output)
        x_grad = fft(torch.einsum( "...mb,mnb->...nb", grad_output_fft, ifft(w))).real
        x_grad = x_grad.reshape(*x_grad.shape[:-2], -1)
        w_grad = fft(torch.einsum( "...mb,...nb->mnb", grad_output_fft, ifft(x))).real
        return x_grad, w_grad