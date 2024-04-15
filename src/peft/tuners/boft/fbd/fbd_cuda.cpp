#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <torch/extension.h>

std::vector<at::Tensor> forward_fast_block_diag_cuda(
        at::Tensor input);

std::vector<at::Tensor> forward_fast_block_diag(
        at::Tensor input
        ) {
    return forward_fast_block_diag_cuda(input);
}

std::vector<at::Tensor> backward_fast_block_diag_cuda(
        at::Tensor grad_output, 
        at::Tensor input);
std::vector<at::Tensor> backward_fast_block_diag(
        at::Tensor grad_output,
        at::Tensor input
        ) {
    return backward_fast_block_diag_cuda(grad_output, input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_fast_block_diag, "FAST BLOCK DIAG (CUDA)");
    m.def("backward", &backward_fast_block_diag, "FAST BLOCK DIAG backward (CUDA)");
}
