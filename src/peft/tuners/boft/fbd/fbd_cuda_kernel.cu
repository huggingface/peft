// Author: Yao Feng
// Date: 2023/08
// Description: cuda kernel for fast block diag

#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace{
template <typename scalar_t>
__global__ void forward_fast_block_diag_cuda_kernel(
        const scalar_t* __restrict__ input, //[z, N, b, b]
        scalar_t*  output, //[z, Nxb, Nxb]
        int z, int N, int b
    ) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= z*N*b*b) {
        return;
    }
    const int zi = i/(N*b*b);
    const int Ni = (i%(N*b*b))/(b*b);
    const int x = ((i%(N*b*b))%(b*b))/b;
    const int y = ((i%(N*b*b))%(b*b))%b;

    output[zi*N*b*N*b + (Ni*b+x)*N*b + Ni*b + y] = input[zi*N*b*b + Ni*b*b + x*b + y];

}

template <typename scalar_t>
__global__ void backward_fast_block_diag_cuda_kernel(
        const scalar_t* __restrict__ grad_output, 
        scalar_t*  grad_input, 
        int z, int N, int b
    ) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= z*N*b*b) {
        return;
    }
    const int zi = i/(N*b*b);
    const int Ni = (i%(N*b*b))/(b*b);
    const int x = ((i%(N*b*b))%(b*b))/b;
    const int y = ((i%(N*b*b))%(b*b))%b;
    
    grad_input[zi*N*b*b + Ni*b*b + x*b + y] = grad_output[zi*N*b*N*b + (Ni*b+x)*N*b + Ni*b + y];

} // namespace
}

std::vector<at::Tensor> forward_fast_block_diag_cuda(
    at::Tensor input
    ){
    const auto z = input.size(0);
    const auto N = input.size(1);
    const auto b = input.size(2);

    // print(channel_size)
    const int threads = 512;
    const dim3 blocks_1 ((z*N*b*b - 1) / threads +1);
    // initlaize output
    auto output = at::zeros({z, N*b, N*b}, input.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "forward_fast_block_diag1", ([&] {
        forward_fast_block_diag_cuda_kernel<scalar_t><<<blocks_1, threads>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        z, N, b);
      }));

   
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in forward_fast_block_diag_cuda_kernel: %s\n", cudaGetErrorString(err));

    return {output};
}

std::vector<at::Tensor> backward_fast_block_diag_cuda(
    at::Tensor grad_output,
    at::Tensor input
    ){

    const auto z = input.size(0);
    const auto N = input.size(1);
    const auto b = input.size(2);
    
    // print(channel_size)
    const int threads = 512;
    const dim3 blocks_1 ((z*N*b*b - 1) / threads +1);
    
    // initialize grad input
    auto grad_input = at::zeros_like(input);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.type(), "backward_fast_block_diag", ([&] {
        backward_fast_block_diag_cuda_kernel<scalar_t><<<blocks_1, threads>>>(
        grad_output.data_ptr<scalar_t>(),
        grad_input.data_ptr<scalar_t>(),
        z, N, b);
      }));
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in backward_fast_block_diag_cuda_kernel: %s\n", cudaGetErrorString(err));

    return {grad_input};
}
