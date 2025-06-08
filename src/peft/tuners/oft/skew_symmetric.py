import torch
import triton
import triton.language as tl
from torch.autograd import Function
import time

# --------------------------
# Triton Forward Kernel
# --------------------------
@triton.autotune(
    configs=[
        # Test smaller stages/warps for smaller blocks
        triton.Config({'BLOCK_SIZE': 32}, num_stages=2, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=3, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=4, num_warps=2),

        triton.Config({'BLOCK_SIZE': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=4, num_warps=4),

        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=8),

        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=16),

        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=16),
    ],
    key=['N'], # Autotune based on matrix size N
)
@triton.jit
def vector_to_skew_symmetric_kernel_working(
    vec_ptr,
    mat_ptr,
    N,
    stride_vec_batch,
    stride_vec_element,
    stride_mat_batch,
    stride_mat_row,
    stride_mat_col,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    vec_batch_ptr = vec_ptr + pid_batch * stride_vec_batch
    mat_batch_ptr = mat_ptr + pid_batch * stride_mat_batch
    
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask_m = offs_m < N
    mask_n = offs_n < N
    full_mask = mask_m[:, None] & mask_n[None, :]
    
    i = offs_m[:, None]
    j = offs_n[None, :]
    
    # Upper triangle
    upper_mask = i < j
    upper_idx = i * (2 * N - i - 1) // 2 + (j - i - 1)
    upper_val = tl.load(vec_batch_ptr + upper_idx * stride_vec_element, mask=upper_mask & full_mask, other=0.0)
    
    # Lower triangle
    lower_mask = i > j
    lower_idx = j * (2 * N - j - 1) // 2 + (i - j - 1)
    lower_val = -tl.load(vec_batch_ptr + lower_idx * stride_vec_element, mask=lower_mask & full_mask, other=0.0)
    
    result = tl.where(upper_mask, upper_val, tl.where(lower_mask, lower_val, 0.0))
    mat_ptrs = mat_batch_ptr + i * stride_mat_row + j * stride_mat_col
    tl.store(mat_ptrs, result, mask=full_mask)

@triton.autotune(
    configs=[
        # Test smaller stages/warps for smaller blocks
        triton.Config({'BLOCK_SIZE': 32}, num_stages=2, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=3, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=4, num_warps=2),

        triton.Config({'BLOCK_SIZE': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=4, num_warps=4),

        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=8),

        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=16),

        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=16),
    ],
    key=['N'], # Autotune based on matrix size N
)
@triton.jit
def skew_symmetric_forward_kernel_optimized(
    vec_ptr,
    mat_ptr,
    N,
    stride_vec_batch,
    stride_vec_element,
    stride_mat_batch,
    stride_mat_row,
    stride_mat_col,
    BLOCK_SIZE: tl.constexpr,
): 
    # 3D program IDs: batch, row block, column block
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Offset calculations for matrix blocks
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid matrix indices
    mask_m = offs_m < N
    mask_n = offs_n < N
    full_mask = mask_m[:, None] & mask_n[None, :]

    # Create 2D indices [BLOCK_SIZE, BLOCK_SIZE]
    i = offs_m[:, None]  # [BLOCK_SIZE, 1]
    j = offs_n[None, :]  # [1, BLOCK_SIZE]
    
    # Upper triangle processing
    upper_mask = (i < j) & full_mask
    
    # Vector index calculation for upper triangle
    upper_idx = i * (2 * N - i - 1) // 2 + (j - i - 1)

    # Batch-aware pointer arithmetic
    vec_batch_ptr = vec_ptr + pid_batch * stride_vec_batch
    vec_ptrs = vec_batch_ptr + upper_idx * stride_vec_element
    
    # Load upper triangle values
    upper_vals = tl.load(vec_ptrs, mask=upper_mask, other=0.0)
    
    # Matrix pointer calculations for current batch
    mat_batch_ptr = mat_ptr + pid_batch * stride_mat_batch
    mat_ptrs_upper = mat_batch_ptr + i * stride_mat_row + j * stride_mat_col
    mat_ptrs_lower = mat_batch_ptr + j * stride_mat_row + i * stride_mat_col
    
    # Store upper values and their negatives (skew-symmetric)
    tl.store(mat_ptrs_upper, upper_vals, mask=upper_mask)
    tl.store(mat_ptrs_lower, -upper_vals, mask=upper_mask)
    
    # Zero out diagonal elements
    diag_mask = (i == j) & full_mask
    diag_ptrs = mat_batch_ptr + i * stride_mat_row + j * stride_mat_col
    tl.store(diag_ptrs, tl.zeros((BLOCK_SIZE, BLOCK_SIZE), 
                               dtype=vec_ptr.dtype.element_ty), 
             mask=diag_mask)

# --------------------------
# Triton Backward Kernel
# --------------------------
@triton.autotune(
    configs=[
        # Test smaller stages/warps for smaller blocks
        triton.Config({'BLOCK_SIZE': 32}, num_stages=2, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=3, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=4, num_warps=2),

        triton.Config({'BLOCK_SIZE': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=4, num_warps=4),

        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=8),

        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=16),

        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=16),
    ],
    key=['N'], # Autotune based on matrix size N
)
@triton.jit
def skew_symmetric_backward_kernel_working(
    grad_mat_ptr,
    grad_vec_ptr,
    batch_size,
    N,
    stride_mat_batch,
    stride_mat_row,
    stride_mat_col,
    stride_vec_batch,
    stride_vec_element,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    total_elements_per_batch = N * (N - 1) // 2
    total_global_elements = batch_size * total_elements_per_batch
    mask = offs < total_global_elements
    
    global_idx = tl.where(mask, offs, 0)
    batch_idx = global_idx // total_elements_per_batch
    k = global_idx % total_elements_per_batch
    
    # Compute i and j from k
    k_float = k.to(tl.float32)
    N_float = N.to(tl.float32)
    a = 2.0 * N_float - 1.0
    sqrt_val = tl.sqrt(a * a - 8.0 * k_float)
    i_float = (a - sqrt_val) / 2.0
    i = tl.floor(i_float).to(tl.int32)
    
    triangular_num = i * (2 * N - i - 1) // 2
    j = k - triangular_num + i + 1
    
    # Check validity
    valid = (i >= 0) & (j < N) & (i < j)
    mask = mask & valid
    
    grad_mat_batch_ptr = grad_mat_ptr + batch_idx * stride_mat_batch
    grad_vec_batch_ptr = grad_vec_ptr + batch_idx * stride_vec_batch
    
    grad_upper = tl.load(
        grad_mat_batch_ptr + i * stride_mat_row + j * stride_mat_col,
        mask=mask,
        other=0.0
    )
    grad_lower = tl.load(
        grad_mat_batch_ptr + j * stride_mat_row + i * stride_mat_col,
        mask=mask,
        other=0.0
    )
    grad_val = grad_upper - grad_lower
    
    tl.store(grad_vec_batch_ptr + k * stride_vec_element, grad_val, mask=mask)



@triton.autotune(
    configs=[
        # Test smaller stages/warps for smaller blocks
        triton.Config({'BLOCK_SIZE': 32}, num_stages=2, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=3, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=4, num_warps=2),

        triton.Config({'BLOCK_SIZE': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=4, num_warps=4),

        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=8),

        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=16),

        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=16),
    ],
    key=['N'], # Autotune based on matrix size N
)
@triton.jit
def skew_symmetric_backward_kernel_optimized(
    grad_mat_ptr,
    grad_vec_ptr,
    N: tl.int32,
    F: tl.int32,
    stride_mat_batch,
    stride_mat_row,
    stride_mat_col,
    stride_vec_batch,
    stride_vec_element,
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: batch index x element blocks
    pid_batch = tl.program_id(0)
    pid_block = tl.program_id(1)

    # Element offsets within current batch
    offs = pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    k = offs
    mask = k < F  # Filter elements within vector length
    
    # Convert to float for calculations
    N_float = N.to(tl.float32)
    k_float = k.to(tl.float32)
    
    # Quadratic formula to find matrix indices (i, j) from vector index k
    a = 2.0 * N_float - 1.0
    sqrt_val = tl.sqrt(a * a - 8.0 * k_float)
    i_float = (a - sqrt_val) / 2.0
    i = tl.floor(i_float).to(tl.int32)
    
    # Calculate column index j
    triangular_num = i * (2 * N - i - 1) // 2
    j = k - triangular_num + i + 1
    
    # Validate indices
    valid = (i >= 0) & (j < N) & (i < j)
    mask = mask & valid
    
    # Matrix pointer calculations for current batch
    mat_batch_ptr = grad_mat_ptr + pid_batch * stride_mat_batch
    upper_ptr = mat_batch_ptr + i * stride_mat_row + j * stride_mat_col
    lower_ptr = mat_batch_ptr + j * stride_row + i * stride_col
    
    # Load gradients from upper and lower triangle
    grad_upper = tl.load(upper_ptr, mask=mask, other=0.0)
    grad_lower = tl.load(lower_ptr, mask=mask, other=0.0)
    
    # Compute vector gradient (upper - lower due to skew-symmetry)
    grad_vec_val = grad_upper - grad_lower
    
    # Vector pointer calculations
    vec_batch_ptr = grad_vec_ptr + pid_batch * stride_vec_batch
    vec_ptr = vec_batch_ptr + k * stride_vec_element
    
    # Store results
    tl.store(vec_ptr, grad_vec_val, mask=mask)

# --------------------------
# Autograd Function
# --------------------------
class SkewSymmetric(Function):
    @staticmethod
    def forward(ctx, vec, N):
        # Calculate matrix size from vector length
        vec_size = vec.shape[1]
        batch_size = vec.shape[0]
        mat = torch.empty((batch_size, N, N), 
                            device=vec.device, dtype=vec.dtype)

        # Configure kernel launch parameters
        grid = lambda meta: (
            batch_size,
            triton.cdiv(N, meta['BLOCK_SIZE']),
            triton.cdiv(N, meta['BLOCK_SIZE'])
        )

        skew_symmetric_forward_kernel_optimized[grid](
            vec_ptr=vec,
            mat_ptr=mat,
            N=N,
            stride_vec_batch=vec.stride(0),
            stride_vec_element=vec.stride(1),
            stride_mat_batch=mat.stride(0),
            stride_mat_row=mat.stride(1),
            stride_mat_col=mat.stride(2),
            # BLOCK_SIZE=BLOCK_SIZE,
        )
        ctx.save_for_backward(vec)
        ctx.N = N
        return mat

    @staticmethod
    def backward(ctx, grad_output):
        vec, = ctx.saved_tensors
        N = ctx.N
        batch_size, F = vec.shape
        grad_vec = torch.zeros_like(vec)
        
        # Configure kernel launch parameters
        F = N * (N - 1) // 2
        total_global_elements = batch_size * F
        grid = lambda meta: (triton.cdiv(total_global_elements, meta['BLOCK_SIZE']), )
        # grid = lambda meta: (batch_size, triton.cdiv(F, meta['BLOCK_SIZE']))
        
        skew_symmetric_backward_kernel_working[grid](
            grad_output,
            grad_vec,
            batch_size,
            N,
            stride_mat_batch=grad_output.stride(0),
            stride_mat_row=grad_output.stride(1),
            stride_mat_col=grad_output.stride(2),
            stride_vec_batch=grad_vec.stride(0),
            stride_vec_element=grad_vec.stride(1),
            # BLOCK_SIZE=BLOCK_SIZE,
        )
        return grad_vec, None

# --------------------------
# PyTorch Baseline
# --------------------------
def pytorch_skew(vec, N):
    if len(vec.shape) == 1:
        mat = torch.zeros((N, N), device=vec.device, dtype=vec.dtype)
        rows, cols = torch.triu_indices(N, N, 1, device=vec.device)
        mat[rows, cols] = vec
        skew_mat = mat - mat.T
    else:
        mat = torch.zeros((vec.shape[0], N, N), device=vec.device, dtype=vec.dtype)
        rows, cols = torch.triu_indices(N, N, 1, device=vec.device)
        mat[:, rows, cols] = vec
        skew_mat = mat - mat.transpose(-2, -1)
    return skew_mat

# --------------------------
# Testing & Benchmarking
# --------------------------
def test_correctness(B=16, N=512):
    vec = torch.randn(B, N*(N-1)//2, device='cuda', dtype=torch.float32, requires_grad=True)  # FIX HERE
    
    # Forward
    mat_triton = SkewSymmetric.apply(vec, N)
    mat_pytorch = pytorch_skew(vec, N)
    
    # Backward
    grad_output = torch.randn_like(mat_triton)
    
    # Triton backward
    vec.grad = None
    mat_triton.backward(grad_output)
    grad_triton = vec.grad.clone()
    
    # PyTorch backward
    vec.grad = None
    mat_pytorch.backward(grad_output)
    grad_pytorch = vec.grad.clone()
    
    # Verification
    assert torch.allclose(mat_triton, mat_pytorch), "Forward mismatch"
    print('mat_triton', grad_triton)
    print('mat_pytorch', grad_pytorch)
    assert torch.allclose(grad_triton, grad_pytorch, atol=1e-5), f"Backward mismatch"

def benchmark_forward(B=4, N=512):
    vec = torch.randn(B, N*(N-1)//2, device='cuda')

    # Warmup
    for _ in range(3):
        _ = pytorch_skew(vec, N)
    torch.cuda.synchronize()
    
    # PyTorch
    start = time.time()
    mat_pytorch = pytorch_skew(vec, N)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start
    
    # Warmup
    for _ in range(3):
        _ = SkewSymmetric.apply(vec, N)
    torch.cuda.synchronize()
    
    # Triton
    start = time.time()
    mat = SkewSymmetric.apply(vec, N)
    torch.cuda.synchronize()
    triton_time = time.time() - start
    
    print(f"Forward N={N}")
    print(f"Triton: {triton_time*1000:.2f}ms")
    print(f"PyTorch: {pytorch_time*1000:.2f}ms")
    print(f"Speedup: {pytorch_time/triton_time:.1f}x\n")

def benchmark_backward(B=4, N=512):
    vec = torch.randn(B, N*(N-1)//2, device='cuda', requires_grad=True)
    grad_output = torch.randn((B, N, N), device='cuda')
    
    # PyTorch
    # Warmup
    for _ in range(3):
        mat_pytorch = pytorch_skew(vec.clone().requires_grad_(), N)
        mat_pytorch.backward(grad_output)
    torch.cuda.synchronize()
    
    # Benchmark
    vec_pytorch = vec.clone().requires_grad_()
    start = time.time()
    mat_pytorch = pytorch_skew(vec_pytorch, N)
    mat_pytorch.backward(grad_output)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start

    # Triton
    # Warmup
    for _ in range(3):
        mat_triton = SkewSymmetric.apply(vec.clone().requires_grad_(), N)
        mat_triton.backward(grad_output)
    torch.cuda.synchronize()
    
    # Benchmark
    vec_triton = vec.clone().requires_grad_()
    start = time.time()
    mat_triton = SkewSymmetric.apply(vec_triton, N)
    mat_triton.backward(grad_output)
    torch.cuda.synchronize()
    triton_time = time.time() - start
    
    print(f"Backward N={N}")
    print(f"Triton: {triton_time*1000:.2f}ms")
    print(f"PyTorch: {pytorch_time*1000:.2f}ms")
    print(f"Speedup: {pytorch_time/triton_time:.1f}x\n")

if __name__ == "__main__":
    torch.manual_seed(42)
    
    print("Running correctness test...")
    test_correctness()
    
    # print("Benchmarking large matrix:")
    benchmark_forward(8192)
    benchmark_backward(8192)