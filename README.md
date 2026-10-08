# LLM Kernels: Triton fused ops and a tiled CUDA GEMM

Hand-written GPU kernels for the hot paths of transformer models, each checked
against a PyTorch/cuBLAS reference and benchmarked.

| Kernel | Language | What it does |
| --- | --- | --- |
| Softmax | Triton | Fuses max, subtract, exp, sum, divide into 1 kernel; forward + backward |
| RMSNorm | Triton | One read, one write per row; forward + backward (dx and dw) |
| SwiGLU | Triton | silu(gate) * up, autotuned block size; forward + backward |
| GEMM | CUDA C++ | naive → shared-memory tiling → 2D register blocking → float4 vectorized loads, vs cuBLAS |
| RMSNorm, SwiGLU | JAX | The same ops under `jax.jit`, to compare XLA's automatic fusion with hand-written Triton |

All three Triton ops are `torch.autograd.Function`s, so they work inside a model and in training
(`loss.backward()` runs our backward kernels). Gradients are tested against PyTorch autograd.

## Why these kernels

Softmax, RMSNorm and SwiGLU are **memory-bound**: they do very little math per byte.
In PyTorch eager each sub-op reads and writes GPU memory again. A fused kernel reads
the input once and writes the output once, so the speedup comes from bytes not moved.
Results are reported in GB/s against the GPU's peak bandwidth.

GEMM is **compute-bound** at large sizes. Each step removes a bottleneck:

1. **Naive:** every multiply reads A and B from global memory.
2. **Shared-memory tiling:** a tile of A and B is loaded once and reused TILE times.
3. **2D register blocking:** each thread computes a 4x4 patch of C from registers: 8 loads per 16 multiply-adds.
4. **float4 vectorized loads:** 16-byte loads and stores, with A transposed in shared memory so reads are contiguous.

## Results

### Correctness

Every Triton kernel is checked against PyTorch autograd, forward and backward, at odd
sizes (4x7, 16x128, 8x1000) so the masking paths are exercised: fp32 everywhere, plus
fp16 on GPU. The CUDA GEMM checks each variant against cuBLAS (max absolute error is
printed next to the timing). The JAX versions are checked against a NumPy reference.

### Benchmarks

`llm_kernels_colab.ipynb` runs all benchmarks on a Colab T4 and prints Markdown tables:

- Triton softmax / RMSNorm / SwiGLU vs PyTorch eager vs `torch.compile` (4096 rows, fp16, hidden 1024 to 8192), in ms and GB/s
- RMSNorm forward + backward (a training step) vs PyTorch eager
- GEMM (fp32, N = 512 to 4096): naive, tiled 16, tiled 32, register-blocked, float4, cuBLAS, in ms, GFLOP/s and % of cuBLAS
- RMSNorm and SwiGLU under `jax.jit` on the same GPU

## Run it

Google Colab, Runtime > Change runtime type > T4 GPU, then open `llm_kernels_colab.ipynb`
and run all cells. Or locally on an NVIDIA GPU:

```bash
pip install torch triton
python test_and_bench.py                              # tests (fwd + bwd) + Triton benchmarks
python jax_bench.py                                   # JAX comparison
nvcc -O3 -arch=sm_75 gemm.cu -lcublas -o gemm && ./gemm  # GEMM benchmark (sm_75 = T4)
```

Correctness tests also run without a GPU through Triton's interpreter:

```bash
TRITON_INTERPRET=1 python test_and_bench.py --test-only
```

## Limitations and next steps

- Softmax and RMSNorm keep a whole row in one block, so very wide rows (over about 64K) need a looped version.
- RMSNorm's dw is reduced with a [rows, cols] fp32 buffer plus one `torch.sum`; a production kernel would reduce in blocks to save memory.
- GEMM next steps: double buffering (overlap loads with math), warp tiling, then Tensor Cores (MMA) on NVIDIA or MFMA on AMD.
- Port to AMD: the Triton kernels run on ROCm unchanged; the CUDA GEMM ports with `hipify`, with 64-wide wavefronts in mind.
