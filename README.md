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

### Benchmarks (Tesla T4, 320 GB/s peak memory bandwidth)

Measured on a Colab T4 with PyTorch 2.11 and Triton 3.6, using `llm_kernels_colab.ipynb`.

**Summary**

- The fused Triton kernels run at **225 to 249 GB/s, 70 to 78% of the T4's peak bandwidth**,
  and are **4.9x to 10.7x faster than PyTorch eager**. RMSNorm gains the most because eager
  runs it as several separate ops (pow, mean, rsqrt, two multiplies), each a full trip to memory.
- `torch.compile` generates fused Triton kernels too, so it lands within a few percent of the
  hand-written ones, and is slightly slower on the widest rows.
- For training, the fused RMSNorm forward + backward is **7.5x to 7.8x faster** than eager autograd.
- The CUDA GEMM goes from **10% to 64% of cuBLAS at N = 4096** (6.2x faster than naive), and
  reaches **93% of cuBLAS at N = 1024**.
- XLA (`jax.jit`) matches Triton on SwiGLU, a simple elementwise op it fuses well. Triton is
  1.33x faster on RMSNorm (0.589 ms vs 0.783 ms at hidden size 8192).

**Triton kernels, forward (4096 rows, fp16)**

| Kernel | Hidden size | Triton ms | Eager ms | torch.compile ms | Triton GB/s | Eager GB/s | Speedup vs eager |
| --- | --- | --- | --- | --- | --- | --- | --- |
| softmax | 1024 | 0.073 | 0.419 | 0.080 | 229 | 40 | 5.72x |
| rmsnorm | 1024 | 0.075 | 0.797 | 0.075 | 225 | 21 | 10.69x |
| swiglu | 1024 | 0.104 | 0.703 | 0.104 | 243 | 36 | 6.78x |
| softmax | 2048 | 0.148 | 0.759 | 0.148 | 227 | 44 | 5.13x |
| rmsnorm | 2048 | 0.149 | 1.446 | 0.146 | 225 | 23 | 9.69x |
| swiglu | 2048 | 0.204 | 1.397 | 0.201 | 246 | 36 | 6.83x |
| softmax | 4096 | 0.292 | 1.436 | 0.293 | 230 | 47 | 4.91x |
| rmsnorm | 4096 | 0.293 | 2.861 | 0.291 | 229 | 23 | 9.75x |
| swiglu | 4096 | 0.404 | 2.776 | 0.396 | 249 | 36 | 6.87x |
| softmax | 8192 | 0.586 | 2.895 | 0.663 | 229 | 46 | 4.94x |
| rmsnorm | 8192 | 0.589 | 5.679 | 0.637 | 228 | 24 | 9.64x |
| swiglu | 8192 | 0.808 | 5.536 | 0.785 | 249 | 36 | 6.85x |

**RMSNorm forward + backward (training step, 4096 rows, fp16)**

| Hidden size | Triton ms | Eager ms | Speedup |
| --- | --- | --- | --- |
| 1024 | 0.352 | 2.624 | 7.46x |
| 4096 | 1.324 | 10.214 | 7.71x |
| 8192 | 2.612 | 20.280 | 7.76x |

**GEMM (fp32, N x N)**

| N | Kernel | ms | GFLOP/s | % of cuBLAS | max error vs cuBLAS |
| --- | --- | --- | --- | --- | --- |
| 1024 | naive | 9.193 | 234 | 9.0% | 1.62e-05 |
| 1024 | tiled 16 | 4.241 | 506 | 19.5% | 1.62e-05 |
| 1024 | tiled 32 | 3.014 | 712 | 27.5% | 1.62e-05 |
| 1024 | regblock 4x4 | 1.127 | 1906 | 73.5% | 1.62e-05 |
| 1024 | vec4 + regblock | 0.895 | 2400 | 92.6% | 1.62e-05 |
| 1024 | cuBLAS | 0.829 | 2592 | 100.0% | 0 |
| 2048 | naive | 39.705 | 433 | 8.4% | 6.10e-05 |
| 2048 | tiled 16 | 25.273 | 680 | 13.2% | 6.10e-05 |
| 2048 | tiled 32 | 19.262 | 892 | 17.3% | 6.10e-05 |
| 2048 | regblock 4x4 | 6.877 | 2498 | 48.4% | 6.10e-05 |
| 2048 | vec4 + regblock | 5.636 | 3048 | 59.0% | 6.10e-05 |
| 2048 | cuBLAS | 3.327 | 5164 | 100.0% | 0 |
| 4096 | naive | 322.714 | 426 | 10.2% | 0 |
| 4096 | tiled 16 | 206.408 | 666 | 16.0% | 0 |
| 4096 | tiled 32 | 153.731 | 894 | 21.5% | 0 |
| 4096 | regblock 4x4 | 64.663 | 2125 | 51.1% | 0 |
| 4096 | vec4 + regblock | 52.057 | 2640 | 63.5% | 0 |
| 4096 | cuBLAS | 33.044 | 4159 | 100.0% | 0 |

At N = 512 the whole problem is too small to fill the GPU, so the full output (in the notebook)
is dominated by launch overhead and is left out here.

**JAX (`jax.jit`, fp16, 4096 rows) on the same GPU**

| Kernel | Hidden size | JAX ms | JAX GB/s | Triton ms |
| --- | --- | --- | --- | --- |
| rmsnorm | 1024 | 0.118 | 143 | 0.075 |
| swiglu | 1024 | 0.129 | 195 | 0.104 |
| rmsnorm | 2048 | 0.213 | 158 | 0.149 |
| swiglu | 2048 | 0.201 | 251 | 0.204 |
| rmsnorm | 4096 | 0.402 | 167 | 0.293 |
| swiglu | 4096 | 0.394 | 255 | 0.404 |
| rmsnorm | 8192 | 0.783 | 171 | 0.589 |
| swiglu | 8192 | 0.776 | 259 | 0.808 |

A first run passed the same array as both SwiGLU inputs, and XLA read it once, which reported
372 GB/s, above the T4's 320 GB/s peak. The benchmark now uses two separate arrays; the table
above is from the corrected run.

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
