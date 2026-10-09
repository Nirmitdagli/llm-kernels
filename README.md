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

Measured on a Colab T4 (driver 580.82) with PyTorch 2.11 and Triton 3.6. Every table below is
copied from the outputs saved in `llm_kernels_colab.ipynb`.

**Summary**

- The fused Triton kernels run at **223 to 250 GB/s, 70 to 78% of the T4's peak bandwidth**.
- They are **4.9x to 10.1x faster than the PyTorch eager reference**. The reference computes in
  fp32 like the kernels do, so eager also pays for fp16 to fp32 casts and back: softmax is 3
  kernels moving about 20 bytes per element against the fused kernel's 4. RMSNorm gains the most
  because eager runs it as several ops (pow, mean, rsqrt, two multiplies), each a trip to memory.
- `torch.compile` generates fused Triton kernels too, so it is the fair baseline: it lands within
  a few percent of the hand-written kernels, and is slower on the 8192-wide softmax and RMSNorm.
- For training, the fused RMSNorm forward + backward is **7.5x to 7.8x faster** than eager autograd.
- The CUDA GEMM goes from **10% to 64% of cuBLAS at N = 4096**, 6.2x faster than naive. N = 4096
  gave 64% in two separate runs. Smaller sizes vary between runs (an earlier run measured 93% at
  N = 1024, this one 53%): each call lasts about 1 ms, so clock boost on the T4 moves the ratio.
- XLA (`jax.jit`) is on par with Triton on SwiGLU, a simple elementwise op it fuses well. Triton is
  1.33x faster on RMSNorm (0.589 ms vs 0.783 ms at hidden size 8192).

**Triton kernels, forward (4096 rows, fp16)**

| Kernel | Hidden size | Triton ms | Eager ms | torch.compile ms | Triton GB/s | Eager GB/s | Speedup vs eager |
| --- | --- | --- | --- | --- | --- | --- | --- |
| softmax | 1024 | 0.073 | 0.420 | 0.079 | 229 | 40 | 5.75x |
| rmsnorm | 1024 | 0.075 | 0.762 | 0.076 | 223 | 22 | 10.12x |
| swiglu | 1024 | 0.104 | 0.702 | 0.104 | 243 | 36 | 6.78x |
| softmax | 2048 | 0.147 | 0.758 | 0.148 | 228 | 44 | 5.14x |
| rmsnorm | 2048 | 0.149 | 1.445 | 0.146 | 225 | 23 | 9.71x |
| swiglu | 2048 | 0.204 | 1.395 | 0.201 | 246 | 36 | 6.82x |
| softmax | 4096 | 0.292 | 1.434 | 0.293 | 230 | 47 | 4.91x |
| rmsnorm | 4096 | 0.293 | 2.855 | 0.291 | 229 | 24 | 9.73x |
| swiglu | 4096 | 0.404 | 2.769 | 0.396 | 249 | 36 | 6.85x |
| softmax | 8192 | 0.586 | 2.877 | 0.666 | 229 | 47 | 4.91x |
| rmsnorm | 8192 | 0.589 | 5.658 | 0.636 | 228 | 24 | 9.61x |
| swiglu | 8192 | 0.806 | 5.525 | 0.783 | 250 | 36 | 6.85x |

**RMSNorm forward + backward (training step, 4096 rows, fp16)**

| Kernel | Hidden size | Triton ms | Eager ms | Speedup |
| --- | --- | --- | --- | --- |
| rmsnorm | 1024 | 0.351 | 2.619 | 7.45x |
| rmsnorm | 4096 | 1.321 | 10.190 | 7.71x |
| rmsnorm | 8192 | 2.604 | 20.245 | 7.77x |

**GEMM (fp32, N x N)**

| N | Kernel | ms | GFLOP/s | Effective GB/s | % of cuBLAS | max error vs cuBLAS |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | naive | 0.904 | 297 | 3 | 11.8% | 0.00e+00 |
| 512 | tiled 16 | 0.583 | 460 | 5 | 18.2% | 0.00e+00 |
| 512 | tiled 32 | 0.555 | 484 | 6 | 19.2% | 0.00e+00 |
| 512 | regblock 4x4 | 0.219 | 1227 | 14 | 48.6% | 0.00e+00 |
| 512 | vec4 + regblock | 0.190 | 1413 | 17 | 55.9% | 0.00e+00 |
| 512 | cuBLAS | 0.106 | 2525 | 30 | 100.0% | 0.00e+00 |
| 1024 | naive | 7.131 | 301 | 2 | 9.1% | 1.62e-05 |
| 1024 | tiled 16 | 4.444 | 483 | 3 | 14.6% | 1.62e-05 |
| 1024 | tiled 32 | 4.116 | 522 | 3 | 15.8% | 1.62e-05 |
| 1024 | regblock 4x4 | 1.482 | 1449 | 8 | 43.8% | 1.62e-05 |
| 1024 | vec4 + regblock | 1.231 | 1744 | 10 | 52.8% | 1.62e-05 |
| 1024 | cuBLAS | 0.650 | 3305 | 19 | 100.0% | 0.00e+00 |
| 2048 | naive | 41.398 | 415 | 1 | 6.2% | 6.10e-05 |
| 2048 | tiled 16 | 26.442 | 650 | 2 | 9.7% | 6.10e-05 |
| 2048 | tiled 32 | 19.518 | 880 | 3 | 13.1% | 6.10e-05 |
| 2048 | regblock 4x4 | 7.183 | 2392 | 7 | 35.7% | 6.10e-05 |
| 2048 | vec4 + regblock | 5.953 | 2886 | 8 | 43.0% | 6.10e-05 |
| 2048 | cuBLAS | 2.561 | 6708 | 20 | 100.0% | 0.00e+00 |
| 4096 | naive | 333.657 | 412 | 1 | 10.3% | 0.00e+00 |
| 4096 | tiled 16 | 216.747 | 634 | 1 | 15.8% | 0.00e+00 |
| 4096 | tiled 32 | 160.753 | 855 | 1 | 21.3% | 0.00e+00 |
| 4096 | regblock 4x4 | 67.237 | 2044 | 3 | 50.9% | 0.00e+00 |
| 4096 | vec4 + regblock | 53.938 | 2548 | 4 | 63.5% | 0.00e+00 |
| 4096 | cuBLAS | 34.250 | 4013 | 6 | 100.0% | 0.00e+00 |

At N = 512 the problem is too small to fill the GPU, so those rows mostly measure launch overhead.

Why some errors are exactly zero: all four kernels sum over k in the same order (0 to N-1) with
one fp32 accumulator per output, so they produce bit-identical results, which is why every kernel
shows the same error at a given N. The most likely reason for exact zeros: at N = 512 and 4096
cuBLAS picked a kernel that sums in the same order, so the outputs match bit for bit; at 1024 and
2048 it picked one with a different order, so results differ by rounding (about 1e-5). C is
zeroed before every kernel, so a kernel that wrote nothing would show a large error, not zero.

**JAX (`jax.jit`, fp16, 4096 rows) on the same GPU**

| Kernel | Hidden size | JAX ms | JAX GB/s | Triton ms |
| --- | --- | --- | --- | --- |
| rmsnorm | 1024 | 0.359 | 47 | 0.075 |
| swiglu | 1024 | 0.416 | 61 | 0.104 |
| rmsnorm | 2048 | 0.212 | 159 | 0.149 |
| swiglu | 2048 | 0.202 | 250 | 0.204 |
| rmsnorm | 4096 | 0.404 | 166 | 0.293 |
| swiglu | 4096 | 0.393 | 256 | 0.404 |
| rmsnorm | 8192 | 0.783 | 171 | 0.589 |
| swiglu | 8192 | 0.775 | 260 | 0.806 |

At hidden size 1024 each JAX call takes about 0.1 ms of GPU time, so the 0.36 ms measured is
mostly Python dispatch overhead on the Colab CPU, not the kernel. Compare JAX at 2048 and above.

A first run passed the same array as both SwiGLU inputs, and XLA read it once, which reported
372 GB/s, above the T4's 320 GB/s peak. The benchmark now uses two separate arrays.

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
