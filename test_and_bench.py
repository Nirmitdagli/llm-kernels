"""
Correctness tests (forward + backward) and benchmarks for the Triton kernels.

On a GPU (Colab T4 works):  python test_and_bench.py
Without a GPU (tests only): TRITON_INTERPRET=1 python test_and_bench.py --test-only
"""
import os
import sys
import torch
import triton

from triton_kernels import (softmax, rmsnorm, swiglu,
                            softmax_ref, rmsnorm_ref, swiglu_ref)

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRET else "cuda"


def check(name, ours, ref, atol, rtol):
    torch.testing.assert_close(ours, ref, atol=atol, rtol=rtol, msg=lambda m: f"{name}: {m}")


def test():
    """Compare outputs AND gradients with PyTorch autograd, in fp32 and (on GPU) fp16."""
    torch.manual_seed(0)
    dtypes = [torch.float32] if INTERPRET else [torch.float32, torch.float16]
    for dtype in dtypes:
        tol = 1e-4 if dtype == torch.float32 else 2e-2
        for rows, cols in [(4, 7), (16, 128), (8, 1000)]:      # odd sizes exercise the masks
            # --- softmax ---
            x = torch.randn(rows, cols, device=DEVICE, dtype=dtype, requires_grad=True)
            x_ref = x.detach().clone().requires_grad_(True)
            dy = torch.randn(rows, cols, device=DEVICE, dtype=dtype)
            y, y_ref = softmax(x), softmax_ref(x_ref)
            check("softmax fwd", y, y_ref, tol, tol)
            y.backward(dy); y_ref.backward(dy)
            check("softmax bwd dx", x.grad, x_ref.grad, tol, tol)

            # --- RMSNorm (3D input, like [batch, seq, hidden]) ---
            x = torch.randn(2, rows, cols, device=DEVICE, dtype=dtype, requires_grad=True)
            w = torch.randn(cols, device=DEVICE, dtype=dtype, requires_grad=True)
            x_ref = x.detach().clone().requires_grad_(True)
            w_ref = w.detach().clone().requires_grad_(True)
            dy = torch.randn(2, rows, cols, device=DEVICE, dtype=dtype)
            y, y_ref = rmsnorm(x, w), rmsnorm_ref(x_ref, w_ref)
            check("rmsnorm fwd", y, y_ref, tol, tol)
            y.backward(dy); y_ref.backward(dy)
            check("rmsnorm bwd dx", x.grad, x_ref.grad, 5 * tol, 5 * tol)
            check("rmsnorm bwd dw", w.grad, w_ref.grad, 5 * tol, 5 * tol)

            # --- SwiGLU ---
            g = torch.randn(rows, cols, device=DEVICE, dtype=dtype, requires_grad=True)
            u = torch.randn(rows, cols, device=DEVICE, dtype=dtype, requires_grad=True)
            g_ref = g.detach().clone().requires_grad_(True)
            u_ref = u.detach().clone().requires_grad_(True)
            dy = torch.randn(rows, cols, device=DEVICE, dtype=dtype)
            y, y_ref = swiglu(g, u), swiglu_ref(g_ref, u_ref)
            check("swiglu fwd", y, y_ref, tol, tol)
            y.backward(dy); y_ref.backward(dy)
            check("swiglu bwd d_gate", g.grad, g_ref.grad, tol, tol)
            check("swiglu bwd d_up", u.grad, u_ref.grad, tol, tol)
    print("All forward and backward correctness tests passed.")


def gbps(nbytes, ms):
    return nbytes / (ms * 1e-3) / 1e9


def bench():
    """Memory-bound kernels: report GB/s. Triton vs PyTorch eager vs torch.compile, fp16."""
    dtype, rows = torch.float16, 4096                       # 4096 tokens per batch
    c_soft, c_rms, c_swi = (torch.compile(f) for f in (softmax_ref, rmsnorm_ref, swiglu_ref))
    print(f"\nGPU: {torch.cuda.get_device_name()}  |  {rows} rows, fp16, forward pass")
    print("| Kernel | Hidden size | Triton ms | Eager ms | torch.compile ms | Triton GB/s | Eager GB/s | Speedup vs eager |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for cols in [1024, 2048, 4096, 8192]:                   # typical hidden sizes
        x = torch.randn(rows, cols, device="cuda", dtype=dtype)
        w = torch.randn(cols, device="cuda", dtype=dtype)
        g = torch.randn(rows, cols, device="cuda", dtype=dtype)
        u = torch.randn(rows, cols, device="cuda", dtype=dtype)
        e = x.element_size()
        cases = [
            ("softmax", 2 * x.numel() * e, lambda: softmax(x), lambda: softmax_ref(x), lambda: c_soft(x)),
            ("rmsnorm", 2 * x.numel() * e, lambda: rmsnorm(x, w), lambda: rmsnorm_ref(x, w), lambda: c_rms(x, w)),
            ("swiglu", 3 * x.numel() * e, lambda: swiglu(g, u), lambda: swiglu_ref(g, u), lambda: c_swi(g, u)),
        ]
        with torch.no_grad():
            for name, nbytes, f_t, f_e, f_c in cases:
                t, e_, c = (triton.testing.do_bench(f) for f in (f_t, f_e, f_c))
                print(f"| {name} | {cols} | {t:.3f} | {e_:.3f} | {c:.3f} | {gbps(nbytes, t):.0f} | {gbps(nbytes, e_):.0f} | {e_ / t:.2f}x |")

    # Forward + backward (training step) for RMSNorm, the most common use.
    print("\n| RMSNorm fwd+bwd | Hidden size | Triton ms | Eager ms | Speedup |")
    print("| --- | --- | --- | --- | --- |")
    for cols in [1024, 4096, 8192]:
        x = torch.randn(rows, cols, device="cuda", dtype=dtype, requires_grad=True)
        w = torch.randn(cols, device="cuda", dtype=dtype, requires_grad=True)
        dy = torch.randn(rows, cols, device="cuda", dtype=dtype)
        t = triton.testing.do_bench(lambda: rmsnorm(x, w).backward(dy), grad_to_none=[x, w])
        e_ = triton.testing.do_bench(lambda: rmsnorm_ref(x, w).backward(dy), grad_to_none=[x, w])
        print(f"| rmsnorm | {cols} | {t:.3f} | {e_:.3f} | {e_ / t:.2f}x |")


if __name__ == "__main__":
    test()
    if "--test-only" not in sys.argv and not INTERPRET:
        bench()
