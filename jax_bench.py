"""
The same RMSNorm and SwiGLU in JAX, compiled with jax.jit (XLA fuses them),
benchmarked against our Triton kernels on the same GPU.

Point of the comparison: XLA's automatic fusion vs hand-written Triton kernels.
Run: python jax_bench.py   (after test_and_bench.py, on the same GPU)
"""
import time
import numpy as np
import jax
import jax.numpy as jnp


@jax.jit
def rmsnorm_jax(x, w, eps=1e-6):
    xf = x.astype(jnp.float32)
    rstd = jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (xf * rstd * w.astype(jnp.float32)).astype(x.dtype)


@jax.jit
def swiglu_jax(g, u):
    gf = g.astype(jnp.float32)
    return (jax.nn.silu(gf) * u.astype(jnp.float32)).astype(g.dtype)


# jax.grad works on our function for free: JAX differentiates the traced math.
rmsnorm_grad = jax.jit(jax.grad(lambda x, w: rmsnorm_jax(x, w).astype(jnp.float32).sum(), argnums=(0, 1)))


def bench_ms(f, *args, iters=50):
    jax.block_until_ready(f(*args))              # first call compiles; keep it out of the timing
    t0 = time.perf_counter()
    for _ in range(iters):
        out = f(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters * 1e3


def check():
    """JAX matches a NumPy reference."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 1000)).astype(np.float32)
    w = rng.standard_normal(1000).astype(np.float32)
    ref = x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * w
    np.testing.assert_allclose(np.asarray(rmsnorm_jax(x, w)), ref, rtol=1e-4, atol=1e-4)
    g, u = x, x[::-1].copy()
    ref = g / (1 + np.exp(-g)) * u
    np.testing.assert_allclose(np.asarray(swiglu_jax(g, u)), ref, rtol=1e-4, atol=1e-4)
    dx, dw = rmsnorm_grad(x, w)
    assert dx.shape == x.shape and dw.shape == w.shape
    print("JAX correctness checks passed on", jax.devices()[0].platform)


if __name__ == "__main__":
    check()
    if jax.devices()[0].platform == "gpu":
        rows = 4096
        print(f"\n| Kernel (JAX jit, fp16) | Hidden size | ms | GB/s |")
        print("| --- | --- | --- | --- |")
        for cols in [1024, 2048, 4096, 8192]:
            key = jax.random.PRNGKey(0)
            x = jax.random.normal(key, (rows, cols), dtype=jnp.float16)
            w = jax.random.normal(key, (cols,), dtype=jnp.float16)
            ms = bench_ms(rmsnorm_jax, x, w)
            print(f"| rmsnorm | {cols} | {ms:.3f} | {2 * x.size * 2 / (ms * 1e-3) / 1e9:.0f} |")
            ms = bench_ms(swiglu_jax, x, x)
            print(f"| swiglu | {cols} | {ms:.3f} | {3 * x.size * 2 / (ms * 1e-3) / 1e9:.0f} |")
