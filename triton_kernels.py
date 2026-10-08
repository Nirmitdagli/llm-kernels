"""
Fused LLM kernels in Triton: softmax, RMSNorm, SwiGLU. Forward AND backward.

Each kernel replaces several PyTorch ops (several trips to GPU memory) with ONE
read and ONE write. These ops are memory-bound, so fewer trips = faster.

Triton model in one line: you write what ONE program (block) does on a whole
tile of data. Triton maps it to threads, coalesced loads and shared memory.

PyTorch integration: each op is a torch.autograd.Function, so it can be used
inside a model and trained (loss.backward() calls our backward kernels).
"""
import os
import torch
import triton
import triton.language as tl

_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"   # CPU interpreter for testing without a GPU


def _row_warps(block):
    """More warps for wider rows so each thread handles fewer elements."""
    return 4 if block <= 2048 else (8 if block <= 8192 else 16)


# =============================================================================
# 1. SOFTMAX over the last dimension
#    forward : y  = exp(x - max) / sum(exp(x - max))
#    backward: dx = y * (dy - sum(dy * y))          (per row)
#    One program per row. The whole row fits in one block (BLOCK >= n_cols).
# =============================================================================
@triton.jit
def softmax_fwd_kernel(x_ptr, y_ptr, stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)                       # which row this program owns
    cols = tl.arange(0, BLOCK)                   # column indices 0..BLOCK-1
    mask = cols < n_cols                         # BLOCK is a power of 2; the row may be shorter

    # Padding = -inf so it never wins the max, and exp(-inf) = 0 adds nothing to the sum.
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)                    # numerical stability: largest value becomes 0
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(y_ptr + row * stride + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def softmax_bwd_kernel(y_ptr, dy_ptr, dx_ptr, stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    y = tl.load(y_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    dot = tl.sum(dy * y, axis=0)                 # one scalar per row
    dx = y * (dy - dot)
    tl.store(dx_ptr + row * stride + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


class TritonSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        rows, cols = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(cols)
        softmax_fwd_kernel[(rows,)](x2, y, x2.stride(0), cols, BLOCK=BLOCK, num_warps=_row_warps(BLOCK))
        ctx.save_for_backward(y)                 # backward only needs the output y
        ctx.shape = shape
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        (y,) = ctx.saved_tensors
        dy2 = dy.reshape(y.shape).contiguous()
        rows, cols = y.shape
        dx = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(cols)
        softmax_bwd_kernel[(rows,)](y, dy2, dx, y.stride(0), cols, BLOCK=BLOCK, num_warps=_row_warps(BLOCK))
        return dx.reshape(ctx.shape)


def softmax(x):
    return TritonSoftmax.apply(x)


# =============================================================================
# 2. RMSNorm (Llama, Mistral, DeepSeek)
#    forward : rstd = 1 / sqrt(mean(x^2) + eps);  xhat = x * rstd;  y = xhat * w
#    backward: dx = rstd * (dy*w - xhat * mean(dy*w*xhat))
#              dw = sum over rows of (dy * xhat)
#    One program per row (one token's hidden vector).
# =============================================================================
@triton.jit
def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, stride, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / n_cols     # padding is 0, so it does not change the sum
    rstd = 1.0 / tl.sqrt(mean_sq + eps)          # reciprocal standard deviation
    tl.store(rstd_ptr + row, rstd)               # saved for the backward pass (1 float per row)
    y = x * rstd * w
    tl.store(y_ptr + row * stride + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def rmsnorm_bwd_kernel(x_ptr, w_ptr, dy_ptr, rstd_ptr, dx_ptr, dw_part_ptr, stride, n_cols,
                       BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)

    xhat = x * rstd
    dyw = dy * w
    c = tl.sum(dyw * xhat, axis=0) / n_cols
    dx = (dyw - xhat * c) * rstd
    tl.store(dx_ptr + row * stride + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)

    # dw needs a sum over ALL rows. Each program writes its row's share (fp32);
    # the host sums the [rows, cols] partials with one torch.sum. Simple and deterministic.
    tl.store(dw_part_ptr + row * n_cols + cols, dy * xhat, mask=mask)


class TritonRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()   # [batch, seq, hidden] -> [tokens, hidden]
        rows, cols = x2.shape
        y = torch.empty_like(x2)
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        BLOCK = triton.next_power_of_2(cols)
        rmsnorm_fwd_kernel[(rows,)](x2, w, y, rstd, x2.stride(0), cols, eps,
                                    BLOCK=BLOCK, num_warps=_row_warps(BLOCK))
        ctx.save_for_backward(x2, w, rstd)
        ctx.shape = shape
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, w, rstd = ctx.saved_tensors
        rows, cols = x2.shape
        dy2 = dy.reshape(rows, cols).contiguous()
        dx = torch.empty_like(x2)
        dw_part = torch.empty(rows, cols, device=x2.device, dtype=torch.float32)
        BLOCK = triton.next_power_of_2(cols)
        rmsnorm_bwd_kernel[(rows,)](x2, w, dy2, rstd, dx, dw_part, x2.stride(0), cols,
                                    BLOCK=BLOCK, num_warps=_row_warps(BLOCK))
        dw = dw_part.sum(0).to(w.dtype)
        return dx.reshape(ctx.shape), dw, None    # no gradient for eps


def rmsnorm(x, weight, eps=1e-6):
    return TritonRMSNormFn.apply(x, weight, eps)


class TritonRMSNorm(torch.nn.Module):
    """Drop-in replacement for the RMSNorm layer in Llama-style models."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return rmsnorm(x, self.weight, self.eps)


# =============================================================================
# 3. SwiGLU (the MLP activation in Llama): y = silu(gate) * up
#    silu(g) = g * sigmoid(g),  silu'(g) = sig * (1 + g * (1 - sig))
#    backward: d_gate = dy * up * silu'(gate);  d_up = dy * silu(gate)
#    Pure elementwise: treat tensors as flat 1D arrays, tile them, AUTOTUNE the block size.
# =============================================================================
@triton.jit
def _swiglu_fwd_kernel(g_ptr, u_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=mask).to(tl.float32)
    y = g * tl.sigmoid(g) * u                    # fused: 2 reads + 1 write instead of 3 kernels
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_bwd_kernel(g_ptr, u_ptr, dy_ptr, dg_ptr, du_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=mask).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask).to(tl.float32)
    sig = tl.sigmoid(g)
    silu = g * sig
    dsilu = sig * (1.0 + g * (1.0 - sig))
    tl.store(dg_ptr + offs, (dy * u * dsilu).to(dg_ptr.dtype.element_ty), mask=mask)
    tl.store(du_ptr + offs, (dy * silu).to(du_ptr.dtype.element_ty), mask=mask)


# Autotune tries each config the first time a size is seen and caches the fastest.
# It needs a real GPU, so the CPU interpreter uses a fixed block size instead.
_EW_CONFIGS = [
    triton.Config({"BLOCK": 512}, num_warps=4),
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=8),
    triton.Config({"BLOCK": 4096}, num_warps=8),
]
if _INTERPRET:
    swiglu_fwd_kernel, swiglu_bwd_kernel = _swiglu_fwd_kernel, _swiglu_bwd_kernel
else:
    swiglu_fwd_kernel = triton.autotune(configs=_EW_CONFIGS, key=["n"])(_swiglu_fwd_kernel)
    swiglu_bwd_kernel = triton.autotune(configs=_EW_CONFIGS, key=["n"])(_swiglu_bwd_kernel)


def _launch_ew(kernel, n, *args):
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    if _INTERPRET:
        kernel[grid](*args, n, BLOCK=1024)
    else:
        kernel[grid](*args, n)


class TritonSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        assert gate.shape == up.shape
        gate, up = gate.contiguous(), up.contiguous()
        y = torch.empty_like(gate)
        _launch_ew(swiglu_fwd_kernel, gate.numel(), gate, up, y)
        ctx.save_for_backward(gate, up)
        return y

    @staticmethod
    def backward(ctx, dy):
        gate, up = ctx.saved_tensors
        dy = dy.contiguous()
        dg, du = torch.empty_like(gate), torch.empty_like(up)
        _launch_ew(swiglu_bwd_kernel, gate.numel(), gate, up, dy, dg, du)
        return dg, du


def swiglu(gate, up):
    return TritonSwiGLU.apply(gate, up)


# =============================================================================
# PyTorch references (what we check against and benchmark against)
# =============================================================================
def softmax_ref(x):
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def rmsnorm_ref(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def swiglu_ref(g, u):
    return (torch.nn.functional.silu(g.float()) * u.float()).to(g.dtype)
