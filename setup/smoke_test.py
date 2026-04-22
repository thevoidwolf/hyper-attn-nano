#!/usr/bin/env python3
"""
HyperAttn-Nano — Phase 0 Smoke Test
====================================
Run this after phase0_setup.sh completes:
    python setup/smoke_test.py
 
All checks should print ✓.
The FP16 arcosh check is expected to show a warning (not a failure) —
that's deliberate: it documents a known hazard you need to work around.
 
Exit code 0 = all critical checks passed.
Exit code 1 = one or more critical checks failed.
"""
 
import sys
import math
import time
 
# ─── Helpers ──────────────────────────────────────────────────────────────────
 
PASS = "\033[0;32m  ✓\033[0m"
FAIL = "\033[0;31m  ✗\033[0m"
WARN = "\033[1;33m  ⚠\033[0m"
INFO = "\033[0;36m  ·\033[0m"
 
failures = 0
 
def check(label, fn, warn_only=False):
    """Run fn(), print result. Returns True on pass."""
    global failures
    try:
        detail = fn()
        msg = f"{PASS}  {label}"
        if detail:
            msg += f"\n{INFO}     {detail}"
        print(msg)
        return True
    except Exception as exc:
        tag = WARN if warn_only else FAIL
        print(f"{tag}  {label}")
        print(f"{INFO}     {exc}")
        if not warn_only:
            failures += 1
        return False
 
def section(title):
    print(f"\n\033[1;34m── {title} ──\033[0m")
 
# ─── Section 1: Python & PyTorch ──────────────────────────────────────────────
section("1 / 6  Python + PyTorch")
 
check("Python version",
      lambda: f"Python {sys.version.split()[0]} (need 3.11+)"
              if sys.version_info >= (3, 11) else
              (_ for _ in ()).throw(RuntimeError(f"Got {sys.version.split()[0]}, need 3.11+")))
 
def _torch_version():
    import torch
    v = torch.__version__
    assert v.startswith("2."), f"Expected PyTorch 2.x, got {v}"
    return f"torch {v}"
 
def _cuda_available():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() = False")
    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{name}  |  {mem_gb:.1f} GB VRAM"
 
def _cuda_compute():
    """Actually run a matmul on GPU and time it."""
    import torch
    N = 2048
    a = torch.randn(N, N, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    b = a @ a.T
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000
    assert b.shape == (N, N)
    return f"{N}×{N} matmul in {elapsed:.1f} ms"
 
def _amp():
    import torch
    with torch.amp.autocast('cuda'):
        a = torch.randn(512, 512, device="cuda")
        b = a @ a.T
    return f"autocast OK — output dtype: {b.dtype}"
 
check("torch import (2.x)", _torch_version)
check("CUDA available", _cuda_available)
check("GPU compute (matmul)", _cuda_compute)
check("Mixed precision (AMP)", _amp)
 
# ─── Section 2: Hyperbolic math ───────────────────────────────────────────────
section("2 / 6  Hyperbolic math — critical stability checks")
 
def _arcosh_fp32():
    """arcosh should be stable in FP32 above 1.0 + epsilon."""
    import torch
    inputs = torch.tensor([1.0 + 1e-6, 1.5, 2.0, 5.0, 50.0],
                          device="cuda", dtype=torch.float32)
    out = torch.acosh(inputs)
    assert not torch.any(torch.isnan(out)), f"NaN detected: {out}"
    assert not torch.any(torch.isinf(out)), f"Inf detected: {out}"
    return f"arcosh({inputs.tolist()}) = {[f'{x:.4f}' for x in out.tolist()]}"
 
def _arcosh_fp16_hazard():
    """
    This is EXPECTED to produce NaN or garbage in FP16.
    It's a WARNING, not a failure — it documents why manifold ops
    must stay in FP32 even inside autocast blocks.
    """
    import torch
    x_fp32 = torch.tensor([1.0 + 1e-5], device="cuda", dtype=torch.float32)
    x_fp16 = x_fp32.half()  # quantises to exactly 1.0 in FP16
    result_fp16 = torch.acosh(x_fp16)
    is_nan = torch.isnan(result_fp16).item()
    # We raise to trigger the warn_only path
    if is_nan:
        raise RuntimeError(
            "arcosh(1.0) in FP16 = NaN — as expected. "
            "Fix: wrap manifold ops with .float() inside autocast blocks."
        )
    return "FP16 arcosh was stable this time (result may vary by GPU/driver)"
 
def _exp_log_roundtrip():
    """exp_map followed by log_map should recover the original vector."""
    import torch
 
    def exp_map(v: torch.Tensor, K: float = -1.0) -> torch.Tensor:
        """Lift Euclidean vector to Lorentz manifold."""
        sqrt_neg_K = math.sqrt(-K)
        norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x0 = torch.cosh(sqrt_neg_K * norm) / sqrt_neg_K
        xi = torch.sinh(sqrt_neg_K * norm) * v / (norm * sqrt_neg_K)
        return torch.cat([x0, xi], dim=-1)
 
    def log_map(x: torch.Tensor, K: float = -1.0) -> torch.Tensor:
        """Project Lorentz manifold point back to Euclidean tangent space."""
        sqrt_neg_K = math.sqrt(-K)
        x0 = x[..., 0:1]
        xi = x[..., 1:]
        arg = (sqrt_neg_K * x0).clamp(min=1.0 + 1e-6)
        scale = torch.acosh(arg) / (
            sqrt_neg_K * xi.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        )
        return scale * xi
 
    torch.manual_seed(42)
    v = torch.randn(64, 128, device="cuda", dtype=torch.float32)
 
    for K in [-0.5, -1.0, -2.0, -5.0]:
        x = exp_map(v, K=K)
        v_hat = log_map(x, K=K)
        err = (v - v_hat).abs().max().item()
        assert err < 1e-4, f"Round-trip error {err:.2e} at K={K} (threshold 1e-4)"
 
    return f"Max round-trip error < 1e-4 across K ∈ {{-0.5, -1, -2, -5}}"
 
def _gradient_flow():
    """Gradients must flow back through exp_map without NaN."""
    import torch
 
    def exp_map(v, K=-1.0):
        sqrt_neg_K = math.sqrt(-K)
        norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x0 = torch.cosh(sqrt_neg_K * norm) / sqrt_neg_K
        xi = torch.sinh(sqrt_neg_K * norm) * v / (norm * sqrt_neg_K)
        return torch.cat([x0, xi], dim=-1)
 
    v = torch.randn(32, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    x = exp_map(v)
    loss = x.pow(2).sum()
    loss.backward()
 
    assert v.grad is not None, "Gradient is None"
    assert not torch.any(torch.isnan(v.grad)), "NaN in gradient"
    assert not torch.any(torch.isinf(v.grad)), "Inf in gradient"
    return f"Gradient norm: {v.grad.norm().item():.4f}"
 
def _learnable_curvature():
    """
    Simulate the per-head K parameterisation:
    K_h = -exp(log_abs_K_h)  →  always negative, gradient flows.
    """
    import torch
 
    # 6 layers × 8 heads = 48 learnable curvature values
    log_abs_K = torch.nn.Parameter(
        torch.zeros(6, 8, device="cuda", dtype=torch.float32)
    )
    K = -log_abs_K.exp()
 
    assert (K < 0).all(), "All K must be negative"
    assert K.shape == (6, 8), f"Wrong shape: {K.shape}"
 
    # Gradient test
    loss = K.sum()
    loss.backward()
    assert log_abs_K.grad is not None, "No gradient to log_abs_K"
    assert not torch.any(torch.isnan(log_abs_K.grad)), "NaN in K gradient"
 
    k_range = f"[{K.min().item():.3f}, {K.max().item():.3f}]"
    return f"K shape (6 layers × 8 heads): {K.shape}  |  initial range: {k_range}"
 
def _multi_curvature_expmap():
    """
    Vectorised exp_map for multiple K values simultaneously —
    the key operation needed for per-head attention.
    """
    import torch
 
    B, S, H, D = 4, 64, 8, 32          # batch, seq, heads, head_dim
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
 
    # Per-head curvatures — shape (H,) → broadcast to (1,1,H,1)
    K_heads = -torch.rand(H, device="cuda").add(0.5)   # K in (-1.5, -0.5)
    sqrt_neg_K = (-K_heads).sqrt().reshape(1, 1, H, 1)  # (1,1,H,1)
 
    norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)   # (B,S,H,1)
    x0 = torch.cosh(sqrt_neg_K * norm) / sqrt_neg_K        # (B,S,H,1)
    xi = torch.sinh(sqrt_neg_K * norm) * v / (norm * sqrt_neg_K)
    x = torch.cat([x0, xi], dim=-1)                         # (B,S,H,D+1)
 
    assert x.shape == (B, S, H, D + 1), f"Shape mismatch: {x.shape}"
    assert not torch.any(torch.isnan(x)), "NaN in vectorised exp_map"
 
    return f"Vectorised exp_map: ({B},{S},{H},{D}) → ({B},{S},{H},{D+1}) — no NaN"
 
check("arcosh stability in FP32", _arcosh_fp32)
check("arcosh FP16 hazard (warning only — expected to fail)",
      _arcosh_fp16_hazard, warn_only=True)
check("exp_map / log_map round-trip (K=-0.5 to -5)", _exp_log_roundtrip)
check("gradient flow through exp_map", _gradient_flow)
check("learnable per-head curvature (6 layers × 8 heads)", _learnable_curvature)
check("vectorised multi-K exp_map (no head loop)", _multi_curvature_expmap)
 
# ─── Section 3: Lorentz attention score shape ─────────────────────────────────
section("3 / 6  Lorentz attention mechanics")
 
def _lorentz_inner():
    """Minkowski inner product: -x0*y0 + x1:·y1:"""
    import torch
 
    B, S, D = 4, 32, 65   # D = d_head + 1 (Lorentz dim)
    x = torch.randn(B, S, D, device="cuda")
    y = torch.randn(B, S, D, device="cuda")
 
    # <x,y>_L = -x0*y0 + sum(x1:*y1:)
    ip = -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)
    assert ip.shape == (B, S), f"Wrong shape: {ip.shape}"
    assert not torch.any(torch.isnan(ip))
    return f"Inner product shape: {ip.shape}"
 
def _attention_scores():
    """Full per-head attention score matrix."""
    import torch
 
    B, S, H, D = 2, 32, 4, 33    # D = d_head + 1
    Q = torch.randn(B, S, H, D, device="cuda")
    K = torch.randn(B, S, H, D, device="cuda")
 
    # Vectorised Lorentz inner product for all (query, key) pairs
    # Score[b,h,i,j] = -K_h * <Q[b,i,h,:], K[b,j,h,:]>_L
    q_t, q_s = Q[..., :1], Q[..., 1:]   # time/space split
    k_t, k_s = K[..., :1], K[..., 1:]
 
    # (B,H,S,S) inner products
    ip = (
        -torch.einsum("bihd,bjhd->bhij", q_t, k_t)
        + torch.einsum("bihd,bjhd->bhij", q_s, k_s)
    )
    K_curv = torch.tensor(1.0, device="cuda")   # |K| scale factor
    scores = K_curv * (-ip) / math.sqrt(D - 1)  # negate: higher score = closer
    attn = torch.softmax(scores, dim=-1)
 
    assert attn.shape == (B, H, S, S)
    assert not torch.any(torch.isnan(attn))
    assert (attn.sum(dim=-1) - 1.0).abs().max() < 1e-5, "Attn rows don't sum to 1"
    return f"Attention weights shape: {attn.shape}  |  row-sum error < 1e-5"
 
check("Lorentz inner product", _lorentz_inner)
check("Attention score matrix (vectorised)", _attention_scores)
 
# ─── Section 4: Geoopt ────────────────────────────────────────────────────────
section("4 / 6  Geoopt")
 
def _geoopt_import():
    import geoopt
    return f"geoopt {geoopt.__version__}"
 
def _geoopt_lorentz():
    import geoopt, torch
    manifold = geoopt.manifolds.Lorentz()
    # Create a point at the manifold origin
    pt = manifold.origin(64, dtype=torch.float32)
    assert pt.shape == (64,)
    # Project a tangent vector
    v = torch.randn(64)
    v_proj = manifold.proju(pt, v)
    assert v_proj.shape == (64,)
    return "Lorentz manifold origin + proju OK"
 
def _geoopt_riemannian_adam():
    import geoopt, torch
    manifold = geoopt.manifolds.PoincareBall()
    # A learnable point on the Poincaré ball
    pt = geoopt.ManifoldParameter(
        torch.zeros(32), manifold=manifold
    )
    opt = geoopt.optim.RiemannianAdam([pt], lr=1e-3)
    loss = pt.norm()
    loss.backward()
    opt.step()
    return "RiemannianAdam step completed"
 
check("geoopt import", _geoopt_import)
check("Lorentz manifold ops", _geoopt_lorentz)
check("RiemannianAdam optimiser", _geoopt_riemannian_adam)
 
# ─── Section 5: HuggingFace stack ─────────────────────────────────────────────
section("5 / 6  HuggingFace datasets + tokenizer")
 
def _hf_dataset():
    from datasets import load_dataset
    ds = load_dataset(
        "wikitext", "wikitext-2-raw-v1",
        split="train[:200]",
        trust_remote_code=True
    )
    total_chars = sum(len(row["text"]) for row in ds)
    return f"WikiText-2 (200 rows) — {total_chars:,} chars loaded"
 
def _hf_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    sample = "The quick brown fox jumps over the lazy dog"
    ids = tok.encode(sample)
    decoded = tok.decode(ids)
    assert decoded == sample, "Round-trip mismatch"
    return f"GPT-2 BPE: '{sample}' → {len(ids)} tokens → round-trip OK"
 
check("Load WikiText-2 (HuggingFace)", _hf_dataset)
check("GPT-2 tokenizer (encode/decode)", _hf_tokenizer)
 
# ─── Section 6: GPU memory report ─────────────────────────────────────────────
section("6 / 6  GPU memory budget")
 
def _memory_report():
    import torch
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1e9
    alloc_mb = torch.cuda.memory_allocated(0) / 1e6
    reserved_mb = torch.cuda.memory_reserved(0) / 1e6
 
    # Estimate budget for HyperNano (d=256, 6 layers)
    # Rough: params + grads + optimizer state + activations (BS=32, SL=256)
    param_mb = 40 * 3          # ~120 MB for 40M params × fp32 + Adam states
    act_mb = 32 * 256 * 257 * 6 * 4 / 1e6   # activations per layer
    est_total_mb = param_mb + act_mb
 
    lines = [
        f"VRAM total: {total_gb:.1f} GB",
        f"Currently allocated: {alloc_mb:.0f} MB  |  reserved: {reserved_mb:.0f} MB",
        f"Estimated HyperNano (d=256, 6L, BS=32): ~{est_total_mb:.0f} MB",
        f"Available headroom: ~{(props.total_memory - est_total_mb*1e6)/1e9:.1f} GB",
    ]
    return "\n".join(f"     {l}" for l in lines)
 
check("Memory budget", _memory_report)
 
# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "─" * 52)
if failures == 0:
    print("\033[0;32m  ✓  All critical checks passed.\033[0m")
    print("  Ready for Phase 1 — model components.")
else:
    print(f"\033[0;31m  ✗  {failures} critical check(s) failed.\033[0m")
    print("  Fix the failures above before proceeding.")
print("─" * 52 + "\n")
 
sys.exit(failures)