"""
test_blocks.py -- Unit tests for LorentzRMSNorm, LorentzFFN, HyperDecoderBlock
===============================================================================

Run with:
    cd ~/hyper-attn-nano
    source activate.sh
    pytest tests/test_blocks.py -v
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from manifolds import exp_map_origin, log_map_origin, lorentz_inner
from blocks    import LorentzRMSNorm, LorentzFFN, HyperDecoderBlock

D_MODEL = 64
D_FF    = 256
N_HEADS = 4
B, S    = 2, 12


def _lorentz_input(B=B, S=S, d_model=D_MODEL, K=-1.0):
    torch.manual_seed(42)
    v = torch.randn(B, S, d_model) * 0.1
    return exp_map_origin(v, K)


class TestLorentzRMSNorm:

    def test_output_shape(self):
        norm = LorentzRMSNorm(D_MODEL)
        x    = _lorentz_input()
        out  = norm(x)
        assert out.shape == (B, S, D_MODEL + 1), f"Expected {(B, S, D_MODEL+1)}, got {out.shape}"

    def test_output_on_manifold(self):
        norm  = LorentzRMSNorm(D_MODEL)
        x     = _lorentz_input()
        out   = norm(x)
        inner = lorentz_inner(out, out)
        max_err = (inner + 1.0).abs().max().item()
        assert max_err < 1e-3, f"Output not on K=-1 manifold; max err = {max_err:.2e}"

    def test_tangent_rms_approx_scale(self):
        norm  = LorentzRMSNorm(D_MODEL)
        x     = _lorentz_input()
        out   = norm(x)
        v     = log_map_origin(out, -1.0)
        rms   = v.pow(2).mean(dim=-1).sqrt()
        # RMS should match the current scale value (not necessarily 1.0)
        scale_val = norm.scale.data[0].item()
        max_err = (rms - scale_val).abs().max().item()
        assert max_err < 0.1, f"Post-norm tangent RMS deviates from scale={scale_val:.4f}; max err = {max_err:.2e}"

    def test_gradient_flows(self):
        norm = LorentzRMSNorm(D_MODEL)
        x    = _lorentz_input()
        norm(x).sum().backward()
        for name, param in norm.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    def test_scale_parameter_exists_and_shape(self):
        norm = LorentzRMSNorm(D_MODEL)
        assert hasattr(norm, "scale"), "LorentzRMSNorm has no scale attribute"
        assert isinstance(norm.scale, torch.nn.Parameter)
        assert norm.scale.shape == (D_MODEL,), f"scale shape wrong: {norm.scale.shape}"


class TestLorentzFFN:

    def test_output_shape(self):
        ffn = LorentzFFN(D_MODEL, D_FF)
        x   = _lorentz_input()
        out = ffn(x)
        assert out.shape == (B, S, D_MODEL + 1), f"Expected {(B, S, D_MODEL+1)}, got {out.shape}"

    def test_output_on_manifold(self):
        ffn   = LorentzFFN(D_MODEL, D_FF)
        x     = _lorentz_input()
        out   = ffn(x)
        inner = lorentz_inner(out, out)
        max_err = (inner + 1.0).abs().max().item()
        assert max_err < 1e-3, f"FFN output not on K=-1 manifold; max err = {max_err:.2e}"

    def test_gradient_flows(self):
        ffn = LorentzFFN(D_MODEL, D_FF)
        x   = _lorentz_input()
        ffn(x).sum().backward()
        for name, param in ffn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf in {name}"

    def test_no_bias_in_linear_layers(self):
        ffn = LorentzFFN(D_MODEL, D_FF)
        assert ffn.fc1.bias is None, "fc1 should have no bias"
        assert ffn.fc2.bias is None, "fc2 should have no bias"


class TestHyperDecoderBlock:

    def test_output_shape(self):
        block = HyperDecoderBlock(D_MODEL, N_HEADS, D_FF)
        x     = _lorentz_input()
        out   = block(x)
        assert out.shape == (B, S, D_MODEL + 1), f"Expected {(B, S, D_MODEL+1)}, got {out.shape}"

    def test_output_on_manifold(self):
        block = HyperDecoderBlock(D_MODEL, N_HEADS, D_FF)
        x     = _lorentz_input()
        out   = block(x)
        inner = lorentz_inner(out, out)
        max_err = (inner + 1.0).abs().max().item()
        assert max_err < 1e-2, f"Block output not on K=-1 manifold; max err = {max_err:.2e}"

    def test_gradient_flows(self):
        block = HyperDecoderBlock(D_MODEL, N_HEADS, D_FF)
        x     = _lorentz_input()
        block(x).sum().backward()
        for name, param in block.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf in {name}"

    def test_attn_K_accessible_and_negative(self):
        block  = HyperDecoderBlock(D_MODEL, N_HEADS, D_FF)
        K_vals = block.attn.K
        assert K_vals is not None
        assert (K_vals < 0).all(), f"Some K values non-negative: {K_vals}"

    def test_causal_output_independence(self):
        torch.manual_seed(0)
        block = HyperDecoderBlock(D_MODEL, N_HEADS, D_FF)
        block.eval()

        x = _lorentz_input(B=1, S=S)

        with torch.no_grad():
            out_orig = block(x)

        for i in range(S - 1):
            x_modified = x.clone()
            torch.manual_seed(99 + i)
            v_noise = torch.randn(1, S - i - 1, D_MODEL) * 0.1
            x_modified[0, i+1:, :] = exp_map_origin(v_noise, -1.0)
            with torch.no_grad():
                out_mod = block(x_modified)
            max_diff = (out_orig[0, i] - out_mod[0, i]).abs().max().item()
            assert max_diff < 1e-5, (
                f"Position {i} output changed when future tokens replaced "
                f"(max diff = {max_diff:.2e}) -- causal mask broken"
            )


if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", __file__, "-v"], check=True)
