"""
test_model.py -- Unit tests for HyperAttnNano and GPTNano
=========================================================

Run with:
    cd ~/hyper-attn-nano
    source activate.sh
    pytest tests/test_model.py -v
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import HyperAttnNano, GPTNano, NANO_CONFIG

B, S = 2, 32


def _ids():
    torch.manual_seed(0)
    return torch.randint(0, NANO_CONFIG["vocab_size"], (B, S))


class TestHyperAttnNano:

    def test_logit_shape_no_targets(self):
        model  = HyperAttnNano(NANO_CONFIG)
        logits, loss = model(_ids())
        assert logits.shape == (B, S, NANO_CONFIG["vocab_size"])
        assert loss is None

    def test_loss_with_targets(self):
        model  = HyperAttnNano(NANO_CONFIG)
        logits, loss = model(_ids(), _ids())
        assert logits.shape == (B, S, NANO_CONFIG["vocab_size"])
        assert loss is not None
        assert loss.shape == ()

    def test_loss_finite_and_positive(self):
        _, loss = HyperAttnNano(NANO_CONFIG)(_ids(), _ids())
        assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
        assert loss.item() > 0, f"Loss should be positive: {loss.item()}"

    def test_gradient_flows_to_embed_and_curvature(self):
        model  = HyperAttnNano(NANO_CONFIG)
        _, loss = model(_ids(), _ids())
        loss.backward()
        assert model.embed.weight.grad is not None
        assert not torch.isnan(model.embed.weight.grad).any()
        found = False
        for name, param in model.named_parameters():
            if "log_abs_K" in name:
                assert param.grad is not None, f"No grad for {name}"
                assert not torch.isnan(param.grad).any()
                found = True
                break
        assert found, "No log_abs_K parameter found"

    def test_get_curvatures_count_and_sign(self):
        model = HyperAttnNano(NANO_CONFIG)
        curvs = model.get_curvatures()
        expected = NANO_CONFIG["n_layers"] * NANO_CONFIG["n_heads"]
        assert len(curvs) == expected, f"Expected {expected} entries, got {len(curvs)}"
        for k, v in curvs.items():
            assert v < 0, f"Curvature {k}={v} not negative"

    def test_weight_tying(self):
        model = HyperAttnNano(NANO_CONFIG)
        assert model.lm_head.weight is model.embed.weight

    def test_no_nan_in_logits_or_loss(self):
        model  = HyperAttnNano(NANO_CONFIG)
        logits, loss = model(_ids(), _ids())
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        assert not torch.isnan(loss)


class TestGPTNano:

    def test_logit_shape_no_targets(self):
        model  = GPTNano(NANO_CONFIG)
        logits, loss = model(_ids())
        assert logits.shape == (B, S, NANO_CONFIG["vocab_size"])
        assert loss is None

    def test_loss_with_targets(self):
        model  = GPTNano(NANO_CONFIG)
        logits, loss = model(_ids(), _ids())
        assert logits.shape == (B, S, NANO_CONFIG["vocab_size"])
        assert loss is not None
        assert loss.shape == ()

    def test_loss_finite_and_positive(self):
        _, loss = GPTNano(NANO_CONFIG)(_ids(), _ids())
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_get_curvatures_empty(self):
        assert GPTNano(NANO_CONFIG).get_curvatures() == {}

    def test_no_nan_in_logits_or_loss(self):
        model  = GPTNano(NANO_CONFIG)
        logits, loss = model(_ids(), _ids())
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        assert not torch.isnan(loss)

    def test_weight_tying(self):
        model = GPTNano(NANO_CONFIG)
        assert model.lm_head.weight is model.embed.weight

    def test_forward_completes_float32(self):
        model  = GPTNano(NANO_CONFIG)
        with torch.no_grad():
            logits, _ = model(_ids())
        assert logits.dtype == torch.float32


class TestParameterEquivalence:

    def test_param_count_difference_equals_K_params(self):
        hyper  = HyperAttnNano(NANO_CONFIG)
        euclid = GPTNano(NANO_CONFIG)
        hyper_total  = sum(p.numel() for p in hyper.parameters())
        euclid_total = sum(p.numel() for p in euclid.parameters())
        expected_diff = NANO_CONFIG["n_layers"] * NANO_CONFIG["n_heads"]
        actual_diff   = hyper_total - euclid_total
        assert actual_diff == expected_diff, (
            f"Param diff: expected {expected_diff}, got {actual_diff}. "
            f"HyperAttnNano={hyper_total}, GPTNano={euclid_total}"
        )


if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", __file__, "-v"], check=True)
