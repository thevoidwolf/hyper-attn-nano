"""
curvature_schedule.py — Curvature annealing schedules for hyper-fixed models
=============================================================================

For hyper-fixed models, the fixed curvature K can be annealed across
training steps instead of being held constant from the start.

The motivation: K=-10 initialisation causes a ~100x early-training PPL
penalty while the model learns to operate in highly curved space. Starting
at K≈-1 (nearly flat) and annealing to K=-10 gives the model time to
establish good representations before the geometry becomes strongly curved.

Usage in YAML config (under model:):

    curvature_schedule:
      type: linear_warmup    # constant | linear_warmup | cosine_warmup | exponential
      k_start: -1.0
      k_end: -10.0
      warmup_steps: 500

If curvature_schedule is absent, train.py falls back to the existing
curvature: <float> field. If both are present, curvature_schedule wins
and a warning is logged.

For hyper-perhead models, schedule_mode controls behaviour:
  - frozen_schedule (default): overrides the learnable log_abs_K parameter
    entirely for the duration of warmup, then releases it.
  - init_only: parameter is frozen at the scheduled value until warmup
    completes, then released to the optimizer.
"""

import math


class CurvatureSchedule:
    """Base class. Subclasses implement k_at_step."""

    def k_at_step(self, step: int) -> float:
        raise NotImplementedError


class ConstantK(CurvatureSchedule):
    """K is fixed at a single value for all steps."""

    def __init__(self, k: float):
        assert k < 0, f"curvature must be negative, got {k}"
        self._k = k

    def k_at_step(self, step: int) -> float:
        return self._k


class LinearWarmupK(CurvatureSchedule):
    """
    K_start at step 0, linearly interpolated to K_end at warmup_steps,
    held constant at K_end thereafter.
    """

    def __init__(self, k_start: float, k_end: float, warmup_steps: int):
        assert k_start < 0, f"k_start must be negative, got {k_start}"
        assert k_end < 0, f"k_end must be negative, got {k_end}"
        assert warmup_steps > 0, f"warmup_steps must be positive, got {warmup_steps}"
        self.k_start = k_start
        self.k_end = k_end
        self.warmup_steps = warmup_steps

    def k_at_step(self, step: int) -> float:
        if step >= self.warmup_steps:
            return self.k_end
        t = step / self.warmup_steps
        return self.k_start + t * (self.k_end - self.k_start)


class CosineWarmupK(CurvatureSchedule):
    """
    K_start at step 0, cosine-interpolated to K_end at warmup_steps,
    held constant at K_end thereafter.
    """

    def __init__(self, k_start: float, k_end: float, warmup_steps: int):
        assert k_start < 0, f"k_start must be negative, got {k_start}"
        assert k_end < 0, f"k_end must be negative, got {k_end}"
        assert warmup_steps > 0, f"warmup_steps must be positive, got {warmup_steps}"
        self.k_start = k_start
        self.k_end = k_end
        self.warmup_steps = warmup_steps

    def k_at_step(self, step: int) -> float:
        if step >= self.warmup_steps:
            return self.k_end
        t = step / self.warmup_steps
        # Cosine interpolation: 0 at t=0, 1 at t=1
        cos_factor = 0.5 * (1.0 - math.cos(math.pi * t))
        return self.k_start + cos_factor * (self.k_end - self.k_start)


class ExponentialK(CurvatureSchedule):
    """
    K(t) = k_start * (k_end / k_start)^(min(t, warmup) / warmup)

    Exponential interpolation in log-curvature space.
    Held constant at K_end for steps >= warmup_steps.
    """

    def __init__(self, k_start: float, k_end: float, warmup_steps: int):
        assert k_start < 0, f"k_start must be negative, got {k_start}"
        assert k_end < 0, f"k_end must be negative, got {k_end}"
        assert warmup_steps > 0, f"warmup_steps must be positive, got {warmup_steps}"
        self.k_start = k_start
        self.k_end = k_end
        self.warmup_steps = warmup_steps
        # Precompute log ratio for efficiency
        self._log_ratio = math.log(abs(k_end) / abs(k_start))

    def k_at_step(self, step: int) -> float:
        if step >= self.warmup_steps:
            return self.k_end
        t = step / self.warmup_steps
        return self.k_start * math.exp(t * self._log_ratio)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_schedule(schedule_cfg: dict) -> CurvatureSchedule:
    """
    Construct a CurvatureSchedule from a config dict.

    Expected keys:
      type         : one of constant | linear_warmup | cosine_warmup | exponential
      k_start      : starting curvature (ignored for constant)
      k_end        : ending curvature (used as the single value for constant)
      warmup_steps : number of steps to anneal over (ignored for constant)

    Example:
        build_schedule({"type": "linear_warmup", "k_start": -1.0,
                        "k_end": -10.0, "warmup_steps": 500})
    """
    stype = schedule_cfg["type"]

    if stype == "constant":
        k = schedule_cfg.get("k_end", schedule_cfg.get("k_start"))
        if k is None:
            raise ValueError("constant schedule requires k_end (or k_start)")
        return ConstantK(k)

    k_start = schedule_cfg["k_start"]
    k_end = schedule_cfg["k_end"]
    warmup_steps = schedule_cfg["warmup_steps"]

    if stype == "linear_warmup":
        return LinearWarmupK(k_start, k_end, warmup_steps)
    elif stype == "cosine_warmup":
        return CosineWarmupK(k_start, k_end, warmup_steps)
    elif stype == "exponential":
        return ExponentialK(k_start, k_end, warmup_steps)
    else:
        raise ValueError(
            f"Unknown curvature schedule type: {stype!r}. "
            "Expected: constant | linear_warmup | cosine_warmup | exponential"
        )
