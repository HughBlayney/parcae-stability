"""Unit checks for the stability-extrapolation knobs (injection_fixed_decay, max_recurrence).

Run with:  PYTHONPATH=. python tests/test_stability_knobs.py   (CPU is fine; pytest also works)
"""
import math

import torch

import parcae_lm


def _cfg(**overrides):
    cfg = parcae_lm.create_config("parcae-small-140m", vocab_size=32768, prelude_norm=True, **overrides)
    return cfg


def test_default_config_unchanged():
    cfg = _cfg()
    assert cfg.injection_fixed_decay is None and cfg.max_recurrence is None
    m = cfg.construct_model()
    assert m.transformer.adapter.A_log is not None
    assert "transformer.adapter.A_log" in m.state_dict()
    # learned decay at init is the ssm_decay init target sqrt(0.2) in every channel
    assert abs(m.transformer.adapter.get_spectral_norm() - math.sqrt(0.2)) < 1e-4
    assert abs(m.transformer.adapter.get_contraction_factor() - math.sqrt(0.2)) < 1e-4


def test_max_recurrence_cap_per_sequence():
    cfg = _cfg(max_recurrence=16)
    m = cfg.construct_model()
    m.train()
    m.step = 0
    n, k = m._sample_batch_depths(100_000)
    total = n + k
    assert int(total.max()) <= 16 and int(total.min()) >= 1
    assert int(k.max()) <= cfg.mean_backprop_depth
    # Poisson(8) truncated to [1, 16] still has mean ~8 (cap removes ~0.4% of mass)
    assert abs(float(total.float().mean()) - 8.0) < 0.05, float(total.float().mean())
    # without the cap, 100k draws from Poisson(8) exceed 16 with overwhelming probability
    m2 = _cfg().construct_model()
    m2.train()
    m2.step = 0
    n2, k2 = m2._sample_batch_depths(100_000)
    assert int((n2 + k2).max()) > 16


def test_max_recurrence_cap_per_batch():
    # NOTE: upstream quirk in the per-batch sampler: for poisson-truncated-full it returns n = Poisson draw and
    # k = min(n, s) *without* subtracting k from n, so the loop runs n + k steps (mean ~12 for mean_recurrence=8).
    # The per-sequence path (used by every shipped Parcae config) subtracts k. We only cap n here, as in the plan.
    cfg = _cfg(max_recurrence=16, recurrent_iteration_method="per-batch")
    m = cfg.construct_model()
    m.train()
    ns = []
    for step in range(2000):
        m.step = step
        n, k = m.randomized_iteration_sampler()
        ns.append(int(n))
        assert int(k) <= cfg.mean_backprop_depth
    assert max(ns) <= 16 and min(ns) >= 1


def test_fixed_scheme_is_exactly_mean_recurrence():
    cfg = _cfg(sampling_scheme="fixed", max_recurrence=16)
    m = cfg.construct_model()
    m.train()
    m.step = 0
    n, k = m._sample_batch_depths(64)
    assert bool(((n + k) == cfg.mean_recurrence).all()) and bool((k == cfg.mean_backprop_depth).all())


def test_fixed_decay_model():
    cfg = _cfg(injection_fixed_decay=1.25)
    m = cfg.construct_model()
    adapter = m.transformer.adapter
    assert adapter.A_log is None
    assert "transformer.adapter.A_log" not in m.state_dict()
    assert adapter.get_spectral_norm() == 1.25 and adapter.get_contraction_factor() == 1.25
    # parameter count differs from the learned control by exactly the state dim
    n_learned = sum(p.numel() for p in _cfg().construct_model().parameters())
    n_fixed = sum(p.numel() for p in m.parameters())
    assert n_learned - n_fixed == cfg.recurrent_embedding_dimension
    # forward/backward works and the injection parameters receive gradients
    x = torch.randint(0, 32768, (2, 64))
    out = m(x, labels=x, num_steps_pair=torch.tensor([4, 4]))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert adapter.B.grad is not None and adapter.dt_bias.grad is not None
    # the adapter really applies the scalar: x_{t+1} = rho * x_t + dt * (e @ B.T)
    x_t = torch.randn(3, 5, cfg.recurrent_embedding_dimension)
    e = torch.randn(3, 5, cfg.n_embd)
    with torch.no_grad():
        dt = torch.nn.functional.softplus(adapter.dt_bias)
        expected = 1.25 * x_t + dt * (e @ adapter.B.T)
        assert torch.allclose(adapter(x_t, e), expected, atol=1e-5)


def test_fixed_decay_telemetry_and_state_growth():
    cfg = _cfg(injection_fixed_decay=1.25)
    m = cfg.construct_model()
    m.extreme_monitoring = True
    metrics = m.compute_extreme_metrics()
    assert metrics["extreme/adapter_decay_max"] == 1.25 and metrics["extreme/adapter_decay_mean"] == 1.25
    learned = _cfg().construct_model().compute_extreme_metrics()
    assert abs(learned["extreme/adapter_decay_max"] - math.sqrt(0.2)) < 1e-4


def test_fixed_decay_roundtrips_through_config_and_state_dict():
    cfg = _cfg(injection_fixed_decay=0.5, max_recurrence=16)
    m = cfg.construct_model()
    sd = m.state_dict()
    # rebuilding from the same overrides (what eval.py does from run_config.json) loads strictly
    m2 = _cfg(injection_fixed_decay=0.5, max_recurrence=16).construct_model()
    m2.load_state_dict(sd, strict=True)
    # rebuilding WITHOUT the override must be caught by the checked loader
    from receval.models.parcae import load_state_dict_checked
    m3 = _cfg().construct_model()
    try:
        load_state_dict_checked(m3, sd)
    except RuntimeError as e:
        assert "A_log" in str(e)
    else:
        raise AssertionError("load_state_dict_checked should reject a checkpoint without A_log for a learned model")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
