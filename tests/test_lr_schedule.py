"""[sim2real fork] The optional cosine LR decay (OptimConfig.lr_decay_*); the default stays upstream's schedule."""

import math

import pytest

from abc_minimal.config import OptimConfig, lr_factor


def upstream(step, warmup=1000):
    return min((step + 1) / warmup, 1.0)


def test_default_is_upstream_warmup_then_constant():
    optim = OptimConfig()
    for step in (0, 1, 499, 999, 1000, 50_000, 100_000, 10**7):
        assert lr_factor(step, optim) == upstream(step)


def test_cosine_anneal_100k_to_120k():
    optim = OptimConfig(lr_decay_start=100_000, lr_decay_steps=20_000, lr_decay_min_ratio=0.1)
    assert lr_factor(99_999, optim) == 1.0
    assert lr_factor(100_000, optim) == 1.0
    assert lr_factor(110_000, optim) == pytest.approx(0.55)
    assert lr_factor(120_000, optim) == pytest.approx(0.1)
    assert lr_factor(150_000, optim) == pytest.approx(0.1)
    values = [lr_factor(s, optim) for s in range(100_000, 120_001, 500)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert optim.learning_rate * lr_factor(120_000, optim) == pytest.approx(1e-5)


def test_cosine_keeps_warmup():
    optim = OptimConfig(lr_decay_start=0, lr_decay_steps=10, lr_decay_min_ratio=0.0)
    assert lr_factor(0, optim) == pytest.approx(upstream(0))
    assert lr_factor(5, optim) == pytest.approx(upstream(5) * 0.5 * (1 + math.cos(math.pi * 0.5)))


def test_resumed_lambdalr_follows_the_new_schedule():
    """A checkpoint saved under the constant schedule, resumed with the decay: LambdaLR restores the step count and
    the new lambda applies from the next step (the lambda itself is not part of the saved state)."""
    torch = pytest.importorskip("torch")
    param = torch.nn.Parameter(torch.zeros(1))

    def make(optim):
        opt = torch.optim.AdamW([param], lr=optim.learning_rate)
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, optim))

    opt, sched = make(OptimConfig(lr_warmup_steps=2))
    for _ in range(100):
        opt.step(); sched.step()
    state = {"optimizer": opt.state_dict(), "scheduler": sched.state_dict()}

    anneal = OptimConfig(lr_warmup_steps=2, lr_decay_start=100, lr_decay_steps=20, lr_decay_min_ratio=0.1)
    opt2, sched2 = make(anneal)
    opt2.load_state_dict(state["optimizer"]); sched2.load_state_dict(state["scheduler"])
    assert opt2.param_groups[0]["lr"] == pytest.approx(1e-4)
    lrs = []
    for _ in range(20):
        opt2.step(); sched2.step(); lrs.append(opt2.param_groups[0]["lr"])
    assert lrs[9] == pytest.approx(1e-4 * 0.55)
    assert lrs[-1] == pytest.approx(1e-5)
