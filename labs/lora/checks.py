"""Lab 12 checks (local lane: needs PyTorch). Run by pytest and verify.py.

Each check receives your module as ``m``. Check ids (the function name
without ``check_``) must match src/data/labs.ts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPT, GPTConfig


def _base(seed=0, i=16, o=24):
    torch.manual_seed(seed)
    return nn.Linear(i, o)


def check_zero_init(m):
    base = _base()
    x = torch.randn(5, 16)
    want = base(x).detach()
    lora = m.LoRALinear(base, r=4, alpha=8)
    assert lora.A.shape == (4, 16) and lora.B.shape == (24, 4), f"A must be (r, in) = (4, 16) and B (out, r) = (24, 4); got {tuple(lora.A.shape)}, {tuple(lora.B.shape)}"
    assert torch.count_nonzero(lora.B) == 0, "B starts at zero"
    assert torch.count_nonzero(lora.A) > 0, "A starts random (otherwise B would never get a gradient)"
    assert torch.allclose(lora(x), want), "with B = 0 the wrapped layer must compute exactly what the base layer did"


def check_frozen_base(m):
    lora = m.LoRALinear(_base(), r=4, alpha=8)
    assert not lora.base.weight.requires_grad and not lora.base.bias.requires_grad, "freeze the base weight and bias"
    trainable = {n for n, p in lora.named_parameters() if p.requires_grad}
    assert trainable == {"A", "B"}, f"only A and B may train; trainable: {sorted(trainable)}"
    x = torch.randn(3, 16)
    lora(x).sum().backward()
    assert lora.base.weight.grad is None, "no gradient may reach the frozen weight"
    assert lora.B.grad is not None and torch.count_nonzero(lora.B.grad) > 0, "B must receive a gradient on the first step"


def check_scale(m):
    base = _base(1)
    lora = m.LoRALinear(base, r=4, alpha=16)
    with torch.no_grad():
        lora.B.normal_()
    x = torch.randn(2, 16)
    want = base(x) + 4.0 * (x @ lora.A.T @ lora.B.T)
    assert abs(lora.scale - 4.0) < 1e-12, "scale = alpha / r = 16 / 4 = 4"
    assert torch.allclose(lora(x), want, atol=1e-5), "forward = base(x) + scale · x·Aᵀ·Bᵀ"


def check_merge_equal(m):
    lora = m.LoRALinear(_base(2), r=3, alpha=6)
    with torch.no_grad():
        lora.B.normal_()
    merged = lora.merged()
    assert isinstance(merged, nn.Linear) and not isinstance(merged, m.LoRALinear), "merged() returns a plain nn.Linear"
    x = torch.randn(7, 16)
    assert torch.allclose(merged(x), lora(x), atol=1e-5), "the merged layer must compute the same function as base + LoRA"


def check_apply_counts(m):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=146, block_size=64, n_layer=4, n_head=4, n_embd=128)
    model = m.apply_lora(GPT(cfg), targets=("qkv", "proj"), r=8, alpha=16)
    wrapped = [n for n, mod in model.named_modules() if isinstance(mod, m.LoRALinear)]
    # "proj" also matches "fc_proj" — both are attention/MLP output projections
    assert len(wrapped) == 3 * 4, f"qkv, proj and fc_proj in each of 4 blocks = 12 LoRA layers; got {len(wrapped)}: {wrapped[:4]}…"
    trainable, total = m.count_params(model)
    per_block = 8 * (128 + 384) + 8 * (128 + 128) + 8 * (512 + 128)
    assert trainable == 4 * per_block, f"r·(in + out) per wrapped layer: expected {4 * per_block:,} trainable, got {trainable:,}"
    base_total = sum(p.numel() for p in GPT(cfg).parameters())
    assert total == base_total + trainable, f"total must be the base model's {base_total:,} plus the LoRA parameters"
    assert trainable / total < 0.2, "LoRA trains a small fraction of the model"


def check_sft_masking(m):
    torch.manual_seed(3)
    logits = torch.randn(2, 6, 10)
    targets = torch.randint(0, 10, (2, 6))
    mask = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1]], dtype=torch.bool)
    got = m.sft_loss(logits, targets, mask)
    per = F.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1), reduction="none").reshape(2, 6)
    want = per[mask].mean()
    assert torch.allclose(got, want, atol=1e-6), "mean cross-entropy over response positions only"
    logits2 = logits.clone()
    logits2[~mask] = 100.0 * torch.randn_like(logits2[~mask])
    assert torch.allclose(m.sft_loss(logits2, targets, mask), want, atol=1e-6), "prompt positions must not affect the loss"


def check_learns_low_rank(m):
    torch.manual_seed(4)
    base = nn.Linear(32, 32)
    W0 = base.weight.detach().clone()
    u, v = torch.randn(32, 1), torch.randn(1, 32)
    target = nn.Linear(32, 32)
    with torch.no_grad():
        target.weight.copy_(W0 + 0.5 * u @ v)
        target.bias.copy_(base.bias)
    lora = m.LoRALinear(base, r=2, alpha=2)
    opt = torch.optim.Adam([lora.A, lora.B], lr=1e-2)
    x = torch.randn(256, 32)
    y = target(x).detach()
    start = F.mse_loss(lora(x), y).item()
    for _ in range(400):
        loss = F.mse_loss(lora(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.equal(base.weight, W0), "the base weight must not change during fine-tuning"
    assert loss.item() < 1e-3 * start, f"a rank-2 adapter should learn a rank-1 change: loss {start:.3f} → {loss.item():.5f}"


CHECKS = [
    check_zero_init,
    check_frozen_base,
    check_scale,
    check_merge_equal,
    check_apply_counts,
    check_sft_masking,
    check_learns_low_rank,
]
