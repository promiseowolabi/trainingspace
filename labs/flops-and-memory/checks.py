"""Lab 08 checks. The same file runs under pytest, verify.py and the browser.

Each check receives your module as ``m``. Check ids (the function name
without ``check_``) must match src/data/labs.ts. Reference values are the
published figures each lesson cites, or arithmetic stated in the message.
"""

import math


def close(a, b, rel=1e-6):
    return math.isclose(float(a), float(b), rel_tol=rel)


def check_param_count(m):
    assert m.param_count(50257, 1024, 12, 768) == 124_439_808, "GPT-2 small: 124,439,808 (M1.L5)"
    assert m.param_count(50257, 1024, 48, 1600) == 1_557_611_200, "GPT-2 XL: 1,557,611,200"
    assert m.param_count(100, 64, 2, 32, tied=False) - m.param_count(100, 64, 2, 32) == 3200, "untied adds exactly V·C"


def check_train_flops(m):
    got = m.train_flops(175e9, 300e9)
    assert close(got, 3.15e23), f"GPT-3: 6 × 175e9 × 300e9 = 3.15e23 FLOPs, got {got:.3g}"


def check_forward_per_token(m):
    got = m.forward_flops_per_token(124_439_808, 12, 1024, 768)
    assert close(got, 267_753_984), f"GPT-2 small at T=1024: 2N + 2·L·T·C = 267,753,984, got {got:,.0f}"


def check_mfu(m):
    got = m.mfu(7e9, 3500, 989e12)
    assert close(got, 0.14863498, rel=1e-6), f"7B at 3,500 tokens/s on one H100 (989 TFLOP/s bf16): MFU ≈ 0.149, got {got:.4f}"
    assert 0 < got < 1, "MFU is a fraction"


def check_training_days(m):
    got = m.training_days(7e9, 2e12, 1024, 989e12, 0.4)
    assert abs(got - 2.40) < 0.01, f"7B on 2T tokens, 1,024 H100s at 40% MFU: ≈ 2.40 days, got {got:.3f}"


def check_state_memory(m):
    assert close(m.state_memory_gb(7e9), 112.0), "16 bytes/param: 7B → 112 GB (T0.L4)"
    assert close(m.state_memory_gb(7e9, bytes_per_param=2), 14.0), "bf16 weights alone: 14 GB"


def check_activation_memory(m):
    full = m.activation_memory_gb(32, 2048, 1, 4096, 32)
    flash = m.activation_memory_gb(32, 2048, 1, 4096, 32, flash=True)
    assert abs(full - 30.60) < 0.01, f"7B shapes, s=2048, b=1: ≈ 30.6 GB without FlashAttention, got {full:.2f}"
    assert abs(flash - 9.13) < 0.01, f"with FlashAttention the s² term goes: ≈ 9.13 GB, got {flash:.2f}"


def check_chinchilla(m):
    n, d = m.chinchilla_optimal(5.76e23)
    assert abs(n / 1e9 - 69.3) < 0.1, f"Chinchilla's budget (5.76e23 FLOPs) → N ≈ 69.3 B (they trained 70 B), got {n / 1e9:.1f} B"
    assert abs(d / 1e12 - 1.386) < 0.01, f"… and D ≈ 1.39 T tokens (they used 1.4 T), got {d / 1e12:.3f} T"
    assert close(6 * n * d, 5.76e23), "N and D must spend exactly the budget: 6·N·D = C"


CHECKS = [
    check_param_count,
    check_train_flops,
    check_forward_per_token,
    check_mfu,
    check_training_days,
    check_state_memory,
    check_activation_memory,
    check_chinchilla,
]
