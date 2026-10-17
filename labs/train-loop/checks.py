"""Lab 07 checks (local lane: needs PyTorch). Run by pytest and verify.py.

Each check receives your module as ``m``. Check ids (the function name
without ``check_``) must match src/data/labs.ts. Everything runs on CPU in
float32 except the one bf16 check; the whole file takes about 10 s.
"""

import math
import os
import tempfile

import torch

import data as dataset
from model import GPT, GPTConfig

TINY = dict(n_layer=2, n_head=4, n_embd=64, block_size=64)
D = dataset.load()


def _model(seed=0, **over):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=D["vocab_size"], **{**TINY, **over}))


def check_get_batch(m):
    data = torch.arange(1000)
    g = torch.Generator().manual_seed(0)
    x, y = m.get_batch(data, 16, 8, g)
    assert x.shape == (8, 16) and y.shape == (8, 16), f"shapes {tuple(x.shape)}, {tuple(y.shape)}; expected (8, 16) each"
    assert torch.equal(y[:, :-1], x[:, 1:]), "y must be x shifted left by one: y[t] is the token after x[t]"
    assert torch.equal(y[:, -1], x[:, -1] + 1), "with data = arange, the last target is the last input + 1"
    assert int(x.max()) < 1000 and int(y.max()) < 1000, "windows must stay inside data"
    x2, _ = m.get_batch(data, 16, 8, torch.Generator().manual_seed(0))
    assert torch.equal(x, x2), "the same generator state must give the same batch — use only `generator` for randomness"


def check_optimizer_groups(m):
    model = _model()
    opt = m.configure_optimizer(model, lr=1e-3, weight_decay=0.1)
    assert isinstance(opt, torch.optim.AdamW), "use torch.optim.AdamW (decoupled weight decay, O1.L3)"
    decayed = {id(p) for g in opt.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            assert id(p) in decayed, f"{name} is a matrix and should be decayed"
        else:
            assert id(p) not in decayed, f"{name} is a bias or norm gain and must not be decayed"
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_opt == model.num_params(), "every parameter must be in exactly one group"
    assert opt.param_groups[0]["betas"] == (0.9, 0.95)


def check_initial_loss(m):
    model = _model()
    loss = m.evaluate(model, D["val"], 64, 16, 4, torch.Generator().manual_seed(1))
    want = math.log(D["vocab_size"])
    assert isinstance(loss, float), "evaluate must return a Python float"
    assert abs(loss - want) < 0.15, f"a fresh model should score ≈ ln V = {want:.2f}; evaluate returned {loss:.3f} (E1.L2)"
    assert model.training, "evaluate must restore train mode if the model was training"


def check_accumulation(m):
    g = torch.Generator().manual_seed(2)
    x, y = m.get_batch(D["train"], 64, 8, g)
    a, b = _model(3), _model(3)
    sa = torch.optim.SGD(a.parameters(), lr=0.5)
    sb = torch.optim.SGD(b.parameters(), lr=0.5)
    la, _ = m.train_step(a, sa, [(x, y)], grad_clip=1e9)
    lb, _ = m.train_step(b, sb, [(x[:4], y[:4]), (x[4:], y[4:])], grad_clip=1e9)
    assert abs(la - lb) < 1e-5, f"mean loss: one batch {la:.6f} vs two micro-batches {lb:.6f}"
    for (n, pa), pb in zip(a.named_parameters(), b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6), (
            f"{n}: two micro-batches of 4 must update the weights exactly like one batch of 8 — divide each loss by the number of micro-batches"
        )
    assert all(p.grad is None for p in a.parameters()), "zero the gradients after the step (set_to_none=True)"


def check_clipping(m):
    model = _model(4)
    before = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    batch = m.get_batch(D["train"], 64, 8, torch.Generator().manual_seed(5))
    _, norm = m.train_step(model, opt, [batch], grad_clip=0.01)
    assert isinstance(norm, float) and norm > 0.01, f"return the norm BEFORE clipping (it should exceed 0.01 here), got {norm}"
    moved = math.sqrt(sum(float(((p.detach() - q) ** 2).sum()) for p, q in zip(model.parameters(), before)))
    assert moved <= 0.0101, f"with SGD lr 1 and clip 0.01 the update norm must be ≤ 0.01; it was {moved:.4f}"


def check_bf16_master(m):
    model = _model(6)
    batch = m.get_batch(D["train"], 64, 8, torch.Generator().manual_seed(7))
    with torch.no_grad():
        _, ref = model(*batch)
    opt = m.configure_optimizer(model, lr=1e-3, weight_decay=0.0)
    loss, _ = m.train_step(model, opt, [batch], amp=True)
    assert all(p.dtype == torch.float32 for p in model.parameters()), (
        "autocast must leave the parameters in float32 — they are the master weights (P1.L3)"
    )
    assert abs(loss - float(ref)) < 0.05, f"bf16 loss {loss:.4f} should be close to the fp32 loss {float(ref):.4f}"


def check_learns(m):
    model = _model(8)
    opt = m.configure_optimizer(model, lr=3e-3, weight_decay=0.1)
    g = torch.Generator().manual_seed(9)
    for s in range(300):
        for group in opt.param_groups:
            group["lr"] = m.lr_at(s, 3e-3, 30, 300, 3e-4)
        m.train_step(model, opt, [m.get_batch(D["train"], 64, 32, g)])
    val = m.evaluate(model, D["val"], 64, 32, 10, torch.Generator().manual_seed(1))
    assert val < 2.9, f"after 300 steps validation loss should be < 2.9 (the reference reaches 2.64, from ln 146 = 4.98); got {val:.3f}"


def check_resume_exact(m):
    def run(n, model, opt, g):
        for _ in range(n):
            m.train_step(model, opt, [m.get_batch(D["train"], 64, 8, g)])

    a = _model(10)
    oa = m.configure_optimizer(a, 1e-3, 0.1)
    run(20, a, oa, torch.Generator().manual_seed(11))

    b = _model(10)
    ob = m.configure_optimizer(b, 1e-3, 0.1)
    gb = torch.Generator().manual_seed(11)
    run(10, b, ob, gb)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.pt")
        m.save_checkpoint(path, b, ob, 10, gb)
        c = _model(99)  # different init: everything must come from the checkpoint
        oc = m.configure_optimizer(c, 1e-3, 0.1)
        gc = torch.Generator().manual_seed(12345)
        step = m.load_checkpoint(path, c, oc, gc)
    assert step == 10, f"load_checkpoint must return the saved step, got {step}"
    run(10, c, oc, gc)
    for (n, pa), pc in zip(a.named_parameters(), c.parameters()):
        assert torch.equal(pa, pc), (
            f"{n}: 10 + resume + 10 steps differs from 20 straight steps — the checkpoint must hold the model, "
            "the optimizer state (Adam's m and v) and the batch generator's state"
        )


CHECKS = [
    check_get_batch,
    check_optimizer_groups,
    check_initial_loss,
    check_accumulation,
    check_clipping,
    check_bf16_master,
    check_learns,
    check_resume_exact,
]
