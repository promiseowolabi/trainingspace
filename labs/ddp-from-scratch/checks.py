"""Lab 10 checks (local lane: needs PyTorch). Run by pytest and verify.py.

Each check receives your module as ``m``. Check ids (the function name
without ``check_``) must match src/data/labs.ts. The distributed checks start
2 processes with the ``spawn`` method, connect them over ``gloo`` on
127.0.0.1, and compare what they computed with one process training on the
whole batch. A check that waits more than 90 s is failed as "hung": usually
one rank skipped a collective the other is waiting in.
"""

import importlib
import multiprocessing as mp
import os
import socket
import sys
import tempfile
import traceback

import torch
import torch.nn as nn

WORLD = 2
STEPS = 5


def _model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(10, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))


def _data():
    g = torch.Generator().manual_seed(123)
    x = torch.randn(16, 10, generator=g)
    return x, x[:, :1] * 2 - x[:, 1:2] + 0.1 * torch.randn(16, 1, generator=g)


def _reference(steps=STEPS):
    model = _model(0)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x, y = _data()
    for _ in range(steps):
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    return [p.detach().clone() for p in model.parameters()]


# ---- scenarios: run inside each spawned rank -------------------------------

def _scenario_broadcast(m, rank):
    model = _model(rank)  # rank 1 starts from different weights
    m.broadcast_params(model)
    return {"params": [p.detach().clone() for p in model.parameters()]}


def _scenario_train(m, rank, bucket_bytes=None):
    model = _model(rank)
    m.broadcast_params(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x, y = _data()
    for _ in range(STEPS):
        m.ddp_step(model, opt, x, y, nn.functional.mse_loss, bucket_bytes=bucket_bytes)
    return {"params": [p.detach().clone() for p in model.parameters()]}


def _scenario_bucket_calls(m, rank):
    model = _model(0)
    x, y = _data()
    nn.functional.mse_loss(model(x), y).backward()
    calls = {b: m.allreduce_grads_bucketed(model, b) for b in (1, 600, 10**9)}
    return {"calls": calls}


SCENARIOS = {
    "broadcast": _scenario_broadcast,
    "train": _scenario_train,
    "train_bucketed": lambda m, rank: _scenario_train(m, rank, bucket_bytes=600),
    "bucket_calls": _scenario_bucket_calls,
}


def _worker(rank, port, module, scenario, out):
    # Pin gloo to the loopback interface: otherwise it tries to resolve the
    # machine's hostname first, which can stall for seconds on a laptop.
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    try:
        m = importlib.import_module(module)
        m.setup(rank, WORLD, port)
        try:
            result = SCENARIOS[scenario](m, rank)
        finally:
            try:
                m.cleanup()
            except Exception:  # noqa: BLE001
                pass
        torch.save(result, os.path.join(out, f"r{rank}.pt"))
    except BaseException:  # noqa: BLE001
        with open(os.path.join(out, f"r{rank}.err"), "w") as f:
            f.write(traceback.format_exc())


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn(m, scenario, timeout=90):
    out = tempfile.mkdtemp()
    ctx = mp.get_context("spawn")
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, port, m.__name__, scenario, out)) for r in range(WORLD)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout)
    if any(p.is_alive() for p in procs):
        for p in procs:
            p.terminate()
        raise AssertionError(f"{scenario}: the ranks hung for {timeout} s — does every rank call the same collectives in the same order?")
    results = []
    for r in range(WORLD):
        err = os.path.join(out, f"r{r}.err")
        if os.path.exists(err):
            tb = open(err).read().strip().splitlines()
            raise AssertionError(f"rank {r} failed: {tb[-1]}")
        results.append(torch.load(os.path.join(out, f"r{r}.pt")))
    return results


# ---- checks ------------------------------------------------------------------

def check_shard_batch(m):
    x = torch.arange(16).view(8, 2)
    y = torch.arange(8)
    x0, y0 = m.shard_batch(x, y, 0, 4)
    x3, y3 = m.shard_batch(x, y, 3, 4)
    assert torch.equal(x0, x[:2]) and torch.equal(y0, y[:2]), "rank 0 gets the first B/W rows"
    assert torch.equal(x3, x[6:]) and torch.equal(y3, y[6:]), "rank 3 of 4 gets the last 2 rows"


def check_plan_buckets(m):
    assert m.plan_buckets([100, 100, 300, 50], 250) == [[0, 1], [2], [3]]
    assert m.plan_buckets([10, 10, 10], 10**9) == [[0, 1, 2]], "a large limit puts everything in one bucket"
    assert m.plan_buckets([10, 10, 10], 1) == [[0], [1], [2]], "a gradient larger than the limit still gets a bucket"


def check_broadcast_sync(m):
    r0, r1 = _spawn(m, "broadcast")
    init0 = [p.detach() for p in _model(0).parameters()]
    for a, b, c in zip(r0["params"], r1["params"], init0):
        assert torch.equal(a, b), "after broadcast_params both ranks must hold identical parameters"
        assert torch.equal(a, c), "…and they must be rank 0's (src=0)"


def check_matches_single_process(m):
    ranks = _spawn(m, "train")
    ref = _reference()
    for r, res in enumerate(ranks):
        for i, (a, b) in enumerate(zip(res["params"], ref)):
            assert torch.allclose(a, b, atol=1e-6), (
                f"rank {r}, parameter {i}: 2 ranks × 8 examples must train exactly like 1 process × 16 — "
                "average (not sum) the gradients, and broadcast before the first step"
            )


def check_bucketed_matches(m):
    ranks = _spawn(m, "train_bucketed")
    ref = _reference()
    for r, res in enumerate(ranks):
        for a, b in zip(res["params"], ref):
            assert torch.allclose(a, b, atol=1e-6), f"rank {r}: bucketed all-reduce must give the same parameters as per-tensor all-reduce"


def check_bucket_calls(m):
    r0, _ = _spawn(m, "bucket_calls")
    sizes = [p.numel() * 4 for p in _model(0).parameters()]
    for b, calls in r0["calls"].items():
        want = len(m.plan_buckets(sizes, b))
        assert calls == want, f"bucket_bytes={b}: expected {want} all_reduce calls, you returned {calls}"
    assert r0["calls"][10**9] == 1 and r0["calls"][1] == len(sizes), "one call for a huge bucket, one per tensor for a tiny one"


CHECKS = [
    check_shard_batch,
    check_plan_buckets,
    check_broadcast_sync,
    check_matches_single_process,
    check_bucketed_matches,
    check_bucket_calls,
]
