"""Lab 01 checks. The same file runs under pytest, verify.py and the browser.

Each check receives your module as ``m``. Check ids (the function name
without ``check_``) must match src/data/labs.ts.
"""

import math
import random


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def check_forward_values(m):
    V = m.Value
    x = V(2.0)
    assert close((x * 3 + 1).data, 7.0), "2·3 + 1 should be 7"
    assert close((x ** 3).data, 8.0)
    assert close((1 / x).data, 0.5), "1/x is built from ** -1"
    assert close(x.exp().data, math.exp(2.0)), "exp forward"
    assert close(x.log().data, math.log(2.0)), "log forward (natural log)"
    assert close(V(0.5).tanh().data, math.tanh(0.5)), "tanh forward"
    assert V(-3.0).relu().data == 0.0 and V(3.0).relu().data == 3.0, "relu forward is max(0, x)"


def check_add_mul(m):
    V = m.Value
    a, b, c = V(2.0), V(-3.0), V(10.0)
    out = a * b + c
    out.backward()
    assert out.grad == 1.0, "backward() must seed the output's grad with 1.0"
    assert close(a.grad, -3.0), f"d(a·b + c)/da = b = -3, got {a.grad}"
    assert close(b.grad, 2.0), f"d(a·b + c)/db = a = 2, got {b.grad}"
    assert close(c.grad, 1.0), f"d(a·b + c)/dc = 1, got {c.grad}"


def check_accumulates(m):
    V = m.Value
    x = V(3.0)
    y = x * x + x  # x is used three times
    y.backward()
    assert close(x.grad, 7.0), (
        f"d(x² + x)/dx at 3 = 2·3 + 1 = 7, got {x.grad}. "
        "Gradient from every use must ADD up: use += in _backward, never ="
    )
    a = V(2.0)
    b = a + a
    b.backward()
    assert close(a.grad, 2.0), f"d(a + a)/da = 2, got {a.grad} — self and other are the same node here"


def check_pow_div(m):
    V = m.Value
    x, y = V(2.0), V(4.0)
    z = x ** 3 / y  # 8/4
    z.backward()
    assert close(z.data, 2.0)
    assert close(x.grad, 3 * 4 / 4), f"d(x³/y)/dx = 3x²/y = 3, got {x.grad}"
    assert close(y.grad, -8 / 16), f"d(x³/y)/dy = -x³/y² = -0.5, got {y.grad}"


def check_nonlinear(m):
    V = m.Value
    for name, f, df, x0 in [
        ("exp", lambda v: v.exp(), lambda x: math.exp(x), 0.7),
        ("log", lambda v: v.log(), lambda x: 1 / x, 1.7),
        ("tanh", lambda v: v.tanh(), lambda x: 1 - math.tanh(x) ** 2, 0.3),
        ("relu (x > 0)", lambda v: v.relu(), lambda x: 1.0, 1.5),
        ("relu (x < 0)", lambda v: v.relu(), lambda x: 0.0, -1.5),
    ]:
        x = V(x0)
        y = f(x) * 2.0  # a non-unit upstream gradient: your local derivative must multiply out.grad
        y.backward()
        assert close(x.grad, 2.0 * df(x0)), f"{name}: expected grad {2 * df(x0):.6f} at x={x0}, got {x.grad:.6f}"


def check_topo_order(m):
    V = m.Value
    x = V(1.5)
    a = x * 3          # a feeds both b and c
    b = a * a
    c = b + a
    d = c * a          # d = (9x² + 3x)·3x = 27x³ + 9x²
    d.backward()
    want = 81 * 1.5 ** 2 + 18 * 1.5
    assert close(x.grad, want), (
        f"expected d/dx = 81x² + 18x = {want}, got {x.grad}. A node ran _backward before every "
        "consumer had pushed its gradient — call _backward in reverse topological order"
    )


def check_deep_graph(m):
    V = m.Value
    x = V(1.0)
    y = x
    for _ in range(5000):
        y = y + x * 0.001
    y.backward()
    assert close(x.grad, 1 + 5000 * 0.001), f"expected 6.0, got {x.grad}"


def check_matches_numeric(m):
    V = m.Value
    rng = random.Random(0)
    w = [rng.uniform(-1, 1) for _ in range(9)]

    def net(ws, xs):
        h1 = (ws[0] * xs[0] + ws[1] * xs[1] + ws[2]).tanh()
        h2 = (ws[3] * xs[0] + ws[4] * xs[1] + ws[5]).relu()
        out = ws[6] * h1 + ws[7] * h2 + ws[8]
        return (out - 0.25) ** 2

    xs = [0.3, -1.2]
    params = [V(v) for v in w]
    loss = net(params, xs)
    loss.backward()

    eps = 1e-6
    for i in range(len(w)):
        hi = net([V(v + (eps if j == i else 0)) for j, v in enumerate(w)], xs).data
        lo = net([V(v - (eps if j == i else 0)) for j, v in enumerate(w)], xs).data
        numeric = (hi - lo) / (2 * eps)
        assert abs(params[i].grad - numeric) < 1e-5, (
            f"w[{i}]: backprop says {params[i].grad:.8f}, central difference says {numeric:.8f}"
        )


CHECKS = [
    check_forward_values,
    check_add_mul,
    check_accumulates,
    check_pow_div,
    check_nonlinear,
    check_topo_order,
    check_deep_graph,
    check_matches_numeric,
]
