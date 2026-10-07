"""Statistics of the two measurable cost estimators (numpy only, no circuits).

Monte Carlo of the estimators from EXACT overlaps (numpy ansatz states) and binomial
draws; it models the sampling noise of the two circuits, it does not run them.

(a) Hadamard test, 3 overlap circuits ``V in {I, decrement, increment}``: each gives
    ``P(0) = (1 + o_V) / 2`` with ``o_V = <psi(theta)|V|psi(theta_prev)>``, estimated
    with ``N_a`` shots; ``O = a0 o_I + a_dec o_dec + a_inc o_inc`` and
    ``C_a = 1 - O^2 / ||b||^2``. This model takes ``||b||^2`` as known exactly (it
    needs 2 more circuits per step on hardware), which favors (a).
(b) LCU post-selection: ``N`` shots, ``N_succ ~ Binomial(N, p_succ)`` post-selected,
    ``N_0 ~ Binomial(N_succ, 1 - C)``, ``C_b = 1 - N_0 / N_succ``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cudaq_pde import classical
from cudaq_pde.solvers import lcu
from cudaq_pde.solvers.ansatz_numpy import numpy_ansatz


def exact_quantities(
    n: int,
    depth: int,
    plan: lcu.LCUPlan,
    theta_prev: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> dict:
    """Exact overlaps, cost and success probability for the two angle sets."""
    psi_prev, psi = numpy_ansatz(n, theta_prev, depth), numpy_ansatz(n, theta, depth)
    o = {
        "I": float(psi @ psi_prev),
        "dec": float(psi @ classical.decrement(psi_prev)),
        "inc": float(psi @ classical.increment(psi_prev)),
    }
    big_o = plan.a0 * o["I"] + plan.a_dec * o["dec"] + plan.a_inc * o["inc"]
    b = plan.a0 * psi_prev + plan.a_dec * classical.decrement(psi_prev)
    b = b + plan.a_inc * classical.increment(psi_prev)
    b_sq = float(b @ b)
    return {
        "o": o,
        "overlap": big_o,
        "b_sq": b_sq,
        "cost": 1.0 - big_o**2 / b_sq,
        "p_succ": b_sq / plan.lam**2,
    }


def analytic_std_b(cost: float, p_succ: float, shots: int) -> float:
    """``sqrt(C (1 - C) / (N p_succ))``: vanishes as ``C -> 0``."""
    return float(np.sqrt(cost * (1.0 - cost) / (shots * p_succ)))


def analytic_std_a(plan: lcu.LCUPlan, q: dict, shots_per_circuit: int) -> float:
    """Delta method: ``std(C_a) = 2 |O| std(O) / ||b||^2`` with
    ``Var(o_V) = (1 - o_V^2) / N`` per overlap circuit."""
    var_o = sum(
        (coef**2) * (1.0 - q["o"][v] ** 2) / shots_per_circuit
        for coef, v in ((plan.a0, "I"), (plan.a_dec, "dec"), (plan.a_inc, "inc"))
    )
    return float(2.0 * abs(q["overlap"]) / q["b_sq"] * np.sqrt(var_o))


def monte_carlo(
    plan: lcu.LCUPlan,
    q: dict,
    total_shots: int,
    repeats: int,
    seed: int,
) -> dict:
    """Mean and std of ``C_a`` and ``C_b`` at the same total shot budget."""
    rng = np.random.default_rng(seed)
    n_a = total_shots // 3
    p0 = {v: (1.0 + q["o"][v]) / 2.0 for v in q["o"]}
    o_hat = {
        v: 2.0 * rng.binomial(n_a, np.clip(p0[v], 0, 1), repeats) / n_a - 1.0
        for v in p0
    }
    big_o = plan.a0 * o_hat["I"] + plan.a_dec * o_hat["dec"] + plan.a_inc * o_hat["inc"]
    c_a = 1.0 - big_o**2 / q["b_sq"]

    n_succ = rng.binomial(total_shots, q["p_succ"], repeats)
    n_zero = rng.binomial(n_succ, np.clip(1.0 - q["cost"], 0, 1))
    c_b = 1.0 - n_zero / np.maximum(n_succ, 1)
    return {
        "exact_cost": q["cost"],
        "a_mean": float(c_a.mean()),
        "a_std": float(c_a.std(ddof=1)),
        "b_mean": float(c_b.mean()),
        "b_std": float(c_b.std(ddof=1)),
        "a_std_analytic": analytic_std_a(plan, q, n_a),
        "b_std_analytic": analytic_std_b(q["cost"], q["p_succ"], total_shots),
    }
