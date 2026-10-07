"""End-to-end shot-based evolution vs explicit-Euler FD (qpp-cpu, reduced shot study).

The starting angles come from the CP4 simulator fit of the initial data: classical
preprocessing (loading data on hardware is exponential). The evolution itself only
samples. Reference quantities use the numpy ansatz simulation, never a simulator state.
"""

from collections.abc import Iterator
from functools import cache

import cudaq
import numpy as np
import pytest

from cudaq_pde import classical, encoding
from cudaq_pde.metrics import infidelity
from cudaq_pde.solvers import measurable
from cudaq_pde.solvers import variational as V
from cudaq_pde.solvers.ansatz_numpy import numpy_ansatz

pytestmark = pytest.mark.slow  # every test samples thousands of circuits

C_ADV, NU, DT, L = 1.0, 0.4, 0.05, classical.TWO_PI
N, DEPTH, STEPS = 3, 2, 3
ITERATIONS, LR = 60, 1.0


@pytest.fixture(autouse=True)
def cpu_target() -> Iterator[None]:
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


@cache
def start():
    x = classical.grid(N, L)
    u0 = 1.0 + 0.5 * np.sin(x) + 0.25 * np.cos(2 * x)
    ref = classical.evolve_fd_euler(u0, DT, STEPS, C_ADV, NU, L)
    psi0, norm0 = encoding.encode(u0)
    cudaq.set_target("qpp-cpu")
    fit = V.optimize_to_target(psi0.real, N, DEPTH, V.init_theta(N, DEPTH, 0))
    return ref, fit.theta, norm0 * abs(fit.overlap)


@cache
def run(shots: int, seed: int = 1):
    ref, theta0, norm0 = start()
    cudaq.set_target("qpp-cpu")
    res = measurable.evolve_measurable(
        theta0,
        norm0,
        DT,
        STEPS,
        C_ADV,
        NU,
        L,
        N,
        DEPTH,
        shots=shots,
        seed=seed,
        iterations=ITERATIONS,
        lr=LR,
    )
    ref_dir = ref / np.linalg.norm(ref, axis=1)[:, None]
    infid = np.array(
        [
            infidelity(numpy_ansatz(N, th, DEPTH), r)
            for th, r in zip(res.thetas, ref_dir, strict=True)
        ]
    )
    norm_err = np.abs(res.norms_abs - np.linalg.norm(ref, axis=1)) / np.linalg.norm(
        ref, axis=1
    )
    return res, infid, norm_err


def test_accounting_of_circuits_and_shots_per_step() -> None:
    shots = 300
    res, _, _ = run(shots)
    P = V.num_params(N, DEPTH)
    assert np.all(res.circuits_per_step[1:] == ITERATIONS * 2 * P + 1)
    assert np.all(res.shots_per_step[1:] == ITERATIONS * 2 * P * shots + shots)


# observed (5 steps, seed 1): 1-F max 5.4e-3 / 1.5e-3 / 2.3e-4 at 1e2 / 1e3 / 1e4 shots;
# thresholds below leave a factor ~4 to 10 of margin for seed-to-seed spread.
@pytest.mark.parametrize("shots, max_infid", [(300, 2e-2), (3000, 5e-3)])
def test_final_infidelity_is_bounded_by_the_shot_noise_level(shots, max_infid) -> None:
    _, infid, _ = run(shots)
    assert infid[0] < 1e-12  # the start (preprocessing fit)
    assert np.all(infid[1:] < max_infid), infid


def test_more_shots_give_a_better_state() -> None:
    lo = run(100)[1][1:].mean()
    hi = run(10000)[1][1:].mean()
    assert hi < lo / 5  # observed ratio about 30


def test_norm_tracking_error_is_small_and_improves_with_shots() -> None:
    lo = run(100)[2][1:]
    hi = run(10000)[2][1:]
    assert lo.max() < 5e-2 and hi.max() < 1e-2  # observed 1e-2 and 1e-3 (5 steps)
    assert hi.mean() < lo.mean()


def test_same_seed_is_deterministic() -> None:
    a = run(300, seed=3)[0]
    ref, theta0, norm0 = start()
    b = measurable.evolve_measurable(
        theta0,
        norm0,
        DT,
        STEPS,
        C_ADV,
        NU,
        L,
        N,
        DEPTH,
        shots=300,
        seed=3,
        iterations=ITERATIONS,
        lr=LR,
    )
    np.testing.assert_array_equal(a.thetas, b.thetas)
    np.testing.assert_array_equal(a.norms_abs, b.norms_abs)
    c = run(300, seed=4)[0]
    assert not np.allclose(a.thetas, c.thetas, atol=1e-6)


def test_stability_limit_is_asserted() -> None:
    _, theta0, norm0 = start()
    with pytest.raises(ValueError, match="unstable"):
        measurable.evolve_measurable(
            theta0, norm0, 10.0, 1, C_ADV, NU, L, N, DEPTH, shots=10, seed=0
        )
