"""Burgers heat-step resources: mirror vs kernel, measured vs constructed, fit flags."""

from collections.abc import Iterator

import cudaq
import numpy as np
import pytest

from burgers_test_kernels import STATE_KERNELS
from cudaq_pde import burgers as B
from cudaq_pde import classical
from cudaq_pde.solvers import burgers_quantum as bq
from cudaq_pde.solvers import burgers_resources as br
from cudaq_pde.solvers import oneshot
from cudaq_pde.solvers import oneshot_resources as rr

L = 2 * np.pi


@pytest.fixture(autouse=True)
def cpu_target() -> Iterator[None]:
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def phi0_for(n: int, nu: float = 0.5) -> np.ndarray:
    return B.discrete_burgers(np.sin(classical.grid(n)), nu, 1.0).phi0


@pytest.mark.parametrize("route", br.ROUTES)
@pytest.mark.parametrize("n", [3, 4, 5])
def test_measured_cx_equals_the_construction(route: str, n: int) -> None:
    row = br.route_row(route, phi0_for(n), 0.5, 1.0)
    assert row["cx_lowered"] == row["cx_constructed"]
    assert row["lowered_qubits"] == row["qubits"]  # no compiler ancillas
    assert row["loader_cx"] == 2**n - 2


@pytest.mark.parametrize("route", br.ROUTES)
@pytest.mark.parametrize("n", [3, 4])
def test_mirror_has_the_gates_of_the_kernel(route: str, n: int) -> None:
    nu, T = 0.5, 1.0
    phi0 = phi0_for(n, nu)
    call = bq.heat_call(phi0, nu, T, L, route)
    a = oneshot.diffusion_exponent(nu, T, L)
    mirror = rr.build_mirror(route, n, a, advection=False, loader=br.loader_for(phi0))
    got = dict(cudaq.estimate_resources(mirror).to_dict())
    want = dict(cudaq.estimate_resources(STATE_KERNELS[route], *call.args).to_dict())
    assert got == want


@pytest.mark.parametrize("route", br.ROUTES)
def test_mirror_has_the_state_of_the_kernel(route: str) -> None:
    n, nu, T = 3, 0.5, 1.0
    phi0 = phi0_for(n, nu)
    call = bq.heat_call(phi0, nu, T, L, route)
    a = oneshot.diffusion_exponent(nu, T, L)
    mirror = rr.build_mirror(route, n, a, advection=False, loader=br.loader_for(phi0))
    real = np.array(cudaq.get_state(STATE_KERNELS[route], *call.args))
    got = np.array(cudaq.get_state(mirror))
    np.testing.assert_allclose(got, real, atol=1e-12)


def test_default_mirror_is_unchanged_for_the_oneshot_circuits() -> None:
    """The CP7 mirror (advection on, no loader) keeps its gate set."""
    n, a = 3, 0.05
    full = dict(cudaq.estimate_resources(rr.build_mirror("A", n, a)).to_dict())
    heat = dict(
        cudaq.estimate_resources(rr.build_mirror("A", n, a, advection=False)).to_dict()
    )
    assert (
        full.get("r1", 0) == n and "r1" not in heat
    )  # only the advection phases differ
    assert {k: v for k, v in full.items() if k != "r1"} == heat


def test_fit_flags_follow_the_qubit_count() -> None:
    rows = {
        (route, n): br.total_qubits(route, n, q)
        for route, n, q in [("A", 5, 0), ("A", 7, 0), ("B2", 5, 0), ("B2", 6, 0)]
    }
    assert rows[("A", 5)] == 6 and rows[("A", 7)] == 8
    assert rows[("B2", 5)] == 20 and rows[("B2", 6)] == 27
    r = br.route_row("A", phi0_for(5), 0.5, 1.0)
    assert r["fits"] == {"20": True, "24": True}
    assert 27 > max(br.DEVICE_QUBITS)  # B2 at n = 6 fits neither device
