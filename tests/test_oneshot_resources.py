"""One-shot resource counts: closed forms, kernel vs builder mirror, CUDA-Q lowering."""

from collections.abc import Iterator

import cudaq
import numpy as np
import pytest

from cudaq_pde.solvers import oneshot
from cudaq_pde.solvers import oneshot_resources as rr

A = 0.013  # decay exponent, generic


@pytest.fixture(autouse=True)
def cpu_target() -> Iterator[None]:
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


@pytest.mark.parametrize("nyquist", ["negative", "zero"])
@pytest.mark.parametrize("route", ["A", "B1", "B2"])
@pytest.mark.parametrize("n", range(3, 9))
def test_native_counts_match_the_closed_forms(n: int, route: str, nyquist: str) -> None:
    got = rr.native_counts(route, n, A, nyquist)["by_gate"]
    assert got == rr.constructed_native(route, n, nyquist)


@pytest.mark.parametrize("nyquist", ["negative", "zero"])
@pytest.mark.parametrize("a", [0.002, 0.05, 0.4])
@pytest.mark.parametrize("n", range(3, 8))
def test_route_c_native_counts_match_the_closed_form(n: int, a: float, nyquist) -> None:
    nat = rr.native_counts("C", n, a, nyquist)
    assert nat["by_gate"] == rr.constructed_native("C", n, nyquist, nat["q"])


@pytest.mark.parametrize("n", range(3, 7))
def test_route_b_has_n_cry_and_n_choose_2_ccry_in_the_logical_count(n: int) -> None:
    fac = oneshot.contractions(n, A)
    assert fac.n_controls() == (n, n * (n - 1) // 2)
    assert rr.n_factors(n) == fac.count
    assert rr.mid_circuit_ops("B1", n) == fac.count and rr.mid_circuit_ops("A", n) == 0


@pytest.mark.parametrize("route", ["A", "B1", "B2"])
@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_lowered_cx_matches_the_construction(n: int, route: str) -> None:
    low = rr.lowered_counts(route, n, A, "negative")
    assert low["cx"] == rr.constructed_lowered_cx(route, n, "negative")


@pytest.mark.parametrize("a", [0.002, 0.05, 0.4])
@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_route_c_lowered_cx_matches_the_construction(n: int, a: float) -> None:
    q = oneshot.gaussian_plan(n, a).q
    low = rr.lowered_counts("C", n, a, "negative")
    assert low["cx"] == rr.constructed_lowered_cx("C", n, "negative", q)
    assert low["qubits"] == n + q  # no compiler ancillas


@pytest.mark.parametrize("n", [3, 4, 5])
def test_factor_decomposition_avoids_the_compilers_ancilla_and_14_cx(n: int) -> None:
    low = rr.lowered_counts("B2", n, A, "negative")
    assert low["qubits"] == n + rr.n_factors(n)  # CUDA-Q's own 2-control RY adds one
    pairs = n * (n - 1) // 2
    # 8 CX per doubly controlled rotation, 2 per singly controlled one, 2(n-1) folds
    assert low["cx"] - 2 * n * (n - 1) == 2 * n + 8 * pairs + 2 * (n - 1)


@pytest.mark.parametrize("route", ["A", "B1", "B2", "C"])
@pytest.mark.parametrize("n", [3, 4])
def test_mirror_native_counts_equal_the_kernels(n: int, route: str) -> None:
    kernel_route = "B2" if route == "B1" else route
    nat = rr.native_counts(kernel_route, n, 0.05)["by_gate"]
    mirror = rr.build_mirror(route, n, 0.05)
    got = dict(cudaq.estimate_resources(mirror).to_dict())
    if route == "B1":  # the measurements and resets are extra operations
        extra = rr.mid_circuit_ops("B1", n)
        assert got.pop("mz") == extra
        assert got.pop("reset") == extra
        # all factors target one ancilla, same gates
    assert got == nat


@pytest.mark.parametrize("route", ["A", "B2", "C"])
def test_mirror_state_equals_kernel_state(route: str) -> None:
    n, a = 3, 0.05
    kernel, args, _ = rr.kernel_args(route, n, a)
    real = np.array(cudaq.get_state(kernel, *args))
    mirror = np.array(cudaq.get_state(rr.build_mirror(route, n, a)))
    np.testing.assert_allclose(mirror, real, atol=1e-12)
    assert np.max(np.abs(real - np.eye(real.size)[0])) > 1e-3


def test_b1_lowering_has_one_ancilla_and_the_mid_circuit_operations() -> None:
    n = 4
    low = rr.lowered_counts("B1", n, A, "negative")
    b2 = rr.lowered_counts("B2", n, A, "negative")
    assert low["cx"] == b2["cx"]
    assert low["qubits"] == n + 1
    assert low["depth"] >= rr.n_factors(n)  # the factors share one ancilla


def test_expected_shots_formula() -> None:
    assert rr.expected_shots(1.0, 0.01) == pytest.approx(2500.0)
    assert rr.expected_shots(0.25, 0.01) == pytest.approx(10000.0)
