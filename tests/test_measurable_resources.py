"""Shift and LCU resource counts: closed forms, mirror vs kernel, lowering rules."""

from collections.abc import Iterator

import cudaq
import numpy as np
import pytest

from cudaq_pde import classical
from cudaq_pde.solvers import lcu
from cudaq_pde.solvers import measurable_resources as mr
from cudaq_pde.solvers.lcu_kernels import lcu_target
from cudaq_pde.solvers.spectral import shift_angles
from shift_test_kernels import apply_shift_variant

NS = range(2, 7)
VARIANTS = ["qft", "mcx", "ladder"]


def plan_params(n: int, negative: bool) -> tuple[float, float, float]:
    """(dt, c, nu) at half the explicit-Euler limit for n qubits.

    ``negative=False``: all coefficients >= 0; ``True``: a_dec = r - s < 0 (c h > 2 nu)
    while the step is still stable.
    """
    c, nu = (1.2, 0.05) if negative else (1.0, 1.0)  # c h < 2 nu up to h = pi/2
    return 0.5 * classical.fd_euler_dt_max(n, c, nu, classical.TWO_PI), c, nu


@pytest.fixture(autouse=True)
def cpu_target() -> Iterator[None]:
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("n", NS)
def test_shift_native_counts_match_closed_forms(n: int, variant: str) -> None:
    expected = mr.expected_shift_native(n, variant)
    assert mr.shift_native(n, variant, +1)["by_gate"] == expected
    assert mr.shift_native(n, variant, -1)["by_gate"] == expected  # inverse: same gates


@pytest.mark.parametrize("select", ["qft", "ladder"])
@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("negative", [False, True])
def test_lcu_native_counts_match_closed_forms(
    negative: bool, n: int, select: str
) -> None:
    dt, c, nu = plan_params(n, negative)
    plan = lcu.lcu_plan(n, dt, c, nu, classical.TWO_PI)
    assert (sum(plan.neg) == 1) == negative
    got = mr.lcu_native(n, dt, c, nu, select)["by_gate"]
    assert got == mr.expected_lcu_native(n, select, sum(plan.neg))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("n", NS)
def test_mirror_has_the_same_native_counts_as_the_kernel(n: int, variant: str) -> None:
    mirror = cudaq.estimate_resources(mr.build_shift_mirror(n, variant))
    assert dict(mirror.to_dict()) == mr.expected_shift_native(n, variant)


@pytest.mark.parametrize("select", ["qft", "ladder"])
@pytest.mark.parametrize("n", NS)
def test_lcu_mirror_has_the_same_native_counts(n: int, select: str) -> None:
    mirror = cudaq.estimate_resources(
        mr.build_lcu_mirror(n, *plan_params(n, False), select)
    )
    plan = lcu.lcu_plan(n, *plan_params(n, False), classical.TWO_PI)
    assert dict(mirror.to_dict()) == mr.expected_lcu_native(n, select, sum(plan.neg))


@pytest.mark.parametrize("n", NS)
def test_lowered_cx_follow_cudaq_rules(n: int) -> None:
    """cr1 = cry = 2 CX, ccx = 6 CX, cx = 1 CX in CUDA-Q's OpenQASM lowering."""
    assert mr.lowered(mr.build_shift_mirror(n, "qft"))["cx"] == 2 * n * (n - 1)
    ladder = mr.lowered(mr.build_shift_mirror(n, "ladder"))
    assert ladder["cx"] == 12 * max(n - 2, 0) + (n - 1)
    assert mr.lowered(mr.build_shift_mirror(n, "ladder"))["qubits"] == n + max(n - 2, 1)
    lcu_qft = mr.lowered(mr.build_lcu_mirror(n, *plan_params(n, False), "qft"))
    assert lcu_qft["cx"] == 2 * (n * n + n) + 4
    lcu_ladder = mr.lowered(mr.build_lcu_mirror(n, *plan_params(n, False), "ladder"))
    assert lcu_ladder["cx"] == 24 * (n - 1) + 2 * n + 4


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_mcx_lowering_uses_compiler_ancillas_from_n_4(n: int) -> None:
    """The cascade tops out at a Toffoli for n <= 3 and needs a C3X from n = 4."""
    low = mr.lowered(mr.build_shift_mirror(n, "mcx"))
    assert (low["qubits"] > n) == (n >= 4)
    assert mr.source_ancillas("mcx", n) == 0


# STATEVECTOR CHECKS (test only): tie the mirrors to the real kernels at n = 3.
@pytest.mark.parametrize("variant", VARIANTS)
def test_shift_mirror_state_equals_kernel_state(variant: str) -> None:
    n, bits = 3, [1, 1, 0]
    kind = mr.SHIFT_KINDS[(variant, +1)]
    m = mr._work(variant, n)
    kernel_state = np.array(
        cudaq.get_state(apply_shift_variant, n, m, bits, kind, shift_angles(n, +1))
    )
    mirror_state = np.array(
        cudaq.get_state(mr.build_shift_mirror(n, variant, with_input_bits=bits))
    )
    np.testing.assert_allclose(mirror_state[: 2**n], kernel_state[: 2**n], atol=1e-12)
    expected = np.zeros(2**n)
    expected[(3 + 1) % 8] = 1.0  # |011> (j = 3) -> j + 1
    np.testing.assert_allclose(np.abs(kernel_state[: 2**n]), expected, atol=1e-12)


@pytest.mark.parametrize("select", ["qft", "ladder"])
def test_lcu_mirror_state_equals_kernel_state(select: str) -> None:
    n = 3
    plan = lcu.lcu_plan(n, *plan_params(n, False), classical.TWO_PI)
    zero_theta = [0.0] * (n * 2)  # ansatz state |0..0>
    kernel_state = np.array(
        cudaq.get_state(
            lcu_target,
            n,
            zero_theta,
            1,
            plan.prep,
            plan.neg,
            lcu.SELECT_KINDS[select],
            lcu.n_work(select, n),
            plan.dec_angles,
            plan.inc_angles,
        )
    )
    mirror_state = np.array(
        cudaq.get_state(mr.build_lcu_mirror(n, *plan_params(n, False), select))
    )
    np.testing.assert_allclose(mirror_state, kernel_state, atol=1e-12)
    assert np.max(np.abs(kernel_state[1:])) > 1e-3  # non-trivial evolution
