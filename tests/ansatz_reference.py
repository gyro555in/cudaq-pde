"""Test helper: independent numpy simulation of the RY/CX ansatz."""

import numpy as np


def apply_ry(psi: np.ndarray, n: int, qubit: int, theta: float) -> np.ndarray:
    """RY(theta) on ``qubit``; qubit 0 is the last axis (least significant bit)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    gate = np.array([[c, -s], [s, c]])
    axis = n - 1 - qubit
    t = psi.reshape([2] * n)
    return np.moveaxis(np.tensordot(gate, t, axes=([1], [axis])), 0, axis).reshape(-1)


def apply_cx(psi: np.ndarray, n: int, control: int, target: int) -> np.ndarray:
    t = psi.reshape([2] * n).copy()
    idx = [slice(None)] * n
    idx[n - 1 - control] = 1
    sub = t[tuple(idx)]
    # after fixing the control axis the target axis index shifts if it came later
    tgt_axis = n - 1 - target
    if tgt_axis > n - 1 - control:
        tgt_axis -= 1
    t[tuple(idx)] = np.flip(sub, axis=tgt_axis)
    return t.reshape(-1)


def numpy_ansatz(n: int, thetas: np.ndarray, depth: int) -> np.ndarray:
    psi = np.zeros(2**n)
    psi[0] = 1.0
    for i in range(n):
        psi = apply_ry(psi, n, i, thetas[i])
    for d in range(depth):
        for i in range(n - 1):
            psi = apply_cx(psi, n, i, i + 1)
        for i in range(n):
            psi = apply_ry(psi, n, i, thetas[(d + 1) * n + i])
    return psi
