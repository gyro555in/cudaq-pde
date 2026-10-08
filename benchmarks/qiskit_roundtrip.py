"""OpenQASM 2 round trip of the n = 3 circuits through Qiskit (the AQT / Piast-Q path).

AQT is not a CUDA-Q 0.16 target. This checks that the circuits CUDA-Q lowers to
OpenQASM 2 survive being loaded in Qiskit, which is the entry point of the AQT tooling
(qiskit-aqt is deliberately NOT installed). Needs the optional extra:
pip install -e ".[aqt]"

    taskset -c 0 python benchmarks/qiskit_roundtrip.py

For the spectral circuit, route A and route B2 (no measurements): statevector of the
loaded circuit against ``cudaq.get_state`` of the same circuit (up to a global phase),
and gate counts against CUDA-Q's lowering. For route B1: whether the mid-circuit
``measure`` and ``reset`` survive loading and a dump/load round trip, and keep their
position relative to the gates. The circuits are the builder mirrors of
``oneshot_resources`` (CUDA-Q 0.16 cannot translate kernels with arguments), lowered
with ``cudaq.translate(..., format="openqasm2")``. QASM files go to
results/qasm_export/.

Precision: CUDA-Q prints OpenQASM angles as ``%e`` (7 significant digits, e.g.
``1.570796e+00``), so the loaded circuit differs from the CUDA-Q state by about 1e-8 to
1e-7 in amplitude, not 1e-12. The pass threshold is TOL_STATE = 1e-6 for that reason.
"""

import json
import sys
from pathlib import Path

import cudaq
import numpy as np

from cudaq_pde.metadata import collect_metadata
from cudaq_pde.solvers import oneshot_resources as rr
from cudaq_pde.solvers import spectral_resources as sr

try:
    from qiskit import QuantumCircuit, qasm2, transpile
    from qiskit.quantum_info import Statevector
except ImportError:
    sys.exit('qiskit is not installed: pip install -e ".[aqt]"')

N, A = 3, 0.05
TOL_STATE = 1e-6  # OpenQASM angle printing, see the module docstring
OUT = Path("results/qasm_export")


def circuits() -> dict:
    return {
        "spectral": sr.build_mirror(N, "negative", False),
        "A": rr.build_mirror("A", N, A),
        "B2": rr.build_mirror("B2", N, A),
        "B1": rr.build_mirror("B1", N, A),
    }


def global_phase_distance(a: np.ndarray, b: np.ndarray) -> float:
    """``min_phi max |a - e^{i phi} b|`` with phi from the overlap."""
    phase = np.vdot(b, a)
    phase = phase / abs(phase) if abs(phase) > 0 else 1.0
    return float(np.max(np.abs(a - phase * b)))


def op_sequence(qc: QuantumCircuit) -> list[str]:
    return [inst.operation.name for inst in qc.data]


def main() -> int:
    cudaq.set_target("qpp-cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"metadata": collect_metadata(), "n": N, "a": A, "circuits": {}}
    ok = True
    for name, kernel in circuits().items():
        qasm = cudaq.translate(kernel, format="openqasm2")
        (OUT / f"{name}_n{N}.qasm").write_text(qasm)
        stats = sr.qasm_stats(qasm)
        qc = qasm2.loads(qasm)
        ops = qc.count_ops()
        row = {
            "cudaq_lowered": stats,
            "qiskit_loaded": {
                "qubits": qc.num_qubits,
                "total_ops": int(sum(ops.values())),
                "cx": int(ops.get("cx", 0)),
                "depth": qc.depth(),
                "ops": {k: int(v) for k, v in ops.items()},
            },
        }
        # statements of the CUDA-Q text vs instructions Qiskit loaded
        row["gate_count_equal"] = (
            row["qiskit_loaded"]["total_ops"] == stats["total_gates"]
        )
        row["cx_equal"] = row["qiskit_loaded"]["cx"] == stats["cx"]
        if name != "B1":
            # STATEVECTOR CHECK: Qiskit vs CUDA-Q (qubit 0 is the LSB in both)
            sv_q = Statevector.from_instruction(qc).data
            sv_c = np.array(cudaq.get_state(kernel), dtype=np.complex128)
            row["statevector_qubits_match"] = bool(sv_q.size == sv_c.size)
            if sv_q.size == sv_c.size:
                row["statevector_max_diff_up_to_global_phase"] = global_phase_distance(
                    sv_q, sv_c
                )
                row["fidelity"] = float(abs(np.vdot(sv_q, sv_c)) ** 2)
            tq = {
                lvl: transpile(
                    qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=lvl
                )
                for lvl in (0, 1)
            }
            row["qiskit_transpiled_cx_rz_sx_x"] = {
                f"opt{lvl}": {"cx": int(t.count_ops().get("cx", 0)), "depth": t.depth()}
                for lvl, t in tq.items()
            }
        else:
            seq = op_sequence(qc)
            back = qasm2.loads(qasm2.dumps(qc))
            seq_back = op_sequence(back)
            n_meas = rr.mid_circuit_ops("B1", N)
            # every measure is directly followed by a reset of the same qubit
            paired = all(
                seq[i] != "measure" or (i + 1 < len(seq) and seq[i + 1] == "reset")
                for i in range(len(seq))
            )
            row["mid_circuit"] = {
                "expected_measurements": n_meas,
                "measure_loaded": seq.count("measure"),
                "reset_loaded": seq.count("reset"),
                "each_measure_followed_by_reset": bool(paired),
                "classical_bits": qc.num_clbits,
                "dump_load_same_op_sequence": seq == seq_back,
                "measure_reset_after_dump_load": [
                    seq_back.count("measure"),
                    seq_back.count("reset"),
                ],
            }
            row["survives"] = bool(
                seq.count("measure") == n_meas
                and seq.count("reset") == n_meas
                and paired
                and seq == seq_back
            )
            ok &= row["survives"]
        if name != "B1":
            ok &= bool(
                row["gate_count_equal"]
                and row["cx_equal"]
                and row.get("statevector_max_diff_up_to_global_phase", 1.0) < TOL_STATE
            )
        report["circuits"][name] = row
        print(f"== {name}")
        print(json.dumps(row, indent=1, default=float))
    out = Path("benchmarks/output")
    out.mkdir(parents=True, exist_ok=True)
    (out / "qiskit_roundtrip.json").write_text(
        json.dumps(report, indent=1, default=float)
    )
    print("ALL OK" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
