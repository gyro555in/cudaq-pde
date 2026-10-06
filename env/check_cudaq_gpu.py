#!/usr/bin/env python3
"""Test CUDA-Q on GPU target (nvidia fp64)."""

import cudaq
import numpy as np
import time

print("CUDA-Q version:", cudaq.__version__)

# GPU target with fp64 precision
cudaq.set_target("nvidia", option="fp64")
print("Target set to: nvidia (fp64)")

# Test 1: GHZ kernel
@cudaq.kernel
def ghz_kernel():
    q = cudaq.qvector(3)
    h(q[0])
    cx(q[0], q[1])
    cx(q[1], q[2])

print("\n=== Test 1: GHZ Kernel ===")
t0 = time.time()
state_gpu = cudaq.get_state(ghz_kernel)
t_gpu = time.time() - t0
print(f"GPU state (fp64) computed in {t_gpu:.4f}s")
print("GPU GHZ statevector:")
print(state_gpu)

# Compare to CPU
cudaq.set_target("qpp-cpu")
t0 = time.time()
state_cpu = cudaq.get_state(ghz_kernel)
t_cpu = time.time() - t0
print(f"CPU state (qpp-cpu) computed in {t_cpu:.4f}s")

# Compute difference
state_gpu_array = np.array(state_gpu)
state_cpu_array = np.array(state_cpu)
diff = np.max(np.abs(state_gpu_array - state_cpu_array))
print(f"\nMax difference (GPU vs CPU): {diff:.2e}")
if diff < 1e-12:
    print("✓ GPU and CPU statevectors agree to 1e-12")
else:
    print(f"⚠ Difference exceeds threshold (got {diff:.2e})")

# Test 2: 20-qubit random-like circuit on GPU
print("\n=== Test 2: 20-Qubit Circuit on GPU ===")
cudaq.set_target("nvidia", option="fp64")

@cudaq.kernel
def random_circuit(seed: int):
    q = cudaq.qvector(20)
    # Simple pattern (not fully random, but deterministic)
    for i in range(20):
        if (seed + i) % 3 == 0:
            h(q[i])
        elif (seed + i) % 3 == 1:
            x(q[i])
    for i in range(19):
        cx(q[i], q[i+1])

t0 = time.time()
state_20q = cudaq.get_state(random_circuit, 42)
t_20q = time.time() - t0
print(f"20-qubit circuit computed in {t_20q:.4f}s")
print(f"Statevector shape: {np.array(state_20q).shape}")
print(f"Norm: {np.linalg.norm(np.array(state_20q)):.10f}")

print("\n=== GPU Verification Complete ===")
