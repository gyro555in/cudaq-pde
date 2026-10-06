#!/usr/bin/env python3
"""Test CUDA-Q on CPU target (qpp-cpu)."""

import cudaq

print("CUDA-Q version:", cudaq.__version__)

cudaq.set_target("qpp-cpu")
print("Target set to: qpp-cpu")

# Test with GHZ kernel
@cudaq.kernel
def ghz_kernel():
    q = cudaq.qvector(3)
    h(q[0])
    cx(q[0], q[1])
    cx(q[1], q[2])

# Sample
result = cudaq.sample(ghz_kernel, shots_count=100)
print("\nGHZ sampling result (100 shots):")
for bitstring, count in result.items():
    print(f"  {bitstring}: {count}")

# Get state
state = cudaq.get_state(ghz_kernel)
print("\nGHZ statevector:")
print(state)

print("\n=== CPU verification successful ===")
