"""Quantum PDE solvers on CUDA-Q.

Importing this package does not import cudaq.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cudaq-pde")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
