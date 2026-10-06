import subprocess
import sys

import cudaq_pde


def test_version_is_string() -> None:
    assert isinstance(cudaq_pde.__version__, str)
    assert cudaq_pde.__version__


def test_import_does_not_load_cudaq() -> None:
    code = (
        "import sys, cudaq_pde, cudaq_pde.metadata;"
        "sys.exit(1 if 'cudaq' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
