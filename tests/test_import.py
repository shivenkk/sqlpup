import subprocess
import sys

import sqlpup


def test_version() -> None:
    assert sqlpup.__version__ == "0.1.0"


def test_core_package_imports_without_torch() -> None:
    # torch is an optional 'train' extra; a bare `import sqlpup` (and anything it
    # pulls in) must not require it. Run in a fresh interpreter so torch already
    # imported by the rest of the test session cannot mask a regression.
    code = (
        "import sys, sqlpup; "
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] == 'torch'); "
        "assert not leaked, leaked"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
