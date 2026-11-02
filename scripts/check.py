"""One-command check: lint (ruff, if installed) and the test suite.

    python scripts/check.py          # lint + all tests
    python scripts/check.py --fast   # lint + tests without the slow ones (optimisations, simulations)
"""

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
py = sys.executable


def run(*cmd) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=root)


status = 0
quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
if subprocess.call([py, "-m", "ruff", "--version"], cwd=root, **quiet) == 0:
    status |= run(py, "-m", "ruff", "check", "src", "tests", "examples", "scripts")
else:
    print("ruff not installed: skipping lint (pip install -e '.[dev]')")
pytest_args = ["-q"] + (["-m", "not slow"] if "--fast" in sys.argv else [])
status |= run(py, "-m", "pytest", *pytest_args)
print("\nALL CHECKS PASSED" if status == 0 else "\nCHECKS FAILED")
sys.exit(1 if status else 0)
