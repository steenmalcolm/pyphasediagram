"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import os
import subprocess as sp
import sys
from pathlib import Path

import pytest

PACKAGE_PATH = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((PACKAGE_PATH / "examples").glob("*.py"))


@pytest.mark.parametrize("path", EXAMPLES)
def test_example(path):
    """Runs an example script given by path."""
    if path.name.startswith("_"):
        pytest.skip("skip examples starting with an underscore")

    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join([str(PACKAGE_PATH / "src"), env.get("PYTHONPATH", "")])
    env["MPLBACKEND"] = "agg"  # select a backend without a GUI
    proc = sp.Popen([sys.executable, path], env=env, stdout=sp.PIPE, stderr=sp.PIPE)
    try:
        outs, errs = proc.communicate(timeout=30)
    except sp.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()

    if proc.returncode > 0:
        if outs:
            print(f"\nSTDOUT:\n{outs}")
        if errs:
            print(f"\nSTDERR:\n{errs}")
        pytest.fail(f"Script `{path}` failed (output above)")
