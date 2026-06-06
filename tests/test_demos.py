"""Level-clearing ("通关") harness for every micro-trainer demo.

Each numbered demo (e.g. ``1_data_parallel/1_ddp_demo.py``) ships with a
"battle zone" that raises ``NotImplementedError`` until you hand-write the
distributed operator. This test simply runs every demo end-to-end and checks
that it finishes without crashing:

    * level cleared  -> the demo runs to completion, exit code 0   -> PASSED
    * battle zone TODO still raises NotImplementedError            -> FAILED
    * any other crash (device mismatch, deadlock, bad shapes, ...) -> FAILED

So on a fresh clone every level is red; as you implement each battle zone the
corresponding test turns green. Run the whole dashboard with::

    make demos          # or: uv run pytest tests/test_demos.py -v
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Per-demo wall-clock budget. Demos spawn a few processes and run a handful of
# steps on CPU, so they finish in a couple of seconds; this is a generous cap
# that still catches deadlocks.
DEMO_TIMEOUT_S = 180

# Discover every demo: numbered files inside numbered module folders.
DEMOS = sorted(
    REPO_ROOT.glob("[0-9]_*/[0-9]_*.py"),
    key=lambda p: (p.parent.name, p.name),
)


def _demo_id(path: pathlib.Path) -> str:
    return f"{path.parent.name}/{path.name}"


@pytest.mark.parametrize("demo", DEMOS, ids=[_demo_id(p) for p in DEMOS])
def test_demo_clears(demo: pathlib.Path) -> None:
    """A level is 'cleared' when its demo runs to completion (exit code 0)."""
    proc = subprocess.run(
        [sys.executable, str(demo)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=DEMO_TIMEOUT_S,
    )

    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
        if "NotImplementedError" in proc.stderr:
            hint = "battle zone not implemented yet -- fill in the # TODO to clear this level"
        else:
            hint = "demo crashed -- see the traceback below"
        pytest.fail(f"{_demo_id(demo)}: {hint}\n\n{tail}", pytrace=False)
