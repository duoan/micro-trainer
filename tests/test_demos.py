"""Level-clearing ("通关") harness for every micro-trainer demo.

Each numbered demo ships with one or more "battle zones" that raise
``NotImplementedError`` until you hand-write the distributed operator. This test runs
every demo (and, where a demo exposes multiple levels, every level) end-to-end and
checks that it finishes without crashing:

    * level cleared  -> the demo runs to completion, exit code 0   -> PASSED
    * battle zone TODO still raises NotImplementedError            -> FAILED
    * any other crash (device mismatch, deadlock, bad shapes, ...) -> FAILED

A demo that defines a module-level ``LEVELS = [...]`` list (e.g. the data-parallel
demos: ``1_ddp.py`` has naive / overlap / bucketing) is run once PER level, as
``python <demo>.py <level>``, so each level is its own row on the dashboard. Demos
without ``LEVELS`` are run once with no argument.

So on a fresh clone every level is red; as you implement each battle zone the
corresponding test turns green. Run the whole dashboard with::

    make demos          # or: uv run pytest tests/test_demos.py -v
"""

from __future__ import annotations

import pathlib
import re
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

_LEVELS_RE = re.compile(r"^LEVELS\s*=\s*\[([^\]]*)\]", re.MULTILINE)


def _levels(path: pathlib.Path) -> list[str | None]:
    """Levels a demo exposes via a module-level ``LEVELS = [...]`` literal (else [None])."""
    match = _LEVELS_RE.search(path.read_text())
    if not match:
        return [None]
    found = re.findall(r"""["']([^"']+)["']""", match.group(1))
    return list(found) or [None]


def _case_id(path: pathlib.Path, level: str | None) -> str:
    base = f"{path.parent.name}/{path.name}"
    return f"{base}::{level}" if level else base


CASES = [(demo, level) for demo in DEMOS for level in _levels(demo)]
IDS = [_case_id(demo, level) for demo, level in CASES]


@pytest.mark.parametrize("demo,level", CASES, ids=IDS)
def test_demo_clears(demo: pathlib.Path, level: str | None) -> None:
    """A level is 'cleared' when its demo runs to completion (exit code 0)."""
    cmd = [sys.executable, str(demo)] + ([level] if level else [])
    proc = subprocess.run(
        cmd,
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
        pytest.fail(f"{_case_id(demo, level)}: {hint}\n\n{tail}", pytrace=False)
