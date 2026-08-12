"""The release gate must be able to say no.

`preflight.py` is the last thing that runs before an irreversible publish — a
PyPI version can be yanked but never reused, and the Hugging Face repo is what
`AutoTokenizer` loads by default. Its entire value is that it prints
`DO NOT PUBLISH` when something is wrong.

It could not, in one case. The failure detail was built as

    out.strip().splitlines()[-1][:90]

and `"".splitlines()` is `[]`, so a gate command that exited non-zero while
printing nothing raised `IndexError` out of `main()`: traceback, no report, no
verdict, and an exit code that came from the crash rather than the count of
failures. A checker that crashes instead of refusing is a fail-open, and it fails
open at the one moment it exists for.

The test below drives the real `main()` with only the subprocess layer replaced,
so it fails on the actual control flow rather than on a helper called in
isolation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parent.parent


def _load_preflight() -> ModuleType:
    """Import `scripts/preflight.py` by path — it is a script, not a package module.

    A fresh module each call, so the module-level `results` list starts empty and
    two tests cannot see each other's checks.
    """
    spec = importlib.util.spec_from_file_location(
        "preflight_under_test", ROOT / "scripts" / "preflight.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_gate_command_that_fails_with_no_output_is_refused_not_crashed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """Every subprocess fails silently. The gate must report, not raise.

    Silent non-zero exits are ordinary: a subprocess killed by the OOM killer or
    by SIGKILL returns a non-zero code with nothing captured, and a tool that
    cannot read its own config can exit before writing a word. Before the fix
    this raised `IndexError: list index out of range` from line 76 and never
    reached the report at all.

    Nothing here shells out — `run` is the single place `preflight` touches a
    subprocess — so the whole test costs milliseconds and needs no wheel.
    """
    preflight = _load_preflight()
    monkeypatch.setattr(preflight, "run", lambda command, cwd=None: (1, "   \n\n  \n"))
    monkeypatch.setattr(sys, "argv", ["preflight.py"])

    failures = preflight.main()

    output = capsys.readouterr().out
    assert failures > 0, "every command failed and the gate reported no failures"
    assert "DO NOT PUBLISH" in output
    assert "Safe to publish" not in output
    # The exit code is documented as the number of failures, so it has to count
    # the same things the report lists.
    assert failures == sum(1 for ok, _ in preflight.results if not ok)
    # And the reader is told *something*, rather than a bare name with a dangling
    # em dash where the detail used to be.
    assert "exit 1, no output" in output
    for name in ("ruff format", "ruff check", "mypy", "pytest"):
        assert name in output


def test_the_detail_is_the_last_line_that_says_something():
    """Blank trailing lines are the normal shape of tool output.

    `pytest -q` ends with a newline; taking `splitlines()[-1]` before filtering
    would report an empty string on output that had a perfectly good error one
    line above it.
    """
    preflight = _load_preflight()
    output = "Found 3 errors in 2 files\nchecked 41 source files\n\n  \n"
    assert preflight.last_line(output, "unused") == "checked 41 source files"
    assert preflight.last_line("only one line", "unused") == "only one line"


def test_a_long_detail_is_truncated_and_a_missing_one_falls_back():
    """The report is a fixed-width table; an untruncated line would wrap it into
    noise. 90 characters is what the column was built for."""
    preflight = _load_preflight()
    assert preflight.last_line("x" * 200, "unused") == "x" * 90
    assert preflight.last_line("", "exit 2, no output") == "exit 2, no output"
    assert preflight.last_line("\n \t\n", "exit 2, no output") == "exit 2, no output"
