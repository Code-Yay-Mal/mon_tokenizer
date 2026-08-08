"""Tests for the things that shipped broken to PyPI.

Three defects reached users because nothing checked packaging: `twine` was a
runtime dependency, `__version__` reported 0.1.5 on a 0.2.3 release, and the
model file shipped only by accident because all four declared data-inclusion
mechanisms were inert.

None of those are logic bugs, and no amount of testing `encode()` would have
found any of them. They need tests that read the manifest and the built
artifact.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
import zipfile
from importlib.metadata import version
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def manifest() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# Publishing tools, CLI framing and docs machinery. Each of these was, or could
# plausibly become, a runtime dependency of a library whose job is to tokenize a
# string.
_NOT_RUNTIME = {"twine", "click", "rich", "sphinx", "sphinx-rtd-theme", "build", "pytest"}


def test_no_publishing_or_cli_tooling_in_runtime_dependencies(manifest: dict):
    """`twine>=6.1.0` was an unconditional runtime dependency of 0.2.3.

    Confirmed in the published wheel's `Requires-Dist`. It dragged in keyring,
    cryptography, requests, docutils, readme-renderer and the jaraco.* tree —
    31 packages and 25MB — onto every machine that installed this to tokenize
    text. `click` and `rich` were there too, for a CLI most consumers never run.
    """
    declared = {_name(spec) for spec in manifest["project"]["dependencies"]}
    leaked = declared & _NOT_RUNTIME
    assert not leaked, (
        f"{sorted(leaked)} are runtime dependencies. Publishing tools belong in "
        f"[dependency-groups].dev; CLI dependencies belong in the [cli] extra."
    )


def test_runtime_dependencies_stay_minimal(manifest: dict):
    """One dependency. A ceiling on the count is what stops the next accident.

    Not a style preference: this package is a build-time dependency of mon-vlm
    and of any Mon tooling downstream, so anything added here is added to all of
    them.
    """
    declared = manifest["project"]["dependencies"]
    assert len(declared) <= 2, f"runtime dependencies grew to {declared}"

    # `tokenizers` pulls `huggingface-hub`, which brings httpx, fsspec and PyYAML:
    # the real install is 17 packages / ~30MB, heavier than 0.2.4's two. That is
    # the documented price of a self-describing artifact, and the README states
    # it rather than claiming "one dependency" — which would be true of this list
    # and misleading about what gets installed.


def test_the_runtime_dependency_is_tokenizers_with_no_upper_bound_needed(manifest: dict):
    """1.0.0 replaced sentencepiece, and the reason was a real failure.

    v1's `.model` raises `RuntimeError: INTERNAL: piece is too long` on
    sentencepiece 0.2.2, which validates piece length on load — so 0.2.4 had to
    pin `>=0.2.0,<0.2.2`, a window of exactly two releases. `tokenizer.json` is
    read by a pure-Rust loader with no such validation, so no ceiling is needed
    here. If one ever becomes necessary, that is a finding worth writing down
    rather than a bound to add quietly.
    """
    deps = {_name(d) for d in manifest["project"]["dependencies"]}
    assert deps == {"tokenizers", "regex"}, f"runtime dependencies are {sorted(deps)}"
    assert "sentencepiece" not in deps, "1.0.0 ships tokenizer.json, not a .model"


def test_there_is_exactly_one_dev_dependency_declaration(manifest: dict):
    """0.2.3 had two, and the documented command resolved the wrong one.

    `[project.optional-dependencies].dev` listed pytest, pytest-cov, black,
    isort, mypy and ruff. `[dependency-groups].dev` listed only pytest.
    `uv sync --dev` — the command in both the README and CONTRIBUTING — reads
    `[dependency-groups]`, so ruff, mypy, black and isort were never installed
    and the declared tooling was decorative.
    """
    extras = manifest["project"].get("optional-dependencies", {})
    assert "dev" not in extras, (
        "dev dependencies are declared in BOTH [project.optional-dependencies] "
        "and [dependency-groups]. Keep [dependency-groups] (PEP 735); a second "
        "declaration is one that will drift."
    )
    assert "dev" in manifest.get("dependency-groups", {})


def test_the_cli_extra_exists_and_carries_the_cli_dependencies(manifest: dict):
    cli = manifest["project"]["optional-dependencies"]["cli"]
    assert {_name(spec) for spec in cli} == {"click", "rich"}


# ---------------------------------------------------------------------------
# The artifact
# ---------------------------------------------------------------------------


def test_the_build_backend_configuration_uses_a_table_the_backend_reads(manifest: dict):
    """0.2.3 declared four data-inclusion mechanisms and all four were inert.

    `[tool.uv-build]` is not a table uv reads — the correct name is
    `[tool.uv.build-backend]`. Proof from the installed 0.2.3: its
    `dist-info/uv_build.json` was literally `{}`, meaning the backend saw no
    configuration at all. `[tool.setuptools.*]` and `MANIFEST.in` are setuptools
    mechanisms, and the backend here is `uv_build`.

    The 1MB model shipped regardless, because uv_build includes the module
    directory by default. That is shipping by accident.
    """
    tool = manifest.get("tool", {})
    assert "uv-build" not in tool, (
        "[tool.uv-build] is not read by uv_build; use [tool.uv.build-backend]"
    )
    assert "setuptools" not in tool, "setuptools tables are inert under the uv_build backend"
    assert "build-backend" in tool.get("uv", {})
    assert not (ROOT / "MANIFEST.in").exists(), "MANIFEST.in is a setuptools-only mechanism"


@pytest.mark.slow
def test_the_model_is_present_in_a_built_wheel(tmp_path):
    """Nothing asserted the model was in the artifact. This does.

    Without it, any future `exclude` rule or move of `data/` out of the package
    directory produces a wheel that installs cleanly and raises
    `FileNotFoundError` on first use — for everyone, on PyPI, silently.
    """
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"wheel build unavailable in this environment: {result.stderr[-300:]}")

    wheels = list(tmp_path.glob("*.whl"))
    assert wheels, "no wheel produced"
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        model = [n for n in names if n.endswith("mon_tokenizer.json")]
        assert model, f"the model is not in the wheel. Contents: {names}"
        assert archive.getinfo(model[0]).file_size > 4_000_000, (
            "the artifact in the wheel is truncated"
        )
        cards = [n for n in names if n.endswith("model_card.json")]
        assert cards, "the model card must ship beside the artifact"


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


def test_the_reported_version_matches_the_installed_distribution():
    """`__version__` was hardcoded "0.1.5" and stayed there through four releases.

    `mon-tokenizer --version` reported 0.1.5 on a 0.2.3 install. A version
    maintained by hand in a second place is a version that will be wrong.
    """
    import mon_tokenizer

    assert mon_tokenizer.__version__ == version("mon-tokenizer")


def test_the_manifest_version_matches_the_installed_distribution(manifest: dict):
    """Catches an edit to pyproject.toml that was never reinstalled or released."""
    assert manifest["project"]["version"] == version("mon-tokenizer")


def _name(spec: str) -> str:
    """Distribution name from a PEP 508 requirement string."""
    for separator in (">=", "<=", "==", "~=", "!=", ">", "<", "[", ";", " "):
        spec = spec.split(separator)[0]
    return spec.strip().lower()
