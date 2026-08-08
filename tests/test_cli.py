"""The CLI is a shipped console entry point, and nothing exercised it.

`README.md` documents four exit codes -- `0` ok, `1` usage, `2` artifact failed to
load, `130` interrupted -- and coverage over `cli.py` was 0%. A documented contract
that no test touches is a claim, not a guarantee.

These use click's own runner, so they cost milliseconds and need no subprocess.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

pytest.importorskip("click", reason="the cli extra is not installed")
from click.testing import CliRunner

if TYPE_CHECKING:
    import click

from mon_tokenizer.cli import main

MON = "ဂွံအခေါင်အရာမွဲသ္ဂောံဒုင်စသိုင်ကၠာကၠာရ။"


@pytest.fixture
def run():
    """Invoke the CLI in-process.

    `main` is defined twice in cli.py -- once as a plain function that explains
    how to install the `cli` extra, once as the click Command when the extra is
    present. mypy resolves the union to the plain form, so the cast names which
    branch these tests run against. The importorskip above guarantees it.
    """
    runner = CliRunner()
    command = cast("click.Command", main)
    return lambda *args: runner.invoke(command, list(args))


def test_tokenizing_text_exits_zero(run):
    result = run(MON)
    assert result.exit_code == 0, result.output
    assert result.output.strip()


def test_verbose_emits_a_per_token_table(run):
    plain = run(MON)
    verbose = run("-v", MON)
    assert verbose.exit_code == 0, verbose.output
    assert len(verbose.output) > len(plain.output), (
        "-v is documented as a per-token table; it produced no more output than the plain form"
    )


def test_decode_round_trips_through_ids(run):
    """Ids come from the library, not from scraping the rendered output.

    Parsing the table would test the formatting rather than the decode path, and
    the first attempt at this test did exactly that -- it tripped over the
    brackets rich draws around each id.
    """
    from mon_tokenizer import MonTokenizer

    tokenizer = MonTokenizer()
    ids = tokenizer.encode_ids(MON)
    result = run("-d", "--ids", ",".join(str(i) for i in ids))
    assert result.exit_code == 0, result.output
    assert tokenizer.decode_ids(ids)[:12] in result.output.replace("\n", "")


# --- the documented failure modes -----------------------------------------


def test_decode_without_a_source_is_a_usage_error(run):
    """Exit 1. Checked before the artifact loads, so a usage error does not pay
    for a 4.6 MB read."""
    result = run("-d")
    assert result.exit_code == 1
    assert "needs --tokens or --ids" in result.output


def test_tokens_and_ids_together_is_a_usage_error(run):
    result = run("-d", "--tokens", "a,b", "--ids", "1,2")
    assert result.exit_code == 1
    assert "not both" in result.output


def test_non_integer_ids_are_a_usage_error(run):
    result = run("-d", "--ids", "1,two,3")
    assert result.exit_code == 1
    assert "must be integers" in result.output


def test_a_missing_artifact_exits_two_not_one(run):
    """Exit 2 is 'artifact failed to load', and it must stay distinct from the
    usage code -- a caller scripting this needs to tell a typo from a broken
    install."""
    result = run("--model-path", "/nonexistent/tokenizer.json", MON)
    assert result.exit_code == 2, result.output


def test_usage_and_load_failures_do_not_share_an_exit_code(run):
    """Pins the distinction the README documents, rather than each code alone."""
    usage = run("-d").exit_code
    load = run("--model-path", "/nonexistent/tokenizer.json", MON).exit_code
    assert usage == 1 and load == 2 and usage != load
