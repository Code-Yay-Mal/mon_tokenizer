"""Command-line interface for the Mon tokenizer.

`click` and `rich` are an optional extra (`pip install mon-tokenizer[cli]`), not
runtime dependencies. They were unconditional, so every library user who never
ran the command still installed both. Importing this module without them gives a
usable instruction instead of a `ModuleNotFoundError` traceback.

Exit codes are distinct so the command is scriptable:
  0 success · 1 usage error · 2 model could not be loaded · 130 interrupted
"""

from __future__ import annotations

import sys

try:
    import click
    from rich.console import Console
    from rich.table import Table

    _CLI_DEPS = True
except ImportError:  # pragma: no cover - exercised only in a bare install
    _CLI_DEPS = False


if not _CLI_DEPS:

    def main() -> None:  # type: ignore[misc]
        """Stand-in that explains what to install."""
        sys.stderr.write(
            "the mon-tokenizer command needs the 'cli' extra:\n"
            "    pip install 'mon-tokenizer[cli]'\n"
            "the library itself (from mon_tokenizer import MonTokenizer) needs nothing extra.\n"
        )
        raise SystemExit(1)

else:
    from .tokenizer import MonTokenizer

    console = Console()

    @click.command()
    @click.argument("text", required=False)
    @click.option("--model-path", "-m", help="Path to a custom tokenizer.json")
    @click.option("--decode", "-d", is_flag=True, help="Decode instead of encoding")
    @click.option(
        "--tokens",
        "-t",
        help="Comma-separated pieces to decode. A piece containing a comma cannot be "
        "expressed this way — use --ids for exactness.",
    )
    @click.option("--ids", help="Comma-separated token ids to decode. Unambiguous.")
    @click.option("--verbose", "-v", is_flag=True, help="Show a per-token table")
    @click.version_option(package_name="mon-tokenizer")
    def main(
        text: str | None,
        model_path: str | None,
        decode: bool,
        tokens: str | None,
        ids: str | None,
        verbose: bool,
    ) -> None:
        """Tokenize Mon, Burmese or English text."""
        # Validate before loading a 1MB model. The old order constructed the
        # tokenizer first, so a usage error still paid for the load.
        if decode and not tokens and not ids:
            console.print("[red]error: --decode needs --tokens or --ids[/red]")
            raise SystemExit(1)
        if ids and tokens:
            console.print("[red]error: pass --tokens or --ids, not both[/red]")
            raise SystemExit(1)

        try:
            tokenizer = MonTokenizer(model_path)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            console.print(f"[red]error: {exc}[/red]")
            raise SystemExit(2) from exc

        if decode:
            if ids:
                try:
                    parsed = [int(part) for part in ids.split(",")]
                except ValueError as exc:
                    console.print(f"[red]error: --ids must be integers: {exc}[/red]")
                    raise SystemExit(1) from exc
                console.print(tokenizer.decode_ids(parsed))
            else:
                # No .strip() on each piece: SentencePiece pieces carry a leading
                # U+2581 word marker, and stripping would silently mutate any
                # piece whose content is whitespace.
                console.print(tokenizer.decode((tokens or "").split(",")))
            return

        if text:
            _show(tokenizer, text, verbose)
            return

        console.print("enter text to tokenize (ctrl+d to exit):")
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            except KeyboardInterrupt:
                # Previously uncaught, so ctrl-C printed a traceback.
                console.print()
                raise SystemExit(130) from None
            if line.strip():
                _show(tokenizer, line, verbose)

    def _show(tokenizer: MonTokenizer, text: str, verbose: bool) -> None:
        result = tokenizer.encode(text)
        if not verbose:
            console.print(f"tokens: {result['pieces']}")
            console.print(f"ids: {result['ids']}")
            return

        table = Table(title="tokenization")
        table.add_column("index", style="cyan")
        table.add_column("piece", style="green")
        table.add_column("id", style="yellow")
        for index, (piece, token_id) in enumerate(
            zip(result["pieces"], result["ids"], strict=True)
        ):
            table.add_row(str(index), piece, str(token_id))
        console.print(table)
        console.print(f"vocab size: {tokenizer.get_vocab_size()}")
        if result["text"] != text:
            # Say so rather than leaving the user to wonder why the pieces do not
            # match the bytes they typed.
            console.print("[dim]input was normalized before encoding[/dim]")


if __name__ == "__main__":
    main()
