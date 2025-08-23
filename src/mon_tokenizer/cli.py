"""cli for mon tokenizer"""

import sys
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from .tokenizer import MonTokenizer

console = Console()


@click.command()
@click.argument('text', required=False)
@click.option('--model-path', '-m', help='path to custom model file')
@click.option('--decode', '-d', is_flag=True, help='decode tokens instead of encoding')
@click.option('--tokens', '-t', help='tokens to decode (comma-separated)')
@click.option('--verbose', '-v', is_flag=True, help='show detailed output')
@click.option('--version', is_flag=True, help='show version and exit')
def main(text: Optional[str], model_path: Optional[str], decode: bool, 
         tokens: Optional[str], verbose: bool, version: bool):
    """mon tokenizer cli"""
    
    if version:
        from . import __version__
        console.print(f"mon tokenizer v{__version__}")
        return
    
    try:
        tokenizer = MonTokenizer(model_path)
        
        if decode:
            if not tokens:
                console.print("[red]error: --tokens required for decode mode[/red]")
                sys.exit(1)
            
            token_list = [t.strip() for t in tokens.split(',')]
            result = tokenizer.decode(token_list)
            console.print(f"decoded: {result}")
            return
        
        if not text:
            # interactive mode
            console.print("enter mon text to tokenize (ctrl+d to exit):")
            while True:
                try:
                    line = input("> ")
                    if line.strip():
                        process_text(tokenizer, line, verbose)
                except EOFError:
                    break
        else:
            process_text(tokenizer, text, verbose)
            
    except Exception as e:
        console.print(f"[red]error: {e}[/red]")
        sys.exit(1)


def process_text(tokenizer: MonTokenizer, text: str, verbose: bool):
    """process and show tokenization results"""
    result = tokenizer.encode(text)
    
    if verbose:
        # detailed table
        table = Table(title="tokenization results")
        table.add_column("index", style="cyan")
        table.add_column("token", style="green")
        table.add_column("id", style="yellow")
        
        for i, (piece, token_id) in enumerate(zip(result["pieces"], result["ids"])):
            table.add_row(str(i), piece, str(token_id))
        
        console.print(table)
        console.print(f"vocab size: {tokenizer.get_vocab_size()}")
    else:
        # simple output
        console.print(f"tokens: {result['pieces']}")
        console.print(f"ids: {result['ids']}")


if __name__ == "__main__":
    main()
