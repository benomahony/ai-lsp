import typer

from ai_lsp.finetune import cli as finetune_cli


app = typer.Typer(
    name="ai-lsp",
    help="AI-powered Language Server",
    no_args_is_help=True,
    add_completion=True,
)


app.add_typer(finetune_cli.app, name="finetune")


@app.command()
def serve() -> None:
    """Start the AI-LSP server"""
    from ai_lsp.main import server

    server.start_io()


if __name__ == "__main__":
    app()
