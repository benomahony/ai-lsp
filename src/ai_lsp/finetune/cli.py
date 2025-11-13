from pathlib import Path

import typer
from rich.console import Console

from ai_lsp.finetune.dataset import convert_otel_to_dataset
from ai_lsp.finetune.train import train_model

app = typer.Typer(
    name="finetune",
    help="Fine-tune an SLM on otel traces",
    no_args_is_help=True,
    add_completion=True,
)
console = Console()


@app.command()
def prepare_dataset(
    otel_file: Path = typer.Argument(..., help="Path to otel JSONL file"),
    output_file: Path = typer.Option(
        "dataset.json", help="Output path for processed dataset"
    ),
) -> None:
    console.print(f"[bold]Converting {otel_file} to training dataset...[/bold]")

    dataset = convert_otel_to_dataset(otel_file)

    console.print(f"[green]✓[/green] Extracted {len(dataset)} training examples")

    dataset.to_json(output_file)

    console.print(f"[green]✓[/green] Saved dataset to {output_file}")


@app.command()
def train(
    dataset_file: Path = typer.Argument(..., help="Path to dataset JSON file"),
    model_name: str = typer.Option(
        "mlx-community/gemma-2-2b-it-4bit", help="Base model to fine-tune"
    ),
    output_dir: Path = typer.Option(
        "./finetuned-model", help="Output directory for fine-tuned model"
    ),
    num_epochs: int = typer.Option(3, help="Number of training epochs"),
    batch_size: int = typer.Option(2, help="Training batch size"),
    learning_rate: float = typer.Option(1e-5, help="Learning rate"),
) -> None:
    from datasets import Dataset

    console.print(f"[bold]Loading dataset from {dataset_file}...[/bold]")
    dataset = Dataset.from_json(str(dataset_file))

    console.print(f"[green]✓[/green] Loaded {len(dataset)} training examples")
    console.print(f"[bold]Starting fine-tuning with {model_name}...[/bold]")

    train_model(
        dataset=dataset,
        model_name=model_name,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    console.print(f"[green]✓[/green] Fine-tuning complete! Model saved to {output_dir}")


@app.command()
def pipeline(
    otel_file: Path = typer.Argument(..., help="Path to otel JSONL file"),
    model_name: str = typer.Option(
        "mlx-community/gemma-2-2b-it-4bit", help="Base model to fine-tune"
    ),
    output_dir: Path = typer.Option(
        "./finetuned-model", help="Output directory for fine-tuned model"
    ),
    num_epochs: int = typer.Option(3, help="Number of training epochs"),
) -> None:
    console.print("[bold]Running complete fine-tuning pipeline...[/bold]")

    console.print("\n[bold blue]Step 1:[/bold blue] Converting otel traces to dataset")
    dataset = convert_otel_to_dataset(otel_file)
    console.print(f"[green]✓[/green] Extracted {len(dataset)} training examples")

    console.print(f"\n[bold blue]Step 2:[/bold blue] Fine-tuning {model_name}")
    train_model(
        dataset=dataset,
        model_name=model_name,
        output_dir=output_dir,
        num_epochs=num_epochs,
    )

    console.print(f"\n[green]✓[/green] Pipeline complete! Model saved to {output_dir}")
