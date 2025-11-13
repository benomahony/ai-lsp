import json
import subprocess
from pathlib import Path

from datasets import Dataset


def format_chat_messages(example: dict) -> str:
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]},
    ]
    return json.dumps(messages)


def prepare_dataset_for_mlx(dataset: Dataset, output_path: Path) -> None:
    formatted = []
    for example in dataset:
        formatted.append({"text": format_chat_messages(example)})
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for item in formatted:
            f.write(json.dumps(item) + "\n")


def train_model(
    dataset: Dataset,
    model_name: str = "mlx-community/gemma-2-2b-it-4bit",
    output_dir: Path | str = "./finetuned-model",
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 1e-5,
    max_seq_length: int = 8192,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / "train.jsonl"
    valid_file = output_dir / "valid.jsonl"
    prepare_dataset_for_mlx(dataset, train_file)
    prepare_dataset_for_mlx(dataset, valid_file)
    
    adapter_path = output_dir / "adapters"
    
    subprocess.run(
        [
            "mlx_lm.lora",
            "--model", model_name,
            "--train",
            "--data", str(output_dir),
            "--adapter-path", str(adapter_path),
            "--iters", str(len(dataset) * num_epochs // batch_size),
            "--batch-size", str(batch_size),
            "--learning-rate", str(learning_rate),
            "--num-layers", "16",
            "--save-every", "100",
            "--max-seq-length", str(max_seq_length),
        ],
        check=True,
    )
    
    fused_model_path = output_dir / "model"
    
    subprocess.run(
        [
            "mlx_lm.fuse",
            "--model", model_name,
            "--adapter-path", str(adapter_path),
            "--save-path", str(fused_model_path),
        ],
        check=True,
    )
    
    print(f"\nModel saved to {fused_model_path}")
    print(f"Adapters saved to {adapter_path}")
