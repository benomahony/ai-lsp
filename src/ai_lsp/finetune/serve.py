import subprocess
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel


def start_mlx_server(
    model_path: Path | str = "./finetuned-model/model",
    port: int = 8080,
) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "mlx_lm.server",
            "--model", str(model_path),
            "--port", str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def create_mlx_agent(
    model_path: str = "./finetuned-model/model",
    base_url: str = "http://localhost:8080/v1",
    system_prompt: str | None = None,
) -> Agent:
    mlx_model = OpenAIModel(
        model_name="mlx-local",
        base_url=base_url,
        api_key="dummy",
    )
    
    return Agent(
        model=mlx_model,
        system_prompt=system_prompt,
    )
