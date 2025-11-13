from pathlib import Path

from mlx_lm import generate, load


def load_finetuned_model(
    model_path: Path | str = "./finetuned-model/model",
):
    model, tokenizer = load(str(model_path))
    return model, tokenizer


def analyze_code(
    model,
    tokenizer,
    code: str,
    system_prompt: str | None = None,
) -> str:
    if system_prompt is None:
        system_prompt = (
            "You are an AI code analyzer specialized in identifying "
            "semantic issues in code that traditional LSPs cannot detect."
        )
    
    prompt = f"""<start_of_turn>user
{system_prompt}

Analyze this Python code:

```python
{code}
```
<end_of_turn>
<start_of_turn>model
"""
    
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=512,
        temp=0.7,
    )
    
    return response


if __name__ == "__main__":
    print("Loading fine-tuned model...")
    model, tokenizer = load_finetuned_model()
    
    example_code = '''
def process_data(data):
    results = []
    for item in data:
        if item > 0:
            results.append(item * 2)
    return results
'''
    
    print("\nAnalyzing code...")
    analysis = analyze_code(model, tokenizer, example_code)
    print("\nAnalysis:")
    print(analysis)
