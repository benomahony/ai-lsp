# AI LSP

An AI-powered Language Server that provides intelligent semantic code analysis beyond what traditional LSPs can detect. Built with modern Python tools and designed to integrate seamlessly with any LSP-compatible editor.

## What It Does

AI LSP focuses exclusively on **semantic issues that require deep understanding**:

- Logic errors in algorithms and business logic
- Subtle race conditions and concurrency issues
- Architectural problems and design pattern violations
- Security vulnerabilities requiring context understanding
- Performance anti-patterns needing semantic analysis
- Complex data flow issues
- Domain-specific best practice violations
- Accessibility issues requiring UX understanding

**What it doesn't do:** Traditional LSP tasks like syntax errors, type errors, undefined variables, or basic formatting - your regular LSP handles those perfectly.

## Features

- **🧠 AI-Powered Analysis**: Uses advanced language models for semantic code understanding
- **⚡ Debounced Analysis**: Smart delays prevent AI spam during rapid typing
- **🔧 Code Actions**: Provides automated fixes for detected issues
- **🎯 Multi-Language**: Supports Python, JavaScript, TypeScript, Rust, Go, Java, C/C++, and Lua
- **⚖️ Async Architecture**: Non-blocking analysis with race condition protection

## Installation

### From PyPI

```bash
uv add ai-lsp
# or
pip install ai-lsp
```

### From Source

```bash
git clone https://github.com/benomahony/ai-lsp
cd ai-lsp
uv sync
```

## Setup

### Environment Variables

Set your AI provider API key:

```bash
# For Google AI (default)
export GOOGLE_API_KEY="your-api-key"

# Or for OpenAI
export OPENAI_API_KEY="your-openai-key" 
```

### Configuration

Configure via environment variables or `.env` file:

```bash
AI_LSP_MODEL=google-gla:gemini-2.5-pro  # Default model
DEBOUNCE_MS=1000                         # Analysis delay (milliseconds)
```

Supported models:

- `google-gla:gemini-2.5-pro` (default)
- `gpt-4o`
- Any model supported by `pydantic-ai`

## Editor Integration

### Neovim (LazyVim)

Create `~/.config/nvim/lua/plugins/ai-lsp.lua`:

```lua
return {
  {
    "neovim/nvim-lspconfig",
    opts = {
      servers = {
        ["ai-lsp"] = {
          mason = false,
          cmd = { "ai-lsp" },
          filetypes = { "python", "javascript", "typescript", "rust", "go", "lua", "java", "cpp", "c" },
        },
      },
    },
  },
}
```

### VS Code

Add to your `settings.json`:

```json
{
  "ai-lsp.command": ["uvx", "ai-lsp"],
  "ai-lsp.filetypes": ["python", "javascript", "typescript", "rust", "go", "lua", "java", "cpp", "c"]
}
```

Or if you have it installed in your active environment:

```json
{
  "ai-lsp.command": ["ai-lsp"],
  "ai-lsp.filetypes": ["python", "javascript", "typescript", "rust", "go", "lua", "java", "cpp", "c"]
}
```

### Other Editors

Any LSP client can use AI LSP with these settings:

```json
{
  "command": ["uvx", "ai-lsp"],
  "filetypes": ["python", "javascript", "typescript", "rust", "go", "lua", "java", "cpp", "c"],
  "rootPatterns": [".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod"]
}
```

## Example Analysis

Given this Python code:

```python
def authenticate(username, password):
    if password == 'admin123':  # AI LSP will flag this
        print(f'User {username} authenticated')  # And suggest logging
        return True
    else:
        print('Authentication failed')
        return False
```

AI LSP might detect:

1. **Security Issue**: Hardcoded password `'admin123'`
   - Severity: Error
   - Suggestion: Use environment variables or secure credential storage

2. **Best Practice**: Using `print` statements
   - Severity: Warning  
   - Code Action: Replace with `logging.info()`

## Troubleshooting

### Common Issues

**LSP not starting**:

- Verify `ai-lsp` command is in PATH: `which ai-lsp`
- Check log file for errors (default: `ai-lsp.log`)
- Ensure API keys are set correctly

**No diagnostics appearing**:

- Check file type is supported  
- Verify AI model has API access
- Look for errors in log file

**Performance issues**:

- Increase `DEBOUNCE_MS` to reduce API calls
- Use a faster model like `gpt-4o-mini`
- Check network connectivity to AI provider

### Debug Mode

Enable verbose logging:

```bash
ai-lsp --log-level DEBUG --log-file debug.log
```

## License

MIT License - see [LICENSE](LICENSE) for details.
