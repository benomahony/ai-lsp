import asyncio
import logging
from pathlib import Path
from typing import Any

from pydantic_ai.models import KnownModelName
import typer
from lsprotocol.types import (
    TEXT_DOCUMENT_CODE_ACTION,
    TEXT_DOCUMENT_DID_CHANGE,
    TEXT_DOCUMENT_DID_OPEN,
    TEXT_DOCUMENT_DID_SAVE,
    CodeAction,
    CodeActionKind,
    CodeActionParams,
    Diagnostic,
    DiagnosticSeverity,
    DidChangeTextDocumentParams,
    DidOpenTextDocumentParams,
    DidSaveTextDocumentParams,
    Position,
    Range,
    TextEdit,
    WorkspaceEdit,
)
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_settings import BaseSettings
from pygls.server import LanguageServer

app = typer.Typer()


def setup_logging(log_level: str = "INFO", log_file: Path | None = None):
    """Setup logging for the LSP server. Must log to file since stdio is used for LSP communication."""
    if log_file is None:
        log_file = Path("ai-lsp.log")

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
        ],
    )

    logger = logging.getLogger("ai-lsp")
    logger.info(f"AI LSP Server starting, logging to {log_file}")
    return logger


class Settings(BaseSettings):
    ai_lsp_model: KnownModelName = Field(default="google-gla:gemini-2.5-flash")
    debounce_ms: int = Field(default=1000)  # 1 second debounce


settings = Settings()


class SuggestedFix(BaseModel):
    title: str
    target_snippet: str
    replacement_snippet: str


class CodeIssue(BaseModel):
    issue_snippet: str
    severity: str  # "error", "warning", "info", "hint"
    message: str
    suggested_fixes: list[SuggestedFix] | None = None


class DiagnosticResult(BaseModel):
    issues: list[CodeIssue]


class AILanguageServer(LanguageServer):
    def __init__(self):
        super().__init__("ai-lsp", "v0.1.0")
        self.logger: logging.Logger = logging.getLogger("ai-lsp.server")
        self.logger.info("Initializing AI Language Server")
        self.diagnostic_cache: dict[str, list[CodeIssue]] = {}
        self._pending_tasks: dict[str, asyncio.Task[Any]] = {}
        self._analysis_locks: dict[str, asyncio.Lock] = {}

        self.agent: Agent[DiagnosticResult, Any] = Agent(
            model=settings.ai_lsp_model,
            output_type=DiagnosticResult,
            system_prompt="""You are an AI code analyzer that provides semantic insights that traditional LSPs cannot detect.

ONLY flag issues that require deep semantic understanding:
- Logic errors in algorithms or business logic
- Subtle race conditions or concurrency issues  
- Architectural problems and design pattern violations
- Security vulnerabilities requiring context understanding
- Performance anti-patterns that need semantic analysis
- Complex data flow issues
- Domain-specific best practice violations
- Accessibility issues requiring UX understanding

DO NOT flag issues that normal LSPs handle:
- Syntax errors
- Basic type errors  
- Undefined variables
- Import issues
- Basic formatting problems
- Simple linting rules

For each issue, provide:
- The exact issue_snippet that has the problem (ONLY the problematic token/value, e.g. just "python" not 'command = "python"')
- Clear explanation of WHY this needs human attention in the message
- Use severity: "info" for suggestions, "warning" for concerns, "error" for serious logic issues
- When possible, provide suggested_fixes with:
  - target_snippet: the exact code to replace (same as issue_snippet usually)
  - replacement_snippet: the replacement code
  - title: clear description of the fix

Focus on insights that require understanding code intent and context. Keep issue_snippet to the absolute minimum - just the problematic token.""",
        )
        self.logger.info("AI agent initialized successfully")

    def find_snippet_in_text(
        self, text: str, snippet: str
    ) -> list[tuple[int, int, int, int]]:
        """Find all occurrences of code snippet in text, return positions as (start_line, start_col, end_line, end_col)"""
        snippet = snippet.strip()
        positions = []

        # Search for all occurrences of snippet
        start_pos = 0
        while True:
            start_pos = text.find(snippet, start_pos)
            if start_pos == -1:
                break

            # Convert character position to line/column
            lines = text[:start_pos].split("\n")
            start_line = len(lines) - 1
            start_col = len(lines[-1])

            # Calculate end position
            snippet_lines = snippet.split("\n")
            if len(snippet_lines) == 1:
                end_line = start_line
                end_col = start_col + len(snippet)
            else:
                end_line = start_line + len(snippet_lines) - 1
                end_col = len(snippet_lines[-1])

            positions.append((start_line, start_col, end_line, end_col))
            start_pos += 1  # Move past this occurrence

        if not positions:
            self.logger.warning(f"Could not find snippet '{snippet}' in text")
        else:
            self.logger.debug(
                f"Found {len(positions)} occurrences of '{snippet[:20]}...'"
            )

        return positions

    async def debounced_analyze(self, uri: str, text: str):
        """Debounced analysis that cancels previous calls"""
        # Cancel any existing task for this URI
        if uri in self._pending_tasks:
            self._pending_tasks[uri].cancel()
            _ = self._pending_tasks.pop(uri, None)

        # Create new task
        task: asyncio.Task[None] = asyncio.create_task(self._delayed_analyze(uri, text))
        self._pending_tasks[uri] = task

        try:
            await task
        except asyncio.CancelledError:
            self.logger.debug(f"Analysis cancelled for {uri}")
        finally:
            # Clean up completed task
            if uri in self._pending_tasks and self._pending_tasks[uri] == task:
                _ = self._pending_tasks.pop(uri)

    async def _delayed_analyze(self, uri: str, text: str):
        """Wait for debounce period then analyze"""
        await asyncio.sleep(settings.debounce_ms / 1000.0)
        diagnostics = await self.analyze_document(uri, text)
        self.publish_diagnostics(uri, diagnostics)
        self.logger.debug(f"Published {len(diagnostics)} diagnostics for {uri}")

    async def analyze_document(self, uri: str, text: str) -> list[Diagnostic]:
        """Analyze document with race condition protection"""
        # Ensure we have a lock for this URI
        lock = self._analysis_locks.setdefault(uri, asyncio.Lock())
        async with lock:
            self.logger.info(f"Starting AI analysis for {uri}")

            # Always run fresh AI analysis - removed buggy caching logic
            try:
                file_path = Path(uri.replace("file://", ""))
                language = self._detect_language(file_path)

                prompt = f"""Analyze this {language} code for issues:

```{language}
{text}
```

File: {file_path.name}"""

                result = await self.agent.run(prompt)
                self.logger.info(
                    f"AI analysis completed, found {len(result.output.issues)} issues"
                )

                # Cache the issues
                self.diagnostic_cache[uri] = result.output.issues

                diagnostics = []
                for issue in result.output.issues:
                    positions = self.find_snippet_in_text(text, issue.issue_snippet)

                    if positions:
                        start_line, start_col, end_line, end_col = positions[0]
                        diagnostic = Diagnostic(
                            range=Range(
                                start=Position(line=start_line, character=start_col),
                                end=Position(line=end_line, character=end_col),
                            ),
                            message=f"AI LSP: {issue.message}",
                            severity=self._get_diagnostic_severity(issue.severity),
                            source="ai-lsp",
                        )
                        diagnostics.append(diagnostic)

                return diagnostics

            except Exception as e:
                self.logger.error(
                    f"AI analysis failed for {uri}: {str(e)}", exc_info=True
                )
                return []

    def _get_diagnostic_severity(self, severity: str) -> DiagnosticSeverity:
        severity_map = {
            "error": DiagnosticSeverity.Error,
            "warning": DiagnosticSeverity.Warning,
            "info": DiagnosticSeverity.Information,
            "hint": DiagnosticSeverity.Hint,
        }
        return severity_map.get(severity, DiagnosticSeverity.Information)

    def _detect_language(self, file_path: Path) -> str:
        suffix_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "jsx",
            ".tsx": "tsx",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".lua": "lua",
        }
        return suffix_map.get(file_path.suffix, "text")


server = AILanguageServer()


@server.feature(TEXT_DOCUMENT_DID_OPEN)
async def did_open(params: DidOpenTextDocumentParams):
    """Analyze document when opened"""
    server.logger.info(f"Document opened: {params.text_document.uri}")
    doc = params.text_document
    await server.debounced_analyze(doc.uri, doc.text)


@server.feature(TEXT_DOCUMENT_DID_SAVE)
async def did_save(params: DidSaveTextDocumentParams):
    """Analyze document when saved"""
    server.logger.info(f"Document saved: {params.text_document.uri}")
    doc = server.workspace.get_text_document(params.text_document.uri)
    await server.debounced_analyze(doc.uri, doc.source)


@server.feature(TEXT_DOCUMENT_DID_CHANGE)
async def did_change(params: DidChangeTextDocumentParams):
    """Debounced analysis on change"""
    server.logger.debug(f"Document changed: {params.text_document.uri}")
    doc = server.workspace.get_text_document(params.text_document.uri)
    # Use debounced analysis to avoid spamming AI
    await server.debounced_analyze(doc.uri, doc.source)


@server.feature(TEXT_DOCUMENT_CODE_ACTION)
async def code_action(params: CodeActionParams) -> list[CodeAction]:
    """Provide code actions for AI LSP diagnostics"""
    actions: list[CodeAction] = []
    uri = params.text_document.uri
    doc = server.workspace.get_text_document(uri)
    cached_issues = server.diagnostic_cache.get(uri, [])

    # Get the request range for filtering
    request_range = params.range

    for issue in cached_issues:
        if not issue.suggested_fixes:
            continue

        # Check if issue overlaps with the request range
        issue_positions = server.find_snippet_in_text(doc.source, issue.issue_snippet)
        if not issue_positions:
            continue

        start_line, start_col, end_line, end_col = issue_positions[0]
        issue_range = Range(
            start=Position(line=start_line, character=start_col),
            end=Position(line=end_line, character=end_col),
        )

        # Skip if issue doesn't overlap with request range
        if (
            issue_range.end.line < request_range.start.line
            or issue_range.start.line > request_range.end.line
            or (
                issue_range.end.line == request_range.start.line
                and issue_range.end.character < request_range.start.character
            )
            or (
                issue_range.start.line == request_range.end.line
                and issue_range.start.character > request_range.end.character
            )
        ):
            continue

        for fix in issue.suggested_fixes:
            target_positions = server.find_snippet_in_text(
                doc.source, fix.target_snippet
            )

            if target_positions:
                start_line, start_col, end_line, end_col = target_positions[0]
                fix_range = Range(
                    start=Position(line=start_line, character=start_col),
                    end=Position(line=end_line, character=end_col),
                )

                edit = WorkspaceEdit(
                    changes={
                        uri: [
                            TextEdit(range=fix_range, new_text=fix.replacement_snippet)
                        ]
                    }
                )

                action = CodeAction(
                    title=fix.title,
                    kind=CodeActionKind.QuickFix,
                    edit=edit,
                    is_preferred=True,
                )
                actions.append(action)

    return actions


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    tcp: bool = False,
    log_level: str = "INFO",
    log_file: str | None = None,
):
    """Start the AI LSP server"""
    log_path = Path(log_file) if log_file else None
    logger = setup_logging(log_level, log_path)

    logger.info(f"Starting AI LSP server (tcp={tcp}, host={host}, port={port})")

    if tcp:
        logger.info(f"Starting TCP server on {host}:{port}")
        server.start_tcp(host, port)
    else:
        logger.info("Starting stdio server")
        server.start_io()


if __name__ == "__main__":
    app()
