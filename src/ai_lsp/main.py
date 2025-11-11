import asyncio
import os
from pathlib import Path
from typing import Literal
from importlib.metadata import version
import logfire
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
from pydantic_ai.models import KnownModelName
from pydantic_settings import BaseSettings
from pygls.server import LanguageServer


class Settings(BaseSettings):
    ai_lsp_model: KnownModelName = Field(default="google-gla:gemini-2.5-flash")
    debounce_ms: int = Field(default=1000)
    max_cache_size: int = Field(default=50)
    configure_logfire: bool = Field(default=True)
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:5173/api/v1/private/otel"
    )


settings = Settings()


os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = settings.otel_exporter_otlp_endpoint

if settings.configure_logfire:
    logfire.configure(send_to_logfire=False, service_name="ai-lsp")  # pyright: ignore[reportUnusedCallResult]
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)


class SuggestedFix(BaseModel):
    title: str
    target_snippet: str
    replacement_snippet: str


class CodeIssue(BaseModel):
    issue_snippet: str
    severity: Literal["error", "warning", "info", "hint"]
    message: str
    suggested_fixes: list[SuggestedFix] | None = None


class DiagnosticResult(BaseModel):
    issues: list[CodeIssue]


class AILanguageServer(LanguageServer):
    def __init__(self):
        super().__init__("ai-lsp", version("ai-lsp"))
        logfire.info("Initializing AI Language Server")
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
        logfire.info("AI agent initialized successfully")

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
            logfire.warn(f"Could not find snippet '{snippet}' in text")
        else:
            logfire.debug(f"Found {len(positions)} occurrences of '{snippet[:20]}...'")

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
            logfire.debug(f"Analysis cancelled for {uri}")
        finally:
            # Clean up completed task
            if uri in self._pending_tasks and self._pending_tasks[uri] == task:
                _ = self._pending_tasks.pop(uri)

    async def _delayed_analyze(self, uri: str, text: str):
        """Wait for debounce period then analyze"""
        await asyncio.sleep(settings.debounce_ms / 1000.0)
        diagnostics = await self.analyze_document(uri, text)
        self.publish_diagnostics(uri, diagnostics)
        logfire.debug(f"Published {len(diagnostics)} diagnostics for {uri}")

    async def analyze_document(self, uri: str, text: str) -> list[Diagnostic]:
        # Ensure we have a lock for this URI
        lock = self._analysis_locks.setdefault(uri, asyncio.Lock())
        async with lock:
            with logfire.span("analyze_document", uri=uri):
                logfire.info(f"Starting AI analysis for {uri}")

                try:
                    file_path = Path(uri.replace("file://", ""))
                    language = self._detect_language(file_path)

                    prompt = f"""Analyze this {language} code for issues:

```{language}
{text}
```

File: {file_path.name}"""

                    result = await self.agent.run(prompt)
                    logfire.info(
                        f"AI analysis completed, found {len(result.output.issues)} issues"
                    )

                    self.diagnostic_cache[uri] = result.output.issues

                    diagnostics = []
                    for issue in result.output.issues:
                        positions = self.find_snippet_in_text(text, issue.issue_snippet)

                        if positions:
                            start_line, start_col, end_line, end_col = positions[0]
                            diagnostic = Diagnostic(
                                range=Range(
                                    start=Position(
                                        line=start_line, character=start_col
                                    ),
                                    end=Position(line=end_line, character=end_col),
                                ),
                                message=f"AI LSP: {issue.message}",
                                severity=self._get_diagnostic_severity(issue.severity),
                                source="ai-lsp",
                            )
                            diagnostics.append(diagnostic)

                    return diagnostics

                except Exception as e:
                    logfire.error(
                        f"AI analysis failed for {uri}: {str(e)}", _exc_info=True
                    )
                    return []

    def _get_diagnostic_severity(self, severity: str) -> DiagnosticSeverity:
        match severity:
            case "error":
                return DiagnosticSeverity.Error
            case "warning":
                return DiagnosticSeverity.Warning
            case "info":
                return DiagnosticSeverity.Information
            case "hint":
                return DiagnosticSeverity.Hint
            case _:
                return DiagnosticSeverity.Information

    def _detect_language(self, file_path: Path) -> str:
        match file_path.suffix:
            case ".py":
                return "python"
            case ".js":
                return "javascript"
            case ".ts":
                return "typescript"
            case ".jsx":
                return "jsx"
            case ".tsx":
                return "tsx"
            case ".rs":
                return "rust"
            case ".go":
                return "go"
            case ".java":
                return "java"
            case ".cpp":
                return "cpp"
            case ".c":
                return "c"
            case ".lua":
                return "lua"
            case _:
                return "text"


server = AILanguageServer()


@server.feature(TEXT_DOCUMENT_DID_OPEN)
async def did_open(params: DidOpenTextDocumentParams):
    logfire.info(f"Document opened: {params.text_document.uri}")
    doc = params.text_document
    await server.debounced_analyze(doc.uri, doc.text)


@server.feature(TEXT_DOCUMENT_DID_SAVE)
async def did_save(params: DidSaveTextDocumentParams):
    logfire.info(f"Document saved: {params.text_document.uri}")
    doc = server.workspace.get_text_document(params.text_document.uri)
    await server.debounced_analyze(doc.uri, doc.source)


@server.feature(TEXT_DOCUMENT_DID_CHANGE)
async def did_change(params: DidChangeTextDocumentParams):
    logfire.debug(f"Document changed: {params.text_document.uri}")
    doc = server.workspace.get_text_document(params.text_document.uri)
    await server.debounced_analyze(doc.uri, doc.source)


@server.feature(TEXT_DOCUMENT_CODE_ACTION)
async def code_action(params: CodeActionParams) -> list[CodeAction]:
    with logfire.span("code_action", uri=params.text_document.uri):
        actions: list[CodeAction] = []
        uri = params.text_document.uri
        doc = server.workspace.get_text_document(uri)
        cached_issues = server.diagnostic_cache.get(uri, [])

        request_range = params.range

        for issue in cached_issues:
            if not issue.suggested_fixes:
                continue

            issue_positions = server.find_snippet_in_text(
                doc.source, issue.issue_snippet
            )
            if not issue_positions:
                continue

            start_line, start_col, end_line, end_col = issue_positions[0]
            issue_range = Range(
                start=Position(line=start_line, character=start_col),
                end=Position(line=end_line, character=end_col),
            )

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
                    with logfire.span("creating_quick_fix", title=fix.title):
                        start_line, start_col, end_line, end_col = target_positions[0]
                        fix_range = Range(
                            start=Position(line=start_line, character=start_col),
                            end=Position(line=end_line, character=end_col),
                        )

                        edit = WorkspaceEdit(
                            changes={
                                uri: [
                                    TextEdit(
                                        range=fix_range,
                                        new_text=fix.replacement_snippet,
                                    )
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


def app():
    server.start_io()


if __name__ == "__main__":
    app()
