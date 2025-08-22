import asyncio
import logging
from pathlib import Path

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
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
        ],
    )

    logger = logging.getLogger("ai-lsp")
    logger.info(f"AI LSP Server starting, logging to {log_file}")
    return logger


class Settings(BaseSettings):
    ai_lsp_model: KnownModelName = Field(default="google-gla:gemini-2.5-pro")
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
        self.logger = logging.getLogger("ai-lsp.server")
        self.logger.info("Initializing AI Language Server")
        self._diagnostic_cache: dict[str, list[CodeIssue]] = {}
        self._pending_tasks: dict[str, asyncio.Task] = {}

        self.agent = Agent(
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
- The exact code_snippet that has the problem (just the minimal problematic part, e.g. "kdap" not entire functions)
- Clear explanation of WHY this needs human attention in the message
- Use severity: "info" for suggestions, "warning" for concerns, "error" for serious logic issues
- When possible, provide suggested_fixes with:
  - target_snippet: the exact code to replace (same as code_snippet usually)
  - code: the replacement code
  - title: clear description of the fix

Focus on insights that require understanding code intent and context. Keep code_snippet minimal.""",
        )
        self.logger.info("AI agent initialized successfully")

    def _find_snippet_in_text(
        self, text: str, snippet: str
    ) -> list[tuple[int, int, int, int]]:
        """Find code snippet in text, return positions as (start_line, start_col, end_line, end_col)"""
        lines = text.split("\n")
        snippet = snippet.strip()

        # Look for the snippet as a substring in any line
        for i, line in enumerate(lines):
            if snippet in line:
                start_col = line.find(snippet)
                end_col = start_col + len(snippet)
                return [(i, start_col, i, end_col)]
        return []

    async def _debounced_analyze(self, uri: str, text: str):
        """Debounced analysis that cancels previous calls"""
        # Cancel any existing task for this URI
        if uri in self._pending_tasks:
            self._pending_tasks[uri].cancel()

        # Create new task
        task = asyncio.create_task(self._delayed_analyze(uri, text))
        self._pending_tasks[uri] = task

        try:
            await task
        except asyncio.CancelledError:
            self.logger.debug(f"Analysis cancelled for {uri}")
        finally:
            # Clean up completed task
            if uri in self._pending_tasks and self._pending_tasks[uri] == task:
                del self._pending_tasks[uri]

    async def _delayed_analyze(self, uri: str, text: str):
        """Wait for debounce period then analyze"""
        await asyncio.sleep(settings.debounce_ms / 1000.0)
        diagnostics = await self.analyze_document(uri, text)
        self.publish_diagnostics(uri, diagnostics)
        self.logger.debug(f"Published {len(diagnostics)} diagnostics for {uri}")

    async def analyze_document(self, uri: str, text: str) -> list[Diagnostic]:
        self.logger.info(f"Starting AI analysis for {uri}")

        # First, re-search existing issues
        diagnostics = []
        cached_issues = self._diagnostic_cache.get(uri, [])

        for issue in cached_issues:
            positions = self._find_snippet_in_text(text, issue.code_snippet)
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
                    code=issue.code,
                )
                diagnostics.append(diagnostic)

        # If we found all existing issues, return them
        if len(diagnostics) == len(cached_issues) and cached_issues:
            self.logger.info(f"Re-used {len(diagnostics)} existing issues for {uri}")
            return diagnostics

        # Otherwise run AI analysis
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
            self._diagnostic_cache[uri] = result.output.issues

            diagnostics = []
            for issue in result.output.issues:
                positions = self._find_snippet_in_text(text, issue.code_snippet)

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
                        code=issue.code,
                    )
                    diagnostics.append(diagnostic)

            return diagnostics

        except Exception as e:
            self.logger.error(f"AI analysis failed for {uri}: {str(e)}", exc_info=True)
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
    diagnostics = await server.analyze_document(doc.uri, doc.text)
    server.publish_diagnostics(doc.uri, diagnostics)
    server.logger.info(f"Published {len(diagnostics)} diagnostics for {doc.uri}")


@server.feature(TEXT_DOCUMENT_DID_SAVE)
async def did_save(params: DidSaveTextDocumentParams):
    """Analyze document when saved"""
    server.logger.info(f"Document saved: {params.text_document.uri}")
    doc = server.workspace.get_text_document(params.text_document.uri)
    diagnostics = await server.analyze_document(doc.uri, doc.source)
    server.publish_diagnostics(doc.uri, diagnostics)
    server.logger.info(f"Published {len(diagnostics)} diagnostics for {doc.uri}")


@server.feature(TEXT_DOCUMENT_DID_CHANGE)
async def did_change(params: DidChangeTextDocumentParams):
    """Debounced analysis on change"""
    server.logger.debug(f"Document changed: {params.text_document.uri}")
    doc = server.workspace.get_text_document(params.text_document.uri)
    # Use debounced analysis to avoid spamming AI
    asyncio.create_task(server._debounced_analyze(doc.uri, doc.source))


@server.feature(TEXT_DOCUMENT_CODE_ACTION)
async def code_action(params: CodeActionParams) -> list[CodeAction]:
    """Provide code actions for AI LSP diagnostics"""
    actions = []
    uri = params.text_document.uri
    doc = server.workspace.get_text_document(uri)
    cached_issues = server._diagnostic_cache.get(uri, [])

    for issue in cached_issues:
        if not issue.suggested_fixes:
            continue

        for fix in issue.suggested_fixes:
            target_positions = server._find_snippet_in_text(
                doc.source, fix.target_snippet
            )

            if target_positions:
                start_line, start_col, end_line, end_col = target_positions[0]
                fix_range = Range(
                    start=Position(line=start_line, character=start_col),
                    end=Position(line=end_line, character=end_col),
                )

                edit = WorkspaceEdit(
                    changes={uri: [TextEdit(range=fix_range, new_text=fix.code)]}
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
