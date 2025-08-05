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


# TODO: Add support for other LLM prompts
class Settings(BaseSettings):
    model: KnownModelName = Field(default="google-gla:gemini-2.5-pro")


settings = Settings()


class SuggestedFix(BaseModel):
    code: str
    line: int
    column: int
    end_line: int
    end_column: int
    title: str


class CodeIssue(BaseModel):
    line: int
    column: int
    end_line: int
    end_column: int
    severity: str  # "error", "warning", "info", "hint"
    message: str
    code: str | None = None
    suggested_fixes: list[SuggestedFix] | None = None


class DiagnosticResult(BaseModel):
    issues: list[CodeIssue]


class AILanguageServer(LanguageServer):
    def __init__(self):
        super().__init__("ai-lsp", "v0.1.0")
        self.logger = logging.getLogger("ai-lsp.server")
        self.logger.info("Initializing AI Language Server")
        self._diagnostic_cache: dict[str, list[CodeIssue]] = {}

        self.agent = Agent(
            model=settings.model,
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
- Exact line and column positions (0-based)
- Clear explanation of WHY this needs human attention
- Suggested architectural or design improvements
- Use severity: "info" for suggestions, "warning" for concerns, "error" for serious logic issues
- When possible, provide suggested_fixes with exact replacement code and a clear title

Focus on insights that require understanding code intent and context.""",
        )
        self.logger.info("AI agent initialized successfully")

    async def analyze_document(self, uri: str, text: str) -> list[Diagnostic]:
        self.logger.info(f"Starting AI analysis for {uri}")
        try:
            file_path = Path(uri.replace("file://", ""))
            language = self._detect_language(file_path)

            self.logger.debug(f"Detected language: {language} for {file_path.name}")

            prompt = f"""Analyze this {language} code for issues:

```{language}
{text}
```

File: {file_path.name}"""

            self.logger.debug("Sending request to AI agent")
            result = await self.agent.run(prompt)
            self.logger.info(
                f"AI analysis completed, found {len(result.output.issues)} issues"
            )

            # Cache the issues for code actions
            self._diagnostic_cache[uri] = result.output.issues

            diagnostics = []
            for issue in result.output.issues:
                severity_map = {
                    "error": DiagnosticSeverity.Error,
                    "warning": DiagnosticSeverity.Warning,
                    "info": DiagnosticSeverity.Information,
                    "hint": DiagnosticSeverity.Hint,
                }

                diagnostic = Diagnostic(
                    range=Range(
                        start=Position(line=issue.line, character=issue.column),
                        end=Position(line=issue.end_line, character=issue.end_column),
                    ),
                    message=f"AI LSP: {issue.message}",
                    severity=severity_map.get(
                        issue.severity, DiagnosticSeverity.Information
                    ),
                    source="ai-lsp",
                    code=issue.code,
                )
                diagnostics.append(diagnostic)

            self.logger.info(f"Generated {len(diagnostics)} diagnostics for {uri}")
            return diagnostics

        except Exception as e:
            self.logger.error(f"AI analysis failed for {uri}: {str(e)}", exc_info=True)
            return [
                Diagnostic(
                    range=Range(
                        start=Position(line=0, character=0),
                        end=Position(line=0, character=0),
                    ),
                    message=f"AI LSP unavailable: {str(e)}",
                    severity=DiagnosticSeverity.Information,
                    source="ai-lsp",
                )
            ]

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
    """Optionally analyze on change (debounced)"""
    server.logger.debug(f"Document changed: {params.text_document.uri}")
    # TODO: Could add debouncing here to avoid too frequent analysis
    pass


@server.feature(TEXT_DOCUMENT_CODE_ACTION)
async def code_action(params: CodeActionParams) -> list[CodeAction]:
    """Provide code actions for AI LSP diagnostics"""
    server.logger.info(f"Code action requested for {params.text_document.uri}")

    actions = []
    uri = params.text_document.uri

    # Get cached issues for this document
    cached_issues = server._diagnostic_cache.get(uri, [])

    # Find issues that overlap with the requested range
    for issue in cached_issues:
        if not issue.suggested_fixes:
            continue

        issue_range = Range(
            start=Position(line=issue.line, character=issue.column),
            end=Position(line=issue.end_line, character=issue.end_column),
        )

        # Check if the issue range overlaps with the requested range
        if _ranges_overlap(issue_range, params.range):
            for fix in issue.suggested_fixes:
                fix_range = Range(
                    start=Position(line=fix.line, character=fix.column),
                    end=Position(line=fix.end_line, character=fix.end_column),
                )

                edit = WorkspaceEdit(
                    changes={
                        uri: [
                            TextEdit(
                                range=fix_range,
                                new_text=fix.code,
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

    server.logger.info(f"Generated {len(actions)} code actions for {uri}")
    return actions


def _ranges_overlap(range1: Range, range2: Range) -> bool:
    """Check if two ranges overlap"""
    start1 = (range1.start.line, range1.start.character)
    end1 = (range1.end.line, range1.end.character)
    start2 = (range2.start.line, range2.start.character)
    end2 = (range2.end.line, range2.end.character)

    return start1 <= end2 and start2 <= end1


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
