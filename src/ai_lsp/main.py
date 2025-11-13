from ai_lsp.agent import create_diagnostic_agent
import asyncio
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

import logfire

# pyrefly: ignore [missing-import]
from lsprotocol.types import (
    TEXT_DOCUMENT_CODE_ACTION,
    TEXT_DOCUMENT_DID_CHANGE,
    TEXT_DOCUMENT_DID_OPEN,
    TEXT_DOCUMENT_DID_SAVE,
    CodeAction,
    CodeActionKind,
    CodeActionOptions,
    CodeActionParams,
    Command,
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
from pydantic_ai import Agent

# pyrefly: ignore [missing-import]
from pygls.server import LanguageServer

from ai_lsp.models import DiagnosticResult, Settings

settings = Settings()


os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = settings.otel_exporter_otlp_endpoint

if settings.configure_logfire:
    logfire.configure(send_to_logfire=False, service_name="ai-lsp")  # pyright: ignore[reportUnusedCallResult]
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)


class AILanguageServer(LanguageServer):
    def __init__(self):
        super().__init__("ai-lsp", version("ai-lsp"))
        logfire.info("Initializing AI Language Server")
        self._pending_tasks: dict[str, asyncio.Task[Any]] = {}
        self._analysis_locks: dict[str, asyncio.Lock] = {}
        self.agent: Agent[DiagnosticResult, Any] = create_diagnostic_agent(
            model=settings.ai_lsp_model
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
        # pyrefly: ignore [missing-attribute]
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
                                data={
                                    "issue_snippet": issue.issue_snippet,
                                    "suggested_fixes": [
                                        {
                                            "title": fix.title,
                                            "target_snippet": fix.target_snippet,
                                            "replacement_snippet": fix.replacement_snippet,
                                        }
                                        for fix in (issue.suggested_fixes or [])
                                    ],
                                },
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


@server.feature(
    TEXT_DOCUMENT_CODE_ACTION,
    CodeActionOptions(code_action_kinds=[CodeActionKind.QuickFix]),
)
async def code_actions(params: CodeActionParams) -> list[CodeAction]:
    with logfire.span("code_actions", uri=params.text_document.uri):
        actions: list[CodeAction] = []
        uri = params.text_document.uri
        doc = server.workspace.get_text_document(uri)

        for diagnostic in params.context.diagnostics:
            if diagnostic.source != "ai-lsp" or not diagnostic.data:
                continue

            suggested_fixes = diagnostic.data.get("suggested_fixes", [])

            for fix_data in suggested_fixes:
                target_positions = server.find_snippet_in_text(
                    doc.source, fix_data["target_snippet"]
                )

                if target_positions:
                    with logfire.span("creating_quick_fix", title=fix_data["title"]):
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
                                        new_text=fix_data["replacement_snippet"],
                                    )
                                ]
                            }
                        )

                        action = CodeAction(
                            title=f"Apply: {fix_data['title']}",
                            kind=CodeActionKind.QuickFix,
                            edit=edit,
                            diagnostics=[diagnostic],
                            is_preferred=True,
                        )
                        actions.append(action)

            actions.append(
                CodeAction(
                    title="Regenerate AI fix",
                    kind=CodeActionKind.QuickFix,
                    command=Command(
                        title="Regenerate fix",
                        command="ai-lsp.regenerateFix",
                        arguments=[
                            uri,
                            diagnostic.data.get("issue_snippet"),
                            diagnostic.range.start.line,
                            diagnostic.range.start.character,
                        ],
                    ),
                    diagnostics=[diagnostic],
                )
            )

            actions.append(
                CodeAction(
                    title="Dismiss AI suggestion",
                    kind=CodeActionKind.QuickFix,
                    command=Command(
                        title="Dismiss",
                        command="ai-lsp.dismiss",
                        arguments=[
                            uri,
                            diagnostic.range.start.line,
                            diagnostic.range.start.character,
                        ],
                    ),
                    diagnostics=[diagnostic],
                )
            )

        return actions


@server.command("ai-lsp.regenerateFix")
async def regenerate_fix(*args):
    if len(args) < 4:
        return

    uri, issue_snippet, line, character = args[0], args[1], args[2], args[3]

    with logfire.span("regenerate_fix", uri=uri, issue_snippet=issue_snippet):
        logfire.info(
            f"Regenerating fix for '{issue_snippet}' at {uri}:{line}:{character}"
        )

        doc = server.workspace.get_text_document(uri)
        file_path = Path(uri.replace("file://", ""))
        language = server._detect_language(file_path)

        prompt = f"""Analyze this specific code issue and provide NEW suggested fixes.

Issue snippet: {issue_snippet}

Full code context:
```{language}
{doc.source}
```

File: {file_path.name}

Focus ONLY on the issue at the snippet shown. Provide fresh, alternative fixes if possible."""

        try:
            result = await server.agent.run(prompt)

            if not result.output.issues:
                logfire.warn("No issues returned for regeneration")
                return

            regenerated_issue = result.output.issues[0]

            current_diags = getattr(server.lsp, "diagnostics", {}).get(uri, [])
            updated_diags = []

            for diag in current_diags:
                if (
                    diag.source == "ai-lsp"
                    and diag.range.start.line == line
                    and diag.range.start.character == character
                ):
                    positions = server.find_snippet_in_text(
                        doc.source, regenerated_issue.issue_snippet
                    )
                    if positions:
                        start_line, start_col, end_line, end_col = positions[0]
                        updated_diag = Diagnostic(
                            range=Range(
                                start=Position(line=start_line, character=start_col),
                                end=Position(line=end_line, character=end_col),
                            ),
                            message=f"AI LSP: {regenerated_issue.message}",
                            severity=server._get_diagnostic_severity(
                                regenerated_issue.severity
                            ),
                            source="ai-lsp",
                            data={
                                "issue_snippet": regenerated_issue.issue_snippet,
                                "suggested_fixes": [
                                    {
                                        "title": fix.title,
                                        "target_snippet": fix.target_snippet,
                                        "replacement_snippet": fix.replacement_snippet,
                                    }
                                    for fix in (regenerated_issue.suggested_fixes or [])
                                ],
                            },
                        )
                        updated_diags.append(updated_diag)
                        logfire.info(
                            f"Regenerated diagnostic with {len(regenerated_issue.suggested_fixes or [])} new fixes"
                        )
                else:
                    updated_diags.append(diag)

            server.publish_diagnostics(uri, updated_diags)

        except Exception as e:
            logfire.error(f"Failed to regenerate fix: {str(e)}", _exc_info=True)


@server.command("ai-lsp.dismiss")
def dismiss_suggestion(*args):
    if len(args) < 3:
        return

    uri, line, character = args[0], args[1], args[2]

    with logfire.span("dismiss_suggestion", uri=uri, line=line, character=character):
        logfire.info(f"Dismissing suggestion at {uri}:{line}:{character}")

        doc = server.workspace.get_text_document(uri)
        diagnostics = []

        current_diags = getattr(server.lsp, "diagnostics", {}).get(uri, [])
        for diag in current_diags:
            if (
                diag.source == "ai-lsp"
                and diag.range.start.line == line
                and diag.range.start.character == character
            ):
                continue
            diagnostics.append(diag)

        server.publish_diagnostics(uri, diagnostics)
