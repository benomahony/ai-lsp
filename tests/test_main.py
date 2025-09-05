import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.result import StreamedRunResult


# Mock Agent before importing to avoid API key requirements
with patch("ai_lsp.main.Agent"):
    from ai_lsp.main import (
        AILanguageServer,
        CodeIssue,
        DiagnosticResult,
        Settings,
        SuggestedFix,
        setup_logging,
    )


@pytest.fixture
def server():
    """Create a test server instance."""
    with patch("ai_lsp.main.Agent"):
        return AILanguageServer()


@pytest.fixture
def mock_agent_result():
    """Mock agent result with test issues."""
    issues = [
        CodeIssue(
            issue_snippet="print",
            severity="warning",
            message="Consider using logging instead of print",
            suggested_fixes=[
                SuggestedFix(
                    title="Replace with logging",
                    target_snippet="print",
                    replacement_snippet="logging.info",
                )
            ],
        ),
        CodeIssue(
            issue_snippet="password",
            severity="error",
            message="Hardcoded password detected",
            suggested_fixes=None,
        ),
    ]
    result = MagicMock(spec=StreamedRunResult)
    result.output = DiagnosticResult(issues=issues)
    return result


class TestSettings:
    """Test settings configuration."""

    def test_default_settings(self):
        settings = Settings()
        assert settings.ai_lsp_model == "google-gla:gemini-2.5-pro"
        assert settings.debounce_ms == 1000

    def test_custom_settings(self):
        settings = Settings(
            ai_lsp_model="gpt-4o",
            debounce_ms=500,
        )
        assert settings.ai_lsp_model == "gpt-4o"
        assert settings.debounce_ms == 500


class TestSetupLogging:
    """Test logging configuration."""

    def test_setup_logging_default(self, tmp_path):
        log_file = tmp_path / "test.log"
        logger = setup_logging(log_file=log_file)

        assert logger.name == "ai-lsp"
        assert log_file.exists()

    def test_setup_logging_invalid_level(self, tmp_path):
        log_file = tmp_path / "invalid.log"
        logger = setup_logging("INVALID_LEVEL", log_file)

        # Should default to INFO when invalid level provided
        # Note: getattr(logging, "INVALID_LEVEL", logging.INFO) returns INFO
        assert logger.level <= logging.INFO


class TestFindSnippetInText:
    """Test snippet finding functionality."""

    def test_find_single_occurrence(self, server):
        text = "def hello():\n    print('Hello')\n    return"
        positions = server.find_snippet_in_text(text, "print")

        assert len(positions) == 1
        start_line, start_col, end_line, end_col = positions[0]
        assert start_line == 1
        assert start_col == 4
        assert end_line == 1
        assert end_col == 9

    def test_find_multiple_occurrences(self, server):
        text = "print('start')\nsome code\nprint('end')"
        positions = server.find_snippet_in_text(text, "print")

        assert len(positions) == 2

        # First occurrence
        start_line, start_col, end_line, end_col = positions[0]
        assert start_line == 0
        assert start_col == 0

        # Second occurrence
        start_line, start_col, end_line, end_col = positions[1]
        assert start_line == 2
        assert start_col == 0

    def test_find_multiline_snippet(self, server):
        text = "def func():\n    if True:\n        return"
        snippet = "if True:\n        return"
        positions = server.find_snippet_in_text(text, snippet)

        assert len(positions) == 1
        start_line, start_col, end_line, end_col = positions[0]
        assert start_line == 1
        assert start_col == 4
        assert end_line == 2
        assert end_col == 14

    def test_find_snippet_not_found(self, server):
        text = "def hello():\n    pass"
        positions = server.find_snippet_in_text(text, "nonexistent")

        assert len(positions) == 0

    def test_find_snippet_empty_text(self, server):
        positions = server.find_snippet_in_text("", "anything")
        assert len(positions) == 0

    def test_find_snippet_empty_snippet(self, server):
        text = "some text"
        positions = server.find_snippet_in_text(text, "")
        # Empty string should be found at position 0
        assert len(positions) > 0


class TestDiagnosticSeverity:
    """Test diagnostic severity mapping."""

    def test_get_diagnostic_severity_mappings(self, server):
        from lsprotocol.types import DiagnosticSeverity

        assert server._get_diagnostic_severity("error") == DiagnosticSeverity.Error
        assert server._get_diagnostic_severity("warning") == DiagnosticSeverity.Warning
        assert server._get_diagnostic_severity("info") == DiagnosticSeverity.Information
        assert server._get_diagnostic_severity("hint") == DiagnosticSeverity.Hint

    def test_get_diagnostic_severity_unknown(self, server):
        from lsprotocol.types import DiagnosticSeverity

        # Unknown severity should default to Information
        assert (
            server._get_diagnostic_severity("unknown") == DiagnosticSeverity.Information
        )


class TestLanguageDetection:
    """Test language detection from file extensions."""

    def test_detect_python(self, server):
        assert server._detect_language(Path("test.py")) == "python"

    def test_detect_javascript(self, server):
        assert server._detect_language(Path("test.js")) == "javascript"

    def test_detect_typescript(self, server):
        assert server._detect_language(Path("test.ts")) == "typescript"

    def test_detect_unknown_extension(self, server):
        assert server._detect_language(Path("test.unknown")) == "text"

    def test_detect_no_extension(self, server):
        assert server._detect_language(Path("README")) == "text"


class TestAnalyzeDocument:
    """Test document analysis functionality."""

    @pytest.mark.asyncio
    async def test_analyze_document_success(self, server, mock_agent_result):
        server.agent.run = AsyncMock(return_value=mock_agent_result)

        uri = "file:///test.py"
        text = "def func():\n    print('test')\n    password = 'secret'"

        diagnostics = await server.analyze_document(uri, text)

        assert len(diagnostics) == 2
        assert (
            diagnostics[0].message == "AI LSP: Consider using logging instead of print"
        )
        assert diagnostics[1].message == "AI LSP: Hardcoded password detected"

        # Check that issues are cached
        assert len(server.diagnostic_cache[uri]) == 2

    @pytest.mark.asyncio
    async def test_analyze_document_agent_failure(self, server):
        server.agent.run = AsyncMock(side_effect=Exception("AI service down"))

        uri = "file:///test.py"
        text = "def func(): pass"

        diagnostics = await server.analyze_document(uri, text)
        # Should return empty list on failure
        assert diagnostics == []
        assert uri not in server.diagnostic_cache

    @pytest.mark.asyncio
    async def test_analyze_document_snippet_not_found(self, server):
        # Create a mock result with snippet that won't be found
        issues = [
            CodeIssue(
                issue_snippet="nonexistent_code",
                severity="warning",
                message="This won't be found",
            )
        ]
        mock_result = MagicMock()
        mock_result.output = DiagnosticResult(issues=issues)
        server.agent.run = AsyncMock(return_value=mock_result)

        uri = "file:///test.py"
        text = "def func(): pass"

        diagnostics = await server.analyze_document(uri, text)

        # Should return empty diagnostics but still cache the issues
        assert diagnostics == []
        assert len(server.diagnostic_cache[uri]) == 1


class TestDebouncedAnalysis:
    """Test debounced analysis functionality."""

    @pytest.mark.asyncio
    async def test_debounced_analyze_cancels_previous(self, server):
        server.analyze_document = AsyncMock(return_value=[])
        server.publish_diagnostics = MagicMock()

        uri = "file:///test.py"
        text = "def func(): pass"

        # Start first analysis
        task1 = asyncio.create_task(server.debounced_analyze(uri, text))

        # Immediately start second analysis (should cancel first)
        task2 = asyncio.create_task(server.debounced_analyze(uri, text))

        # Wait for both to complete
        await asyncio.gather(task1, task2, return_exceptions=True)

        # Verify cleanup happened
        assert uri not in server._pending_tasks

    @pytest.mark.asyncio
    async def test_debounced_analyze_with_settings_delay(self, server):
        original_debounce = Settings().debounce_ms

        with patch("ai_lsp.main.settings") as mock_settings:
            mock_settings.debounce_ms = 10  # Very short delay for testing
            server.analyze_document = AsyncMock(return_value=[])
            server.publish_diagnostics = MagicMock()

            uri = "file:///test.py"
            text = "def func(): pass"

            await server.debounced_analyze(uri, text)

            # Verify analysis was called
            server.analyze_document.assert_called_once_with(uri, text)
            server.publish_diagnostics.assert_called_once()
