from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from lsprotocol.types import (
    CodeActionParams,
    DidChangeTextDocumentParams,
    DidOpenTextDocumentParams,
    DidSaveTextDocumentParams,
    Position,
    Range,
    TextDocumentIdentifier,
    TextDocumentItem,
    VersionedTextDocumentIdentifier,
)

with patch("ai_lsp.main.Agent"):
    from ai_lsp.main import (
        CodeIssue,
        SuggestedFix,
        code_action,
        did_change,
        did_open,
        did_save,
        server,
    )


@pytest.fixture
def mock_server():
    """Mock server with workspace."""
    with patch.object(server, "workspace", MagicMock()):
        server.debounced_analyze = AsyncMock()
        server.diagnostic_cache = {}
        server.find_snippet_in_text = MagicMock(return_value=[(0, 0, 0, 5)])
        yield server


class TestLSPFeatures:
    """Test LSP feature handlers."""

    @pytest.mark.asyncio
    async def test_did_open(self, mock_server):
        """Test document open handler."""
        params = DidOpenTextDocumentParams(
            text_document=TextDocumentItem(
                uri="file:///test.py",
                language_id="python",
                version=1,
                text="print('hello')",
            )
        )

        await did_open(params)
        mock_server.debounced_analyze.assert_called_once_with(
            "file:///test.py", "print('hello')"
        )

    @pytest.mark.asyncio
    async def test_did_save(self, mock_server):
        """Test document save handler."""
        mock_doc = MagicMock()
        mock_doc.uri = "file:///test.py"
        mock_doc.source = "print('saved')"
        mock_server.workspace.get_text_document.return_value = mock_doc

        params = DidSaveTextDocumentParams(
            text_document=TextDocumentIdentifier(uri="file:///test.py")
        )

        await did_save(params)
        mock_server.debounced_analyze.assert_called_once_with(
            "file:///test.py", "print('saved')"
        )

    @pytest.mark.asyncio
    async def test_did_change(self, mock_server):
        """Test document change handler."""
        mock_doc = MagicMock()
        mock_doc.uri = "file:///test.py"
        mock_doc.source = "print('changed')"
        mock_server.workspace.get_text_document.return_value = mock_doc

        params = DidChangeTextDocumentParams(
            text_document=VersionedTextDocumentIdentifier(
                uri="file:///test.py", version=2
            ),
            content_changes=[],
        )

        await did_change(params)
        mock_server.debounced_analyze.assert_called_once_with(
            "file:///test.py", "print('changed')"
        )


class TestCodeAction:
    """Test code action functionality."""

    @pytest.mark.asyncio
    async def test_code_action_with_fixes(self, mock_server):
        """Test code action with suggested fixes."""
        uri = "file:///test.py"
        mock_doc = MagicMock()
        mock_doc.source = "print('test')"
        mock_server.workspace.get_text_document.return_value = mock_doc

        # Set up cached issues with fixes
        mock_server.diagnostic_cache[uri] = [
            CodeIssue(
                issue_snippet="print",
                severity="warning",
                message="Use logging",
                suggested_fixes=[
                    SuggestedFix(
                        title="Replace with logging",
                        target_snippet="print",
                        replacement_snippet="logging.info",
                    )
                ],
            )
        ]

        params = CodeActionParams(
            text_document=TextDocumentIdentifier(uri=uri),
            range=Range(
                start=Position(line=0, character=0), end=Position(line=0, character=5)
            ),
            context=MagicMock(),
        )

        actions = await code_action(params)
        assert len(actions) == 1
        assert actions[0].title == "Replace with logging"

    @pytest.mark.asyncio
    async def test_code_action_no_fixes(self, mock_server):
        """Test code action with no suggested fixes."""
        uri = "file:///test.py"
        mock_doc = MagicMock()
        mock_doc.source = "print('test')"
        mock_server.workspace.get_text_document.return_value = mock_doc

        # Issue without fixes
        mock_server.diagnostic_cache[uri] = [
            CodeIssue(
                issue_snippet="print",
                severity="warning",
                message="Use logging",
                suggested_fixes=None,
            )
        ]

        params = CodeActionParams(
            text_document=TextDocumentIdentifier(uri=uri),
            range=Range(
                start=Position(line=0, character=0), end=Position(line=0, character=5)
            ),
            context=MagicMock(),
        )

        actions = await code_action(params)
        assert len(actions) == 0

    @pytest.mark.asyncio
    async def test_code_action_no_cached_issues(self, mock_server):
        """Test code action with no cached issues."""
        uri = "file:///test.py"
        mock_doc = MagicMock()
        mock_doc.source = "print('test')"
        mock_server.workspace.get_text_document.return_value = mock_doc

        params = CodeActionParams(
            text_document=TextDocumentIdentifier(uri=uri),
            range=Range(
                start=Position(line=0, character=0), end=Position(line=0, character=5)
            ),
            context=MagicMock(),
        )

        actions = await code_action(params)
        assert len(actions) == 0
