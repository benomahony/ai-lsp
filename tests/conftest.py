"""
Shared test fixtures and utilities for AI LSP tests.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Set up test environment before importing main module
os.environ.setdefault("GOOGLE_API_KEY", "test-api-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from ai_lsp.main import AILanguageServer, CodeIssue, DiagnosticResult, SuggestedFix
from pydantic_ai.result import StreamedRunResult


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def server_with_mock_agent():
    """Create server with mocked agent for testing."""
    with patch("ai_lsp.main.Agent") as mock_agent_class:
        server = AILanguageServer()

        # Create mock issues for testing
        mock_issues = [
            CodeIssue(
                issue_snippet="print",
                severity="warning",
                message="Consider using logging instead of print",
                suggested_fixes=[
                    SuggestedFix(
                        title="Replace with logging.info",
                        target_snippet="print",
                        replacement_snippet="logging.info",
                    )
                ],
            )
        ]

        mock_result = MagicMock(spec=StreamedRunResult)
        mock_result.output = DiagnosticResult(issues=mock_issues)
        server.agent.run = AsyncMock(return_value=mock_result)

        return server


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing."""
    return """
def authenticate(username, password):
    if password == 'admin123':  # Hardcoded password
        print(f'User {username} authenticated')
        return True
    else:
        print('Authentication failed')
        return False
"""


# Test utilities
def create_test_file(directory: Path, filename: str, content: str) -> Path:
    """Create a test file with given content."""
    file_path = directory / filename
    file_path.write_text(content)
    return file_path
