import tempfile
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from ai_lsp.main import app, serve, setup_logging


class TestCLI:
    """Test the CLI interface."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_app_help(self, runner):
        """Test that help works correctly."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert (
            "AI LSP server" in result.output
            or "Start the AI LSP server" in result.output
        )

    def test_serve_help(self, runner):
        """Test that serve command help works."""
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "Start the AI LSP server" in result.output


class TestLoggingSetup:
    """Test logging setup functionality."""

    def test_setup_logging_creates_file(self):
        """Test that setup_logging creates log file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            logger = setup_logging(log_file=log_file)

            assert logger.name == "ai-lsp"
            assert log_file.exists()

    def test_setup_logging_default_file(self):
        """Test setup_logging with default file location."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use a real temporary file instead of mocking
            default_log = Path(temp_dir) / "ai-lsp.log"

            with patch("ai_lsp.main.Path") as mock_path:
                mock_path.return_value = default_log

                logger = setup_logging()

                assert logger.name == "ai-lsp"
                assert default_log.exists()


class TestErrorHandling:
    """Test CLI error handling."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_invalid_port_number(self, runner):
        """Test behavior with invalid port number."""
        result = runner.invoke(
            app,
            [
                "serve",
                "--tcp",
                "--port",
                "99999",  # Invalid port
            ],
        )

        # Typer should handle this validation or the command should fail
        # We don't test execution since it would try to start the server
        assert result.exit_code != 0 or "99999" in str(result.output)


if __name__ == "__main__":
    pytest.main([__file__])
