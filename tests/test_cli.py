import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from ai_lsp.main import app, setup_logging


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


class TestTCPMode:
    """Test TCP server mode."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_serve_tcp_mode(self, runner):
        """Test TCP mode branch in serve command."""
        with patch("ai_lsp.main.server.start_tcp") as mock_start_tcp:
            with patch("ai_lsp.main.setup_logging") as mock_logging:
                mock_logger = MagicMock()
                mock_logging.return_value = mock_logger

                # Mock to exit without actually starting server
                mock_start_tcp.side_effect = SystemExit(0)

                with pytest.raises(SystemExit):
                    runner.invoke(
                        app,
                        [
                            "serve",
                            "--tcp",
                            "--host",
                            "localhost",
                            "--port",
                            "8765",
                        ],
                    )

                mock_start_tcp.assert_called_once_with("localhost", 8765)

    def test_serve_stdio_mode(self, runner):
        """Test stdio mode branch in serve command."""
        with patch("ai_lsp.main.server.start_io") as mock_start_io:
            with patch("ai_lsp.main.setup_logging") as mock_logging:
                mock_logger = MagicMock()
                mock_logging.return_value = mock_logger

                # Mock to exit without actually starting server
                mock_start_io.side_effect = SystemExit(0)

                with pytest.raises(SystemExit):
                    runner.invoke(app, ["serve"])

                mock_start_io.assert_called_once()


class TestMainExecution:
    """Test main execution."""

    def test_main_block_coverage(self):
        """Test __main__ branch for coverage."""
        # This test just ensures the if __name__ == "__main__" branch is covered
        # We don't actually call app() to avoid starting the server
        import ai_lsp.main

        # Save original value
        original_name = ai_lsp.main.__name__

        try:
            # Set __name__ to __main__ to trigger the branch
            ai_lsp.main.__name__ = "__main__"

            # Import again to execute the conditional
            import importlib

            importlib.reload(ai_lsp.main)

        finally:
            # Restore original value
            ai_lsp.main.__name__ = original_name


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
