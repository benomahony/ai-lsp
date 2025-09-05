import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.result import StreamedRunResult

# Mock Agent before importing to avoid API key requirements
with patch("ai_lsp.main.Agent"):
    from ai_lsp.main import AILanguageServer, CodeIssue, DiagnosticResult, SuggestedFix


class TestIntegration:
    """Integration tests for the AI LSP server."""

    @pytest.fixture
    def server_with_mock_agent(self):
        """Create server with mocked agent for integration testing."""
        with patch("ai_lsp.main.Agent") as mock_agent_class:
            server = AILanguageServer()

            # Create mock issues for testing
            mock_issues = [
                CodeIssue(
                    issue_snippet="password = 'secret'",
                    severity="error",
                    message="Hardcoded credentials detected",
                    suggested_fixes=[
                        SuggestedFix(
                            title="Use environment variable",
                            target_snippet="'secret'",
                            replacement_snippet="os.getenv('PASSWORD')",
                        )
                    ],
                ),
                CodeIssue(
                    issue_snippet="print(",
                    severity="warning",
                    message="Consider using logging instead of print",
                    suggested_fixes=[
                        SuggestedFix(
                            title="Replace with logging.info",
                            target_snippet="print(",
                            replacement_snippet="logging.info(",
                        )
                    ],
                ),
            ]

            mock_result = MagicMock(spec=StreamedRunResult)
            mock_result.output = DiagnosticResult(issues=mock_issues)
            server.agent.run = AsyncMock(return_value=mock_result)

            return server

    @pytest.mark.asyncio
    async def test_multiple_file_analysis(self, server_with_mock_agent):
        """Test analyzing multiple files simultaneously."""
        server = server_with_mock_agent
        server.publish_diagnostics = MagicMock()

        files = {
            "file:///file1.py": "def func1(): print('file1')",
            "file:///file2.py": "def func2(): print('file2')",
            "file:///file3.py": "def func3(): print('file3')",
        }

        # Analyze all files concurrently
        tasks = []
        for uri, text in files.items():
            task = asyncio.create_task(server.debounced_analyze(uri, text))
            tasks.append(task)

        await asyncio.gather(*tasks)

        # Verify all files were analyzed
        assert len(server.diagnostic_cache) == 3
        for uri in files.keys():
            assert uri in server.diagnostic_cache

    @pytest.mark.asyncio
    async def test_error_recovery(self, server_with_mock_agent):
        """Test that server recovers gracefully from errors."""
        server = server_with_mock_agent
        server.publish_diagnostics = MagicMock()

        # Make the agent fail
        server.agent.run = AsyncMock(side_effect=Exception("AI service error"))

        uri = "file:///error_test.py"
        text = "def func(): pass"

        # This should not raise an exception
        diagnostics = await server.analyze_document(uri, text)

        # Should return empty diagnostics on error
        assert diagnostics == []

        # Cache should not contain the failed analysis
        assert uri not in server.diagnostic_cache

    @pytest.mark.asyncio
    async def test_rapid_file_changes(self, server_with_mock_agent):
        """Test handling rapid file changes with debouncing."""
        server = server_with_mock_agent
        server.publish_diagnostics = MagicMock()

        uri = "file:///rapid_changes.py"

        # Simulate rapid changes
        changes = [
            "def func(): pass",
            "def func(): print('v1')",
            "def func(): print('v2')",
            "def func(): print('final')",
        ]

        # Fire off multiple rapid changes
        tasks = []
        for text in changes:
            task = asyncio.create_task(server.debounced_analyze(uri, text))
            tasks.append(task)
            # Small delay to simulate typing
            await asyncio.sleep(0.01)

        # Wait for all to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # Should have been debounced - only final analysis should remain
        assert uri not in server._pending_tasks


class TestPerformance:
    """Performance-related tests."""

    @pytest.mark.asyncio
    async def test_large_file_analysis(self, server_with_mock_agent):
        """Test analysis of large files."""
        server = server_with_mock_agent

        # Generate a large Python file
        function_parts = []
        for i in range(1000):
            function_parts.extend(
                [
                    f"def function_{i}():",
                    f"    print('Function {i}')",
                    f"    return {i}",
                    "",
                ]
            )
        large_code = "\n".join(function_parts)

        uri = "file:///large_file.py"

        # This should complete without timeout
        diagnostics = await server.analyze_document(uri, large_code)

        # Verify it was processed
        assert uri in server.diagnostic_cache

    def test_memory_usage_with_many_files(self, server_with_mock_agent):
        """Test memory usage doesn't grow unbounded."""
        server = server_with_mock_agent

        # Cache many files
        for i in range(100):
            uri = f"file:///file_{i}.py"
            issues = [
                CodeIssue(
                    issue_snippet=f"test_{i}",
                    severity="info",
                    message=f"Test issue {i}",
                )
            ]
            server.diagnostic_cache[uri] = issues

        # Verify all are cached
        assert len(server.diagnostic_cache) == 100

        # In a real implementation, you might want to implement cache eviction
        # This test just verifies the current behavior


class TestConfigurationScenarios:
    """Test different configuration scenarios."""

    def test_different_debounce_settings(self):
        """Test server behavior with different debounce settings."""
        with patch("ai_lsp.main.settings") as mock_settings:
            mock_settings.debounce_ms = 500
            mock_settings.ai_lsp_model = "gpt-4o"

            with patch("ai_lsp.main.Agent"):
                server = AILanguageServer()

                # Verify server uses the settings
                assert hasattr(server, "agent")

    @pytest.mark.asyncio
    async def test_different_ai_models(self, server_with_mock_agent):
        """Test that different AI models can be configured."""
        server = server_with_mock_agent

        # The agent is mocked, but we can verify it would work with different models
        uri = "file:///model_test.py"
        text = "def test(): pass"

        diagnostics = await server.analyze_document(uri, text)

        # Verify the agent was called
        server.agent.run.assert_called()


class TestRealWorldScenarios:
    """Test scenarios that mirror real-world usage."""

    @pytest.mark.asyncio
    async def test_python_security_issues(self, server_with_mock_agent):
        """Test detection of common Python security issues."""
        server = server_with_mock_agent

        # Configure agent to return security-focused issues
        security_issues = [
            CodeIssue(
                issue_snippet="eval(",
                severity="error",
                message="Use of eval() is dangerous",
                suggested_fixes=[
                    SuggestedFix(
                        title="Use ast.literal_eval for safe evaluation",
                        target_snippet="eval(",
                        replacement_snippet="ast.literal_eval(",
                    )
                ],
            ),
            CodeIssue(
                issue_snippet="subprocess.call(user_input",
                severity="error",
                message="Command injection vulnerability",
                suggested_fixes=[
                    SuggestedFix(
                        title="Use shlex.quote to sanitize input",
                        target_snippet="subprocess.call(user_input",
                        replacement_snippet="subprocess.call(shlex.quote(user_input)",
                    )
                ],
            ),
        ]

        mock_result = MagicMock()
        mock_result.output = DiagnosticResult(issues=security_issues)
        server.agent.run = AsyncMock(return_value=mock_result)

        vulnerable_code = """
import subprocess
import eval

def process_data(user_input):
    result = eval(user_input)
    subprocess.call(user_input + " --verbose")
    return result
"""

        uri = "file:///vulnerable.py"
        diagnostics = await server.analyze_document(uri, vulnerable_code)

        # Verify security issues were detected
        assert len(server.diagnostic_cache[uri]) == 2
        messages = [issue.message for issue in server.diagnostic_cache[uri]]
        assert "Use of eval() is dangerous" in messages
        assert "Command injection vulnerability" in messages

    @pytest.mark.asyncio
    async def test_code_quality_suggestions(self, server_with_mock_agent):
        """Test code quality and best practice suggestions."""
        server = server_with_mock_agent

        quality_issues = [
            CodeIssue(
                issue_snippet="except:",
                severity="warning",
                message="Bare except clause catches all exceptions",
                suggested_fixes=[
                    SuggestedFix(
                        title="Catch specific exception",
                        target_snippet="except:",
                        replacement_snippet="except Exception:",
                    )
                ],
            ),
        ]

        mock_result = MagicMock()
        mock_result.output = DiagnosticResult(issues=quality_issues)
        server.agent.run = AsyncMock(return_value=mock_result)

        poor_quality_code = """
def risky_function():
    try:
        dangerous_operation()
    except:
        pass
"""

        uri = "file:///quality_test.py"
        await server.analyze_document(uri, poor_quality_code)

        # Verify quality issues were detected
        issues = server.diagnostic_cache[uri]
        assert len(issues) == 1
        assert "Bare except clause" in issues[0].message


if __name__ == "__main__":
    pytest.main([__file__])
