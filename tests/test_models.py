import pytest
from pydantic import ValidationError
from unittest.mock import patch

# Mock Agent before importing to avoid API key requirements
with patch("ai_lsp.main.Agent"):
    from ai_lsp.main import CodeIssue, DiagnosticResult, Settings, SuggestedFix


class TestPydanticModels:
    """Test Pydantic model validation and behavior."""

    class TestSuggestedFix:
        """Test SuggestedFix model."""

        def test_valid_suggested_fix(self):
            fix = SuggestedFix(
                title="Replace with logging",
                target_snippet="print",
                replacement_snippet="logging.info",
            )

            assert fix.title == "Replace with logging"
            assert fix.target_snippet == "print"
            assert fix.replacement_snippet == "logging.info"

        def test_suggested_fix_required_fields(self):
            """Test that all fields are required."""
            with pytest.raises(ValidationError) as exc_info:
                SuggestedFix()

            errors = exc_info.value.errors()
            error_fields = {error["loc"][0] for error in errors}
            assert "title" in error_fields
            assert "target_snippet" in error_fields
            assert "replacement_snippet" in error_fields

    class TestCodeIssue:
        """Test CodeIssue model."""

        def test_valid_code_issue_minimal(self):
            issue = CodeIssue(
                issue_snippet="print",
                severity="warning",
                message="Use logging instead",
            )

            assert issue.issue_snippet == "print"
            assert issue.severity == "warning"
            assert issue.message == "Use logging instead"
            assert issue.suggested_fixes is None

        def test_code_issue_required_fields(self):
            """Test that required fields are enforced."""
            with pytest.raises(ValidationError) as exc_info:
                CodeIssue()

            errors = exc_info.value.errors()
            error_fields = {error["loc"][0] for error in errors}
            assert "issue_snippet" in error_fields
            assert "severity" in error_fields
            assert "message" in error_fields

    class TestDiagnosticResult:
        """Test DiagnosticResult model."""

        def test_valid_diagnostic_result_empty(self):
            result = DiagnosticResult(issues=[])
            assert result.issues == []

        def test_diagnostic_result_required_field(self):
            """Test that issues field is required."""
            with pytest.raises(ValidationError) as exc_info:
                DiagnosticResult()

            errors = exc_info.value.errors()
            error_fields = {error["loc"][0] for error in errors}
            assert "issues" in error_fields

    class TestSettings:
        """Test Settings model."""

        def test_default_settings(self):
            settings = Settings()
            # The default may be overridden by environment variables in tests
            assert settings.ai_lsp_model in ["google-gla:gemini-2.5-pro", "gpt-4o"]
            assert settings.debounce_ms == 1000

        def test_custom_settings(self):
            settings = Settings(
                ai_lsp_model="gpt-4o",
                debounce_ms=500,
            )
            assert settings.ai_lsp_model == "gpt-4o"
            assert settings.debounce_ms == 500


if __name__ == "__main__":
    pytest.main([__file__])
