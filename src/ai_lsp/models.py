from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai.models import KnownModelName
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ai_lsp_model: KnownModelName = Field(default="google-gla:gemini-2.5-flash")
    debounce_ms: int = Field(default=1000)
    max_cache_size: int = Field(default=50)
    configure_logfire: bool = Field(default=True)
    otel_exporter_otlp_endpoint: str = Field(default="http://localhost:4318")


class SuggestedFix(BaseModel):
    title: str
    target_snippet: str
    replacement_snippet: str


class CodeIssue(BaseModel):
    issue_snippet: str
    severity: Literal["error", "warning", "info", "hint"]
    message: str
    suggested_fixes: list[SuggestedFix] | None = None


class DiagnosticResult(BaseModel):
    issues: list[CodeIssue]
