"""Unit tests for core pipeline components."""

import pytest
from src.config import Config
from src.preprocessing import LogPreprocessor
from src.summarizer import TemplateSummarizer, _infer_severity, _extract_component


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def preprocessor(config):
    return LogPreprocessor(config)


class TestPreprocessor:
    def test_clean_text_lowercases(self, preprocessor):
        assert preprocessor.clean_text("ERROR in Module") == "error in module"

    def test_clean_text_normalizes_ip(self, preprocessor):
        result = preprocessor.clean_text("Connection from 192.168.1.1 refused")
        assert "<IP_ADDR>" in result
        assert "192.168.1.1" not in result

    def test_clean_text_normalizes_hex(self, preprocessor):
        result = preprocessor.clean_text("Segfault at 0x7fff5fbff8c0")
        assert "<HEX_ADDR>" in result

    def test_clean_text_handles_empty(self, preprocessor):
        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text(None) == ""
        assert preprocessor.clean_text(123) == ""

    def test_clean_text_strips_timestamps(self, preprocessor):
        result = preprocessor.clean_text("2024-01-15T10:30:00Z ERROR disk full")
        assert "2024" not in result
        assert "error disk full" in result


class TestSeverityInference:
    def test_critical(self):
        assert _infer_severity("FATAL: kernel panic") == "critical"

    def test_high(self):
        assert _infer_severity("Connection timeout on port 443") == "high"

    def test_medium(self):
        assert _infer_severity("WARNING: high memory usage") == "medium"

    def test_low(self):
        assert _infer_severity("DEBUG: request processed") == "low"

    def test_default_medium(self):
        assert _infer_severity("something happened") == "medium"


class TestComponentExtraction:
    def test_bracket_format(self):
        assert _extract_component("[nginx] upstream error") == "nginx"

    def test_colon_format(self):
        assert _extract_component("sshd: connection closed") == "sshd"

    def test_service_equals(self):
        assert _extract_component("service=payment-api request failed") == "payment-api"

    def test_unknown_fallback(self):
        assert _extract_component("something broke") == "unknown"


class TestTemplateSummarizer:
    def test_summary_structure(self):
        summarizer = TemplateSummarizer()
        result = summarizer.summarize(
            log_entry="[auth-service] ERROR: login failed for user",
            predicted_category="authentication_failure",
            confidence=0.92,
        )
        assert "category" in result
        assert "severity" in result
        assert "component" in result
        assert "description" in result
        assert "recommended_action" in result
        assert result["category"] == "authentication_failure"
        assert result["component"] == "auth-service"
