"""Issue summary generation for classified log entries."""

import re
import logging
from typing import Dict, Any, Optional

from pydantic import BaseModel
from src.config import Config


class _LogSummary(BaseModel):
    category: str
    severity: str
    component: str
    description: str
    confidence: str
    recommended_action: str

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Severity heuristics                                                        #
# --------------------------------------------------------------------------- #

_SEVERITY_KEYWORDS = {
    "critical": ["fatal", "crash", "unrecoverable", "kernel panic", "segfault", "oom killed", "data loss"],
    "high": ["error", "failure", "failed", "exception", "timeout", "refused", "denied", "corrupt"],
    "medium": ["warning", "warn", "retry", "degraded", "slow", "latency", "high usage"],
    "low": ["info", "notice", "debug", "deprecated", "minor"],
}


def _infer_severity(text: str) -> str:
    """Rule-based severity inference from log text."""
    text_lower = text.lower()
    for severity, keywords in _SEVERITY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return severity
    return "medium"


def _extract_component(text: str) -> str:
    """Attempt to extract the affected component/service from a log entry."""
    # Match common patterns: [component], service=component, component:
    patterns = [
        r"\[([a-zA-Z][\w.-]+)\]",             # [nginx], [auth-service]
        r"service[=: ]+([a-zA-Z][\w.-]+)",     # service=payment-api
        r"^([a-zA-Z][\w.-]+):",                 # sshd: , kernel:
        r"module[=: ]+([a-zA-Z][\w.-]+)",      # module=database
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return "unknown"


# --------------------------------------------------------------------------- #
#  Template-based summarizer                                                  #
# --------------------------------------------------------------------------- #

class TemplateSummarizer:
    """Generate structured summaries using rule-based templates."""

    def summarize(self, log_entry: str, predicted_category: str, confidence: float) -> Dict[str, str]:
        severity = _infer_severity(log_entry)
        component = _extract_component(log_entry)

        # Truncate the raw log for the description field
        short_desc = log_entry[:200].strip()
        if len(log_entry) > 200:
            short_desc += "..."

        return {
            "category": predicted_category,
            "severity": severity,
            "component": component,
            "description": short_desc,
            "confidence": f"{confidence:.1%}",
            "recommended_action": self._recommend_action(predicted_category, severity),
        }

    @staticmethod
    def _recommend_action(category: str, severity: str) -> str:
        """Map category + severity to a recommended next step."""
        # Map root cause codes to specific actions (from the label reference)
        category_actions = {
            "RC-01": "Rotate credentials, refresh tokens, re-authenticate service accounts.",
            "RC-02": "Check DB health, increase connection pool size, investigate network latency.",
            "RC-03": "Check vendor status page, implement retry with backoff, activate fallback.",
            "RC-04": "Implement exponential backoff, request quota increase, distribute load.",
            "RC-05": "Fix upstream data source, update validation logic, sanitize inputs.",
            "RC-06": "Review IAM policies, update role assignments, audit permission grants.",
            "RC-07": "Scale resources, increase limits, investigate memory/disk leaks.",
            "RC-08": "Check network routes, verify DNS resolution, inspect firewall rules.",
        }
        # Try to match by RC code prefix
        for code, action in category_actions.items():
            if code in category:
                return action

        # Fall back to severity-based
        severity_actions = {
            "critical": "Escalate immediately to on-call engineering team.",
            "high": "Investigate within current sprint. Check recent deployments.",
            "medium": "Add to triage backlog. Monitor for recurrence.",
            "low": "Log for trend analysis. No immediate action required.",
        }
        return severity_actions.get(severity, "Review and triage accordingly.")



#  LLM-based summarizer (optional, requires API key)                          
# --------------------------------------------------------------------------- #

class LLMSummarizer:
    """Generate summaries using an LLM API call."""

    def __init__(self, config: Config):
        self.config = config

    def summarize(self, log_entry: str, predicted_category: str, confidence: float) -> Dict[str, str]:
        """Call LLM to generate a structured summary.

        Falls back to template if the API call fails.
        """
        try:
            return self._call_llm(log_entry, predicted_category, confidence)
        except Exception as e:
            logger.warning(f"LLM summary failed, falling back to template: {e}")
            fallback = TemplateSummarizer()
            return fallback.summarize(log_entry, predicted_category, confidence)

    def _call_llm(self, log_entry: str, predicted_category: str, confidence: float) -> Dict[str, str]:
        if self.config.llm_provider == "gemini":
            import os
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=os.environ["GEMINI_KEY"])
            response = client.models.generate_content(
                model=self.config.llm_model,
                contents=[self._build_prompt(log_entry, predicted_category, confidence)],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=_LogSummary,
                    temperature=0,
                ),
            )
            parsed: _LogSummary = response.parsed
            return {
                "category": parsed.category,
                "severity": parsed.severity,
                "component": parsed.component,
                "description": parsed.description,
                "confidence": parsed.confidence,
                "recommended_action": parsed.recommended_action,
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")

    @staticmethod
    def _build_prompt(log_entry: str, predicted_category: str, confidence: float) -> str:
        return f"""Analyze this system error log entry and return a structured summary.

Log entry: {log_entry}
Predicted root cause category: {predicted_category}
Classification confidence: {confidence:.1%}

Fields to populate:
- category: the root cause category as given
- severity: one of "critical", "high", "medium", "low"
- component: the affected system component extracted from the log
- description: 1-2 sentence plain-language description of the issue
- confidence: the classification confidence as a percentage string (e.g. "87.5%")
- recommended_action: a specific next step for the ops team"""



class IssueSummarizer:
    """Facade that delegates to the configured summarizer backend."""

    def __init__(self, config: Config):
        if config.summarizer_type == "llm":
            self._backend = LLMSummarizer(config)
        else:
            self._backend = TemplateSummarizer()

    def summarize(self, **kwargs) -> Dict[str, str]:
        return self._backend.summarize(**kwargs)
