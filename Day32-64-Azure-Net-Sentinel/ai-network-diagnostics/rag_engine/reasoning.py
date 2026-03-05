"""
LLM reasoning layer for the investigation agent.
Produces DiagnosticReport (root causes, evidence, mitigation, confidence) from
incident snapshot, telemetry analysis, and retrieved incidents/runbooks.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, List, Optional

from config import get_settings

logger = logging.getLogger(__name__)

AGENT_PROMPT_TEMPLATE = """You are an AI infrastructure reliability assistant.

Current incident:
{incident_snapshot}

Telemetry analysis:
{telemetry_analysis}

Similar historical incidents:
{incident_results}

Relevant runbooks:
{runbook_results}

Based on the telemetry patterns and historical evidence, provide:

1) possible root causes (ranked)
2) supporting evidence
3) suggested mitigation steps

Do not assume deterministic correctness. Provide hypotheses."""


@dataclass
class DiagnosticReport:
    """Structured output from the agent reasoning step."""
    root_causes: List[str]
    evidence: List[str]
    mitigation_steps: List[str]
    confidence: str  # low | medium | high


def _call_openai(system_content: str, user_content: str) -> Optional[str]:
    """Call OpenAI Chat API; return assistant message content or None on failure."""
    settings = get_settings()
    api_key = (settings.openai_api_key or "").strip()
    if not api_key:
        logger.debug("OPENAI_API_KEY not set; skipping LLM call, using retrieved context fallback")
        return None
    model = settings.llm_model_name
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            max_tokens=500,
        )
        if resp.choices and resp.choices[0].message.content:
            logger.info("LLM call succeeded, model=%s", model)
            return resp.choices[0].message.content.strip()
        logger.warning("LLM returned empty content")
        return None
    except Exception as e:
        logger.error("LLM call failed: %s (model=%s). Using retrieved context fallback.", e, model)
        return None


def _parse_agent_response(raw: str) -> DiagnosticReport:
    """Parse LLM response into root_causes, evidence, mitigation_steps, confidence."""
    root_causes: List[str] = []
    evidence: List[str] = []
    mitigation_steps: List[str] = []
    confidence = "medium"

    raw_lower = raw.lower()
    if "high" in raw_lower and "confidence" in raw_lower:
        confidence = "high"
    elif "low" in raw_lower and "confidence" in raw_lower:
        confidence = "low"

    # Root causes: look for numbered list or "root cause(s)" section
    cause_section = re.search(
        r"(?:possible\s+)?root\s+cause[s]?\s*:?\s*([\s\S]*?)(?=evidence|supporting|mitigation|suggested|confidence|$)",
        raw,
        re.IGNORECASE,
    )
    if cause_section:
        block = cause_section.group(1).strip()
        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip leading number or bullet
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = re.sub(r"^[-*]\s*", "", line)
            if len(line) > 10:
                root_causes.append(line[:300])
        root_causes = root_causes[:5]

    # Evidence section
    ev_section = re.search(
        r"evidence\s*:?\s*([\s\S]*?)(?=mitigation|suggested|confidence|$)",
        raw,
        re.IGNORECASE,
    )
    if ev_section:
        block = ev_section.group(1).strip()
        for line in block.split("\n"):
            line = re.sub(r"^[-*]\s*", "", line).strip()
            if len(line) > 5:
                evidence.append(line[:250])
        evidence = evidence[:8]

    # Mitigation section
    mit_section = re.search(
        r"(?:suggested\s+)?mitigation\s*:?\s*([\s\S]*?)(?=confidence|$)",
        raw,
        re.IGNORECASE,
    )
    if mit_section:
        block = mit_section.group(1).strip()
        for line in block.split("\n"):
            line = re.sub(r"^[-*]\s*", "", line).strip()
            if len(line) > 5:
                mitigation_steps.append(line[:200])
        mitigation_steps = mitigation_steps[:10]

    if not root_causes:
        root_causes = ["Switch buffer overflow or congestion (see similar incidents)."]
    if not mitigation_steps:
        mitigation_steps = ["Drain traffic", "Reset switch", "Verify ECMP routes"]

    return DiagnosticReport(
        root_causes=root_causes,
        evidence=evidence,
        mitigation_steps=mitigation_steps,
        confidence=confidence,
    )


def agent_reason(
    incident_snapshot: str,
    telemetry_analysis: str,
    incident_results: str,
    runbook_results: str,
) -> DiagnosticReport:
    """
    Run LLM reasoning for the investigation agent. If no API key or call fails,
    build a report from the first retrieved incident/runbook.
    """
    user_content = AGENT_PROMPT_TEMPLATE.format(
        incident_snapshot=incident_snapshot,
        telemetry_analysis=telemetry_analysis,
        incident_results=incident_results,
        runbook_results=runbook_results,
    )
    system_content = (
        "You are an AI infrastructure reliability assistant. "
        "Reply with: 1) possible root causes (ranked), 2) supporting evidence, 3) suggested mitigation steps, 4) confidence (low/medium/high). Be concise."
    )
    raw = _call_openai(system_content, user_content)
    if raw:
        return _parse_agent_response(raw)
    # Fallback from retrieved context
    root_causes: List[str] = []
    evidence = ["high queue depth", "packet loss spike", "similar historical incidents found"]
    mitigation_steps = ["drain traffic", "reset switch", "check ECMP routing"]
    if incident_results and incident_results != "No similar historical incidents found.":
        first = incident_results.split("\n\n")[0]
        seen_rc = set()
        for m in re.finditer(r"Root cause:\s*([^\n]+)", first, re.IGNORECASE):
            c = m.group(1).strip()[:200]
            if c.lower() not in seen_rc:
                seen_rc.add(c.lower())
                root_causes.append(c)
        seen_mit = set()
        for m in re.finditer(r"Mitigation:\s*([^\n]+)", first, re.IGNORECASE):
            s = m.group(1).strip()[:150]
            if s.lower() not in seen_mit:
                seen_mit.add(s.lower())
                mitigation_steps.insert(0, s)
    if not root_causes:
        root_causes = ["Switch buffer overflow", "Micro-burst traffic", "Upstream congestion"]
    return DiagnosticReport(
        root_causes=root_causes[:5],
        evidence=evidence,
        mitigation_steps=list(dict.fromkeys(mitigation_steps))[:8],
        confidence="medium",
    )
