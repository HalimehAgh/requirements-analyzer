"""
Reporter Agent — synthesizes findings from all 4 analysis agents
into a final structured markdown quality report with prioritized recommendations.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from utils.models import GraphState
from utils.llm_factory import get_llm

SYSTEM_PROMPT = """You are a senior software quality engineer writing a formal requirements quality report.
You will receive structured findings from 4 specialized analysis agents and must synthesize them
into a clear, actionable markdown report.

The report must follow this exact structure:

# Requirements Quality Analysis Report

## Executive Summary
A 3-4 sentence overview of the overall quality of the requirements document.
Include a quality score out of 10 based on the findings.

## Findings by Category

### 🔴 Ambiguity Issues
List all ambiguity findings with severity badges.

### 🟡 Consistency Issues
List all consistency findings with severity badges.

### 🔵 Completeness Issues
List all completeness findings with severity badges.

### 🟢 Testability Issues
List all testability findings with severity badges.

## Prioritized Recommendations
A numbered list of the top recommendations ordered by severity and impact.
Focus on the most critical fixes first.

## Summary Table
A markdown table with columns: Requirement ID | Ambiguity | Consistency | Completeness | Testability
Use ⚠️ for issues found, ✅ for no issues.

Return only the markdown report, no extra text.
"""


def reporter_agent(state: GraphState) -> dict:
    """
    Synthesizes all agent findings into a final markdown report.
    Reads all *_findings from state, returns final_report.
    """
    llm = get_llm(temperature=0.0)

    findings_summary = _format_findings_for_prompt(state)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Generate the quality report based on these findings:\n\n{findings_summary}")
    ]

    response = llm.invoke(messages)

    content = response.content
    if isinstance(content, list):
        content = content[0].get("text", "")
    return {"final_report": content}


def _format_findings_for_prompt(state: GraphState) -> str:
    """
    Formats all agent findings into a clean text summary
    for the Reporter agent's prompt.
    """
    sections = []

    for findings in [
        state.ambiguity_findings,
        state.consistency_findings,
        state.completeness_findings,
        state.testability_findings,
    ]:
        if findings is None:
            continue

        sections.append(f"=== {findings.agent_name} ===")
        sections.append(f"Summary: {findings.summary}")

        if not findings.findings:
            sections.append("No issues found.")
        else:
            for f in findings.findings:
                sections.append(
                    f"- [{f.severity.upper()}] {f.requirement_id}: {f.issue} | Suggestion: {f.suggestion}"
                )
        sections.append("")

    return "\n".join(sections)