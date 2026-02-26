"""
Shared data models for the Requirements Quality Analyzer.
All agents use these structured types to ensure consistent output.
"""

from pydantic import BaseModel, Field
from typing import Literal


class Finding(BaseModel):
    """A single quality issue found by an agent."""
    requirement_id: str = Field(description="ID of the requirement, e.g. REQ-1")
    requirement_text: str = Field(description="The original requirement text")
    issue: str = Field(description="Clear description of the quality issue found")
    suggestion: str = Field(description="Concrete suggestion to fix the issue")
    severity: Literal["high", "medium", "low"] = Field(
        description="Severity of the issue: high, medium, or low"
    )


class AgentFindings(BaseModel):
    """Structured output from a quality analysis agent."""
    agent_name: str
    findings: list[Finding]
    summary: str = Field(description="One-sentence summary of the overall findings")


class GraphState(BaseModel):
    """
    The shared state that flows through the LangGraph pipeline.
    Each agent reads from and writes to this state.
    """
    requirements_text: str = Field(description="The raw input requirements document")
    ambiguity_findings: AgentFindings | None = None
    consistency_findings: AgentFindings | None = None
    completeness_findings: AgentFindings | None = None
    testability_findings: AgentFindings | None = None
    final_report: str | None = Field(
        default=None,
        description="The synthesized markdown report from the Reporter agent"
    )