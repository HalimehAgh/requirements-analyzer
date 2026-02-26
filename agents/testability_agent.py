"""
Testability Agent — evaluates whether each requirement can be
objectively verified or tested.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from utils.models import GraphState, AgentFindings, Finding
from utils.llm_factory import get_llm
from utils.guardrails import validate_agent_output

SYSTEM_PROMPT = """You are a requirements quality expert specializing in testability analysis.
Your job is to analyze software requirements and identify whether each requirement can be
objectively verified, measured, or tested.

Common testability issues to look for:
- No measurable acceptance criteria: how do we know when it's done?
- Untestable qualifiers: "the system shall be reliable", "the UI shall be pleasant"
- Missing numeric thresholds: "fast response time" vs "response time < 2 seconds"
- Subjective success criteria: "users should be satisfied"
- Requirements that depend on uncontrollable external factors
- No clear pass/fail condition for verification

For each requirement, return your findings in this exact JSON format:
{
  "agent_name": "Testability Agent",
  "summary": "one sentence summary of overall testability findings",
  "findings": [
    {
      "requirement_id": "REQ-X",
      "requirement_text": "original requirement text",
      "issue": "specific reason why this requirement is difficult to test or verify",
      "suggestion": "concrete rewrite with measurable acceptance criteria",
      "severity": "high|medium|low"
    }
  ]
}

Only include requirements that have actual testability issues. If a requirement is clearly testable, skip it.
Return valid JSON only, no extra text.
"""


def testability_agent(state: GraphState) -> dict:
    """
    Analyzes requirements for testability issues.
    Reads requirements_text from state, returns testability_findings.
    """
    llm = get_llm(temperature=0.0)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Analyze these requirements for testability issues:\n\n{state.requirements_text}")
    ]

    for attempt in range(2):
        response = llm.invoke(messages)
        is_valid, data, reason = validate_agent_output(response.content)

        if is_valid:
            findings = AgentFindings(
                agent_name=data["agent_name"],
                summary=data["summary"],
                findings=[Finding(**f) for f in data["findings"]]
            )
            return {"testability_findings": findings}

        if attempt == 0:
            print(f"[Testability Agent] Invalid output on attempt 1, retrying... Reason: {reason}")

    raise ValueError(f"[Testability Agent] Failed after 2 attempts. Last reason: {reason}")