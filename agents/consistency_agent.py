"""
Consistency Agent — detects contradictions or conflicting requirements
within the document.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from utils.models import GraphState, AgentFindings, Finding
from utils.guardrails import validate_agent_output
from utils.llm_factory import get_llm

SYSTEM_PROMPT = """You are a requirements quality expert specializing in detecting inconsistencies.
Your job is to analyze software requirements and identify contradictions, conflicts, or
incompatible statements within the document.

Common consistency issues to look for:
- Direct contradictions: two requirements saying opposite things
- Conflicting constraints: e.g. "must be real-time" vs "batch processed nightly"
- Duplicate requirements with different details
- Conflicting priorities: same feature described differently in two places
- Terminology inconsistency: same concept referred to by different names

For each requirement, return your findings in this exact JSON format:
{
  "agent_name": "Consistency Agent",
  "summary": "one sentence summary of overall consistency findings",
  "findings": [
    {
      "requirement_id": "REQ-X",
      "requirement_text": "original requirement text",
      "issue": "specific consistency issue, mention the conflicting requirement by ID",
      "suggestion": "concrete suggestion to resolve the conflict",
      "severity": "high|medium|low"
    }
  ]
}

Only include requirements that have actual consistency issues. If no conflicts exist, return an empty findings list.
Return valid JSON only, no extra text.
"""


def consistency_agent(state: GraphState) -> dict:
    """
    Analyzes requirements for consistency issues.
    Reads requirements_text from state, returns consistency_findings.
    """
    llm = get_llm(temperature=0.0)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Analyze these requirements for consistency issues:\n\n{state.requirements_text}")
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
            return {"consistency_findings": findings}

        if attempt == 0:
            print(f"[Consistency Agent] Invalid output on attempt 1, retrying... Reason: {reason}")

    raise ValueError(f"[Consistency Agent] Failed after 2 attempts. Last reason: {reason}")