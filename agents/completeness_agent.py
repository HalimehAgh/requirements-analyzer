"""
Completeness Agent — flags missing edge cases, undefined terms,
or gaps in coverage within the requirements document.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from utils.models import GraphState, AgentFindings, Finding
from utils.llm_factory import get_llm
from utils.guardrails import validate_agent_output

SYSTEM_PROMPT = """You are a requirements quality expert specializing in detecting completeness issues.
Your job is to analyze software requirements and identify missing information, undefined terms,
unhandled edge cases, or gaps in coverage.

Common completeness issues to look for:
- Undefined terms or acronyms used without explanation
- Missing error handling: what happens when something goes wrong?
- Unhandled edge cases: empty inputs, maximum limits, concurrent users
- Missing non-functional requirements: performance, security, scalability
- Incomplete user roles: are all actors and their permissions defined?
- Missing preconditions or postconditions for key operations
- No mention of data validation or input constraints

For each requirement, return your findings in this exact JSON format:
{
  "agent_name": "Completeness Agent",
  "summary": "one sentence summary of overall completeness findings",
  "findings": [
    {
      "requirement_id": "REQ-X",
      "requirement_text": "original requirement text",
      "issue": "specific description of what is missing or undefined",
      "suggestion": "concrete suggestion of what should be added",
      "severity": "high|medium|low"
    }
  ]
}

Only include requirements that have actual completeness issues. If a requirement is fully specified, skip it.
Return valid JSON only, no extra text.
"""


def completeness_agent(state: GraphState) -> dict:
    """
    Analyzes requirements for completeness issues.
    Reads requirements_text from state, returns completeness_findings.
    Includes output guardrail with one retry on invalid response.
    """
    llm = get_llm(temperature=0.0)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Analyze these requirements for completeness issues:\n\n{state.requirements_text}")
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
            return {"completeness_findings": findings}

        if attempt == 0:
            print(f"[Completeness Agent] Invalid output on attempt 1, retrying... Reason: {reason}")

    raise ValueError(f"[Completeness Agent] Failed after 2 attempts. Last reason: {reason}")