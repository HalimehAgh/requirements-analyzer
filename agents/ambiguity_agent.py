"""
Ambiguity Agent — identifies vague or unclear requirements
that could lead to misinterpretation during development.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from utils.models import GraphState, AgentFindings, Finding
from utils.guardrails import validate_agent_output
from utils.llm_factory import get_llm

SYSTEM_PROMPT = """You are a requirements quality expert specializing in detecting ambiguity.
Your job is to analyze software requirements and identify vague, unclear, or subjective language
that could lead to different interpretations by different stakeholders.

Common ambiguity patterns to look for:
- Vague terms: "fast", "user-friendly", "efficient", "appropriate", "sufficient"
- Unclear pronouns: "it", "they", "this" without clear referents
- Missing subject: who performs the action?
- Passive voice hiding responsibility: "the data will be processed"
- Subjective qualifiers: "good", "easy", "modern", "robust"

For each requirement, return your findings in this exact JSON format:
{
  "agent_name": "Ambiguity Agent",
  "summary": "one sentence summary of overall ambiguity findings",
  "findings": [
    {
      "requirement_id": "REQ-X",
      "requirement_text": "original requirement text",
      "issue": "specific ambiguity issue description",
      "suggestion": "concrete rewrite suggestion",
      "severity": "high|medium|low"
    }
  ]
}

Only include requirements that have actual ambiguity issues. If a requirement is clear, skip it.
Return valid JSON only, no extra text.
"""


def ambiguity_agent(state: GraphState) -> dict:
    """
    Analyzes requirements for ambiguity issues.
    Reads requirements_text from state, returns ambiguity_findings.
    Includes output guardrail with one retry on invalid response.
    """
    llm = get_llm(temperature=0.0)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Analyze these requirements for ambiguity:\n\n{state.requirements_text}")
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
            return {"ambiguity_findings": findings}

        if attempt == 0:
            print(f"[Ambiguity Agent] Invalid output on attempt 1, retrying... Reason: {reason}")

    raise ValueError(f"[Ambiguity Agent] Failed after 2 attempts. Last reason: {reason}")