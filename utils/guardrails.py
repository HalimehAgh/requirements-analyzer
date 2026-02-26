"""
Guardrails for the Requirements Quality Analyzer.

- Input Guardrail: validates the input before entering the graph
- Output Guardrail: validates each agent's JSON response before parsing
"""

import json
from langchain_core.messages import HumanMessage, SystemMessage
from utils.llm_factory import get_llm


# ---------------------------------------------------------------------------
# Input Guardrail
# ---------------------------------------------------------------------------

INPUT_GUARD_PROMPT = """You are a document classifier. Your only job is to determine whether
the given text is a software requirements document or something else entirely.

A valid requirements document:
- Describes features, behaviors, or constraints of a software system
- May use terms like "the system shall", "users must", "the application should"
- Can be informal but must be about software functionality

Respond with valid JSON only:
{
  "is_valid": true or false,
  "reason": "one sentence explanation"
}
"""


def validate_input(text: str) -> tuple[bool, str]:
    """
    Checks whether the input text is a valid requirements document.

    Returns:
        (True, "") if valid
        (False, reason) if invalid
    """
    if not text or len(text.strip()) < 20:
        return False, "Input is too short to be a requirements document."

    if len(text.strip()) > 10000:
        return False, "Input is too long. Please limit to 10,000 characters for this demo."

    llm = get_llm(temperature=0.0)

    messages = [
        SystemMessage(content=INPUT_GUARD_PROMPT),
        HumanMessage(content=f"Classify this text:\n\n{text}")
    ]

    response = llm.invoke(messages)

    try:
        result = json.loads(response.content)
        if result["is_valid"]:
            return True, ""
        else:
            return False, result["reason"]
    except (json.JSONDecodeError, KeyError):
        # If the guardrail itself fails to parse, fail open and allow the input
        return True, ""


# ---------------------------------------------------------------------------
# Output Guardrail
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"agent_name", "summary", "findings"}
REQUIRED_FINDING_KEYS = {"requirement_id", "requirement_text", "issue", "suggestion", "severity"}
VALID_SEVERITIES = {"high", "medium", "low"}


def extract_json(raw_content: str) -> str:
    """
    Cleans LLM response to extract pure JSON.
    Handles markdown code blocks and extra text around JSON.
    """
    content = raw_content.strip()

    # Handle ```json ... ``` or ``` ... ```
    if "```" in content:
        import re
        match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if match:
            content = match.group(1).strip()

    # Find the first { and last } to extract JSON object
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1:
        content = content[start:end + 1]

    return content


def validate_agent_output(raw_content: str) -> tuple[bool, dict | None, str]:
    """
    Validates that an agent's JSON response matches the expected schema.

    Returns:
        (True, parsed_dict, "") if valid
        (False, None, reason) if invalid
    """
    # Clean the response first
    cleaned = extract_json(raw_content)

    if not cleaned:
        return False, None, "Response is empty after cleaning."

    # Check 1: is it valid JSON?
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return False, None, f"Response is not valid JSON: {e}"

    # Check 2: does it have required top-level keys?
    missing_keys = REQUIRED_KEYS - data.keys()
    if missing_keys:
        return False, None, f"Response is missing required keys: {missing_keys}"

    # Check 3: is findings a list?
    if not isinstance(data["findings"], list):
        return False, None, "findings must be a list."

    # Check 4: validate each finding
    for i, finding in enumerate(data["findings"]):
        missing = REQUIRED_FINDING_KEYS - finding.keys()
        if missing:
            return False, None, f"Finding {i} is missing keys: {missing}"
        if finding["severity"] not in VALID_SEVERITIES:
            return False, None, f"Finding {i} has invalid severity: {finding['severity']}"

    return True, data, ""