"""
LLM Factory — returns the configured LLM client based on .env settings.
Supports Anthropic (Claude) and OpenAI (GPT-4) interchangeably.
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

load_dotenv(override=True)


def get_llm(temperature: float = 0.0):
    """
    Returns a LangChain-compatible chat model based on LLM_PROVIDER in .env.

    Args:
        temperature: Controls randomness. 0.0 = deterministic (best for analysis tasks).

    Returns:
        ChatAnthropic or ChatOpenAI instance.
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in your .env file")
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=api_key,
            temperature=temperature,
        )

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in your .env file")
        return ChatOpenAI(
            model="gpt-4o",
            api_key=api_key,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Must be 'anthropic' or 'openai'."
        )