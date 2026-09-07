"""LLM provider helpers for IntelliAssist AI.

The project intentionally keeps provider creation in one module.  The
Gemini model is a current stable model and does not use the old 1.5/2.5
model IDs that caused the 404 errors in the original project.
"""

from __future__ import annotations

import os

PROVIDERS = ["Gemini", "OpenAI"]

# Stable Gemini model.  Override with GEMINI_MODEL if Google changes the
# recommended model again; the sidebar does not expose this implementation
# detail to the user.
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_llm(provider: str, api_key: str):
    """Return a LangChain chat model or ``None`` when no key is supplied."""
    api_key = (api_key or "").strip()
    if not api_key:
        return None

    if provider == "Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Do not pass temperature/top_p/top_k here. Gemini 3.x does not
        # support the old sampling parameters used by the original app.
        return ChatGoogleGenerativeAI(
            model=DEFAULT_GEMINI_MODEL,
            google_api_key=api_key,
        )

    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=DEFAULT_OPENAI_MODEL,
            api_key=api_key,
            temperature=0.2,
        )

    raise ValueError(f"Unknown provider: {provider}")


def provider_model_name(provider: str) -> str:
    """Return the model name used by the selected provider."""
    return DEFAULT_GEMINI_MODEL if provider == "Gemini" else DEFAULT_OPENAI_MODEL
