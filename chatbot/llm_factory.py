"""
LLM factory for the thesis chatbot. Default temperature 0 for reproducible runs.
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel

DEFAULT_MODEL_IDS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1:8b",
    # Google Generative Language API; set GOOGLE_API_KEY
    "google": "gemini-2.5-flash-lite",
}


def resolve_model_id(provider: str, model_id: str | None = None) -> str:
    """
    Resolved primary LLM id string for logging and Chat* construction.
    provider: openai | ollama | google
    """
    provider = provider.lower().strip()
    return (model_id or DEFAULT_MODEL_IDS.get(provider) or DEFAULT_MODEL_IDS["openai"]).strip()


def get_llm(provider: str, model_id: str | None = None, temperature: float = 0.3) -> BaseChatModel:
    """
    provider: "openai" | "ollama" | "google"
    model_id: optional override (e.g. "llama3.1:8b", "gemini-2.5-flash-lite").
    """
    provider = provider.lower().strip()
    model = resolve_model_id(provider, model_id)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=temperature)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    raise ValueError(
        f"Unknown LLM provider: {provider}. Use openai, ollama, or google."
    )
