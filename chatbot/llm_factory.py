"""
LLM factory for the thesis chatbot. Default temperature 0 for reproducible runs.
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel

DEFAULT_MODEL_IDS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    # claude-3-5-haiku-20241022 was retired on the API (404 not_found_error).
    # Override with --model or set here; check Console / GET /v1/models for current IDs.
    "anthropic": "claude-3-haiku-20240307",
    "ollama": "llama3.1:8b",
}


def resolve_model_id(provider: str, model_id: str | None = None) -> str:
    """
    Resolved primary LLM id string for logging and Chat* construction.
    provider: openai | anthropic | ollama
    """
    provider = provider.lower().strip()
    return (model_id or DEFAULT_MODEL_IDS.get(provider) or DEFAULT_MODEL_IDS["openai"]).strip()


def get_llm(provider: str, model_id: str | None = None, temperature: float = 0.0) -> BaseChatModel:
    """
    provider: "openai" | "anthropic" | "ollama"
    model_id: optional override (e.g. "llama3.1:8b").
    """
    provider = provider.lower().strip()
    model = resolve_model_id(provider, model_id)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=temperature)
    raise ValueError(f"Unknown LLM provider: {provider}. Use openai, anthropic, or ollama.")
