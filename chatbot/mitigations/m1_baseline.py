"""M1 – Baseline system prompt (unmitigated)."""
from ..rag_chain import DEFAULT_SYSTEM_PROMPT

def get_system_prompt() -> str:
    return DEFAULT_SYSTEM_PROMPT
