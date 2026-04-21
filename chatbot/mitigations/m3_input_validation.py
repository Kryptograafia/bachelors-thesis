"""M3 – Input validation and sanitization: remove instruction-override phrases, obfuscation patterns, syntactic separation."""
import re

from ..rag_chain import DEFAULT_FALLBACK_RESPONSE

# Patterns and keywords that suggest prompt injection. Sanitization removes or neutralizes matches.
INSTRUCTION_OVERRIDE_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|text)",
    r"disregard\s+(all\s+)?(previous|above)",
    r"forget\s+(everything|all\s+above)",
    r"override\s+(your\s+)?(instructions?|system)",
    r"do\s+not\s+follow\s+(your\s+)?(original\s+)?instructions",
    r"new\s+instruction[s]?\s*:",
    r"system\s*:\s*you\s+are",
    r"you\s+are\s+now\s+(in\s+)?(developer|admin|unrestricted)\s+mode",
]
# Long Base64-like runs (RFC 4648 alphabet); min length reduces accidental matches on short codes.
_BASE64_LIKE = r"[A-Za-z0-9+/=]{24,}"
# Long hex runs (e.g. OB-hex attacks); 0-9a-f only, any starting character in that set.
_HEX_LIKE = r"[0-9a-fA-F]{32,}"

OBFUSCATION_PATTERNS = [
    r"decode\s+this\s+and\s+execute",
    r"hex\s+decoding\s+required",  # OB-style framing
    r"convert\s+from\s+hex\s+and\s+run",
    r"decode\s+hex\s+then\s+perform",
    _BASE64_LIKE,
    _HEX_LIKE,
]
SYNTACTIC_SEP = re.compile(r"[\n\t]{2,}\s*(ignore|disregard|override|new\s+instruction)", re.I)

def _strip_suspicious(text: str, patterns: list[str]) -> str:
    out = text
    for p in patterns:
        out = re.sub(p, " ", out, flags=re.IGNORECASE)
    return out

def _strip_syntactic_sep(text: str) -> str:
    return SYNTACTIC_SEP.sub(" ", text)

def sanitize_input(user_input: str) -> str:
    if not user_input or not user_input.strip():
        return user_input
    cleaned = _strip_suspicious(user_input, INSTRUCTION_OVERRIDE_PATTERNS)
    cleaned = _strip_suspicious(cleaned, OBFUSCATION_PATTERNS)
    cleaned = _strip_syntactic_sep(cleaned)
    normalized = " ".join(cleaned.split())
    if not normalized.strip():
        return DEFAULT_FALLBACK_RESPONSE
    return normalized
