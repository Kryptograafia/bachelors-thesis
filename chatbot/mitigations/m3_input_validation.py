"""
M3 – Input sanitization (simple deny-list).

Strips substrings matching:
1. **Keyword / term blocklist** — single-token triggers (word boundaries), plus chat-template tokens.
2. **Encoding-style runs** — long Base64- or hex-like spans without decoding them.
3. **Syntactic separation** — instructions hidden after blank lines/tabs.

Matches are replaced with a space; whitespace is normalized; length capped; empty → DEFAULT_FALLBACK_RESPONSE.

**False positives:** broad keywords like `ignore` or `paste` can appear in benign support text.
This is intentionally simple; refine the list for production.
"""
from __future__ import annotations

import re

from ..rag_chain import DEFAULT_FALLBACK_RESPONSE

_MAX_INPUT_CHARS = 2500

# High-signal instruction / exfil / attack-vocabulary terms (word-boundary; case-insensitive).
_KEYWORD_PATTERNS = [
    r"\bignore\b",
    r"\bdisregard\b",
    r"\boverride\b",
    r"\bforget\b",
    r"\bpaste\b",
    r"\bextract\b",
    r"\bdump\b",
    r"\bleak\b",
    r"\breveal\b",
    r"\bjailbreak\b",
    r"\bbypass\b",
    r"\binject\b",
    r"\bexfiltrate\b",
    r"\bdecode\b",
    r"\bdecrypt\b",
    r"\bbase64\b",
    r"\bhex\b",
    r"\brot[- ]?13\b",
    r"\bcaesar\b",
    r"\bcipher\b",
    r"\bsystem\s+prompt\b",
    r"\bnew\s+instruction\b",
]

# Delimiter / role-smuggling tokens (prompt-in-prompt class).
_DELIMITER_PATTERNS = [
    r"\[\s*INST\s*\]",
    r"\[\s*SYSTEM\s*\]",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"\boverride\s*:\s*",
]

# Long opaque runs (typical of Base64/hex blobs; not a full decoder).
_BASE64_LIKE = r"[A-Za-z0-9+/=]{32,}"
_HEX_LIKE = r"\b(?:0x)?[0-9a-fA-F]{32,}\b"

_ENCODING_RUN_PATTERNS = [
    _BASE64_LIKE,
    _HEX_LIKE,
]

_SYNTACTIC_SEP = re.compile(
    r"[\n\t]{2,}\s*(ignore|disregard|override|new\s+instruction)\b",
    re.I,
)

# One flat pass order: keywords → delimiters → encoding runs → layout.
_ALL_STRIP_PATTERNS: list[str] = (
    _KEYWORD_PATTERNS + _DELIMITER_PATTERNS + _ENCODING_RUN_PATTERNS
)


def _apply_patterns(text: str, patterns: list[str]) -> str:
    out = text
    for pattern in patterns:
        out = re.sub(pattern, " ", out, flags=re.IGNORECASE)
    return out


def _strip_control_chars(text: str) -> str:
    return "".join(c for c in text if ord(c) >= 32 or c in "\n\t")


def sanitize_input(user_input: str) -> str:
    if not user_input or not user_input.strip():
        return user_input

    text = user_input.strip()
    text = _apply_patterns(text, _ALL_STRIP_PATTERNS)
    text = _SYNTACTIC_SEP.sub(" ", text)
    text = _strip_control_chars(text)
    text = " ".join(text.split())

    if len(text) > _MAX_INPUT_CHARS:
        text = text[:_MAX_INPUT_CHARS]

    if not text.strip():
        return DEFAULT_FALLBACK_RESPONSE

    return text
