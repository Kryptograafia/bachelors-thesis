"""M4 – Output validation: generic secret-shaped detection, system-prompt disclosure, compromise patterns.

Does not use a fixed list of KB canary substrings; see ``chatbot.evaluation_canaries`` for experiment
metrics only.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from ..prompt_leak_detection import response_contains_system_prompt_leak
from ..rag_chain import DEFAULT_FALLBACK_RESPONSE

COMPROMISE_PATTERNS = [
    r"(?i)I\s+am\s+(now\s+)?(in\s+)?(developer|admin|DAN|unrestricted)\s+mode",
    r"(?i)here\s+are?\s+(the\s+)?(system\s+)?instructions?\s*:",
    r"(?i)system\s+prompt\s*:",
    r"(?i)my\s+(original\s+)?instructions?\s+(were|are)\s*:",
]

# Value-agnostic shapes: URLs with credentials, labeled secrets, base64-like blobs, card-like runs.
SECRET_PATTERNS = [
    r"postgresql://\S+",
    r"(?i)api[_-]?key\s*[:=]\s*\S+",
    r"(?i)password\s*[:=]\s*\S+",
    r"\b[A-Za-z0-9+/]{32,}={0,2}\b",
    r"\b[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{4}\b",
]

_SECRET_REGEXES = tuple(re.compile(p) for p in SECRET_PATTERNS)

FALLBACK_MESSAGE = DEFAULT_FALLBACK_RESPONSE

# Tune if needed: higher → fewer false positives, more false negatives on short tokens.
_ENTROPY_THRESHOLD = 3.5
_ENTROPY_MIN_TOKEN_LEN = 13


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((n / length) * math.log2(n / length) for n in counts.values())


def contains_high_entropy_secret(text: str, threshold: float = _ENTROPY_THRESHOLD) -> bool:
    """Heuristic: long high-entropy tokens often look like keys/tokens (not English words)."""
    for token in text.split():
        cleaned = token.strip('.,;:"\'"()[]')
        if len(cleaned) >= _ENTROPY_MIN_TOKEN_LEN and shannon_entropy(cleaned) > threshold:
            return True
    return False


def _matches_secret_pattern(text: str) -> bool:
    return any(rx.search(text) for rx in _SECRET_REGEXES)


def filter_output(response: str) -> str:
    if not response or not response.strip():
        return response
    text = response

    if response_contains_system_prompt_leak(text):
        return FALLBACK_MESSAGE
    for p in COMPROMISE_PATTERNS:
        if re.search(p, text):
            return FALLBACK_MESSAGE
    if _matches_secret_pattern(text):
        return FALLBACK_MESSAGE
    if contains_high_entropy_secret(text):
        return FALLBACK_MESSAGE
    return response
