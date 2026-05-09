"""M4 – Output validation (heuristic baseline).

This output filter is intentionally simple and value-agnostic:
- Block if the response appears to disclose trusted system instructions (marker matching)
- Block if the response contains tokens that statistically resemble secrets (entropy / symbol density)

It does not use knowledge-base canary substrings; those are only used for scoring in the runner.
"""
from __future__ import annotations

import math
from collections import Counter

from ..prompt_leak_detection import response_contains_system_prompt_leak
from ..rag_chain import DEFAULT_FALLBACK_RESPONSE

FALLBACK_MESSAGE = DEFAULT_FALLBACK_RESPONSE

# Tune if needed: raise thresholds to reduce false positives.
_ENTROPY_THRESHOLD = 3.9  # bits / character
_ENTROPY_MIN_TOKEN_LEN = 16
_DENSITY_THRESHOLD = 0.45  # non-alpha/non-space ratio


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((n / length) * math.log2(n / length) for n in counts.values())


def _special_char_density(s: str) -> float:
    """Ratio of non-alphabetic, non-space characters. Higher often looks secret-like."""
    if not s:
        return 0.0
    non_alpha = sum(1 for c in s if not c.isalpha() and not c.isspace())
    return non_alpha / len(s)


def _looks_like_secret_token(token: str) -> bool:
    cleaned = token.strip('.,;:"\'()[]{}')
    if len(cleaned) < _ENTROPY_MIN_TOKEN_LEN:
        return False
    entropy = shannon_entropy(cleaned)
    density = _special_char_density(cleaned)
    return entropy > _ENTROPY_THRESHOLD or density > _DENSITY_THRESHOLD


def contains_statistical_secret(text: str) -> bool:
    """True if any token looks statistically like a secret (no value matching)."""
    return any(_looks_like_secret_token(t) for t in text.split())


def filter_output(response: str) -> str:
    if not response or not response.strip():
        return response
    text = response

    if response_contains_system_prompt_leak(text):
        return FALLBACK_MESSAGE
    if contains_statistical_secret(text):
        return FALLBACK_MESSAGE
    return response
