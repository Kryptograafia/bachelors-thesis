# Security mitigations M1–M6. Runner composes these; each can be toggled.
# m4 is lazy-imported to avoid import cycles (m4 -> prompt_leak_detection -> mitigations).
from .m1_baseline import get_system_prompt as m1_get_system_prompt
from .m2_prompt_hardening import get_system_prompt as m2_get_system_prompt
from .m3_input_validation import sanitize_input as m3_sanitize_input
from .m5_structured_prompt import get_system_prompt as m5_get_system_prompt, wrap_user_input as m5_wrap_user_input
from .m6_guardrail_llm import sanitize_input as m6_sanitize_input  # guardrail: OpenAI GPT-4o (see m6_guardrail_llm)


def __getattr__(name: str):
    if name == "m4_filter_output":
        from .m4_output_validation import filter_output as m4_filter_output

        return m4_filter_output
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "m1_get_system_prompt",
    "m2_get_system_prompt",
    "m3_sanitize_input",
    "m4_filter_output",
    "m5_get_system_prompt",
    "m5_wrap_user_input",
    "m6_sanitize_input",
]
