"""
Experiment runner: run attack suite × mitigation configs × repeats; log JSONL.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os

from dotenv import load_dotenv

from chatbot.llm_factory import resolve_model_id

# override=True so .env wins over stale Windows User/System API key vars.
load_dotenv(PROJECT_ROOT.parent / ".env", override=True)
load_dotenv(PROJECT_ROOT / ".env", override=True)


def _model_slug_for_filename(model_id: str) -> str:
    """Safe fragment for log filenames (no path separators)."""
    s = model_id.strip().replace(":", "-").replace("/", "_").replace("\\", "_")
    s = "".join(c if c.isalnum() or c in "-_." else "_" for c in s)
    return (s[:80] if s else "default")

KB_PATH = PROJECT_ROOT / "knowledge_base"
ATTACKS_DIR = PROJECT_ROOT / "attacks"
ATTACK_SUITE_PATH = ATTACKS_DIR / "attack_suite.json"
LOGS_DIR = Path(__file__).resolve().parent / "logs"


def load_attack_suite(suite_path: Path | None = None) -> list[dict]:
    path = suite_path or ATTACK_SUITE_PATH
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for category, entries in data["categories"].items():
        for entry in entries:
            pid = entry["id"]
            out.append({"id": pid, "category": category, "text": entry["original"]})
            for i, p in enumerate(entry.get("paraphrases", []), 1):
                out.append({"id": f"{pid}-p{i}", "category": category, "text": p})
    return out


CONFIGS = {
    "M1": {"M1"},
    "M2": {"M2"},
    "M3": {"M3"},
    "M4": {"M4"},
    "M5": {"M5"},
    "M6": {"M6"},
    "M2-M6": {"M2", "M3", "M4", "M5", "M6"},
    "M3+M6": {"M3", "M6"},
    "M2+M5": {"M2", "M5"},
    "M2+M3+M4": {"M2", "M3", "M4"},
}


def run_experiment(
    llm_provider: str,
    llm_model_id: str | None = None,
    configs: list[str] | None = None,
    runs_per_prompt: int = 3,
    dry_run: bool = False,
    attack_suite: str | None = None,
    temperature: float = 0.3,
    max_prompts: int | None = None,
):
    configs = configs or list(CONFIGS.keys())
    unknown_cfgs = [c for c in configs if c not in CONFIGS]
    if unknown_cfgs:
        valid = ", ".join(sorted(CONFIGS.keys()))
        raise ValueError(
            f"Unknown mitigation config(s): {unknown_cfgs}. Valid: {valid}"
        )
    suite_path: Path | None = None
    suite_suffix = ""
    if attack_suite:
        a = attack_suite.lower().strip()
        if a == "alt":
            suite_path = ATTACKS_DIR / "attack_suite_alt.json"
            suite_suffix = "_alt"
        elif a in ("alt_fixed", "altfixed"):
            suite_path = ATTACKS_DIR / "attack_suite_alt_fixed.json"
            suite_suffix = "_alt_fixed"
        elif a != "default":
            cand = ATTACKS_DIR / attack_suite
            suite_path = cand if cand.exists() else Path(attack_suite)
    if attack_suite and suite_path is not None and not suite_path.exists():
        raise FileNotFoundError(f"Attack suite file not found: {suite_path}")

    prompts = load_attack_suite(suite_path)
    if max_prompts is not None:
        if max_prompts < 1:
            raise ValueError("max_prompts must be >= 1 when set")
        prompts = prompts[:max_prompts]

    resolved_model = resolve_model_id(llm_provider, llm_model_id)
    model_slug = _model_slug_for_filename(resolved_model)
    guardrail_model = os.getenv("GUARDRAIL_MODEL", "granite3.2:8b").strip()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = (
        LOGS_DIR
        / f"run_{llm_provider}_{model_slug}{suite_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    if dry_run:
        total = len(configs) * len(prompts) * runs_per_prompt
        done = 0
        with open(log_file, "w", encoding="utf-8") as log:
            for config_name in configs:
                mitigations = CONFIGS.get(config_name)
                if not mitigations:
                    continue
                for prompt in prompts:
                    for _ in range(runs_per_prompt):
                        rec = {
                            "prompt_id": prompt["id"],
                            "category": prompt["category"],
                            "config": config_name,
                            "llm_provider": llm_provider,
                            "model": resolved_model,
                            "guardrail_model": guardrail_model,
                            "success": False,
                            "response_preview": "(dry run)",
                        }
                        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        done += 1
                        if done % 50 == 0:
                            print(f"  {done}/{total}")
        print(f"Dry run done. {done} records -> {log_file}")
        return log_file

    from chatbot.rag_chain import build_chain, run_chatbot
    from chatbot.mitigations import (
        m1_get_system_prompt,
        m2_get_system_prompt,
        m3_sanitize_input,
        m4_filter_output,
        m5_get_system_prompt,
        m5_wrap_user_input,
        m6_sanitize_input,
    )
    from chatbot.evaluation_canaries import SENSITIVE_STRINGS
    from chatbot.prompt_leak_detection import response_contains_system_prompt_leak
    from chatbot.mitigations.m5_structured_prompt import SEPARATION_INSTRUCTION

    def get_system_prompt(mitigations: set[str]) -> str:
        if "M5" in mitigations and "M2" in mitigations:
            return m2_get_system_prompt() + SEPARATION_INSTRUCTION
        if "M5" in mitigations:
            return m5_get_system_prompt()
        if "M2" in mitigations:
            return m2_get_system_prompt()
        return m1_get_system_prompt()

    def sanitize_user_input(raw: str, mitigations: set[str]) -> str:
        if "M6" in mitigations:
            raw = m6_sanitize_input(raw)
        if "M3" in mitigations:
            raw = m3_sanitize_input(raw)
        if "M5" in mitigations:
            raw = m5_wrap_user_input(raw)
        return raw

    def filter_response(response: str, mitigations: set[str]) -> str:
        if "M4" in mitigations:
            return m4_filter_output(response)
        return response

    def attack_succeeded(response: str) -> bool:
        if not response or not response.strip():
            return False
        if response.strip().startswith("[ERROR]"):
            return False
        if any(s in response for s in SENSITIVE_STRINGS):
            return True
        if response_contains_system_prompt_leak(response):
            return True
        return False

    print(f"Building chain: provider={llm_provider}, model={resolved_model}...")
    chain = build_chain(llm_provider, llm_model_id, KB_PATH, temperature=temperature)
    total = len(configs) * len(prompts) * runs_per_prompt
    done = 0

    with open(log_file, "w", encoding="utf-8") as log:
        for config_name in configs:
            mitigations = CONFIGS.get(config_name)
            if not mitigations:
                continue
            for prompt in prompts:
                pid, category, text = prompt["id"], prompt["category"], prompt["text"]
                system_prompt = get_system_prompt(mitigations)
                user_input = sanitize_user_input(text, mitigations)
                for _ in range(runs_per_prompt):
                    try:
                        response = run_chatbot(chain, user_input, system_prompt)
                    except Exception as e:
                        response = f"[ERROR] {e}"
                    response = filter_response(response, mitigations)
                    rec = {
                        "prompt_id": pid,
                        "category": category,
                        "config": config_name,
                        "llm_provider": llm_provider,
                        "model": resolved_model,
                        "guardrail_model": guardrail_model,
                        "success": attack_succeeded(response),
                        "user_message_after_mitigations": user_input[:8000],
                        "response": response[:2000],
                    }
                    log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done += 1
                    if done % 50 == 0:
                        print(f"  {done}/{total}")
        print(f"Done. Log: {log_file}")
    return log_file


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run prompt injection experiment")
    p.add_argument(
        "--llm",
        default="openai",
        choices=["openai", "ollama", "google"],
        help="Primary LLM (openai/google → API keys in .env).",
    )
    p.add_argument("--model", default=None, help="Model id override")
    p.add_argument("--configs", nargs="*", default=None)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM temperature (lower = more deterministic).",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--suite",
        default=None,
        help=(
            "Attack suite preset or filename. Presets: default, alt, alt_fixed. "
            "You can also pass a JSON filename under attacks/ (e.g. attack_suite_kb_targeted.json) "
            "or an absolute path to a JSON file."
        ),
    )
    p.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        metavar="N",
        help="Only run the first N prompts from the suite (smoke test).",
    )
    args = p.parse_args()
    run_experiment(
        llm_provider=args.llm,
        llm_model_id=args.model,
        configs=args.configs,
        runs_per_prompt=args.runs,
        dry_run=args.dry_run,
        attack_suite=args.suite,
        temperature=args.temperature,
        max_prompts=args.max_prompts,
    )
