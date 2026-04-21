"""
Analyze experiment logs: attack success rates (per category, prompt, LLM) and
mitigation effectiveness (individual, combined, selected combinations).
Reads JSONL files from experiment/logs/ and prints metrics.

Rates are always successes / (number of JSONL rows in that aggregate). The runner
may use any --runs value (e.g. one run per prompt per config ≈ 1200 lines per LLM
for a full suite); the analysis does not assume a fixed repeat count.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

# Default: sibling of analysis/
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "experiment"
LOGS_DIR = EXPERIMENT_DIR / "logs"


def record_model_key(r: dict) -> str:
    """Group key: primary model id (new logs), else legacy llm field (provider only)."""
    m = r.get("model")
    if m:
        return str(m)
    return str(r.get("llm", ""))


def _append_jsonl_file(records: list[dict], path: Path) -> None:
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue


def load_logs(log_dir: Path | None = None) -> list[dict]:
    """Load all JSONL records from log_dir."""
    log_dir = log_dir or LOGS_DIR
    if not log_dir.is_dir():
        return []
    records: list[dict] = []
    for f in sorted(log_dir.glob("*.jsonl")):
        _append_jsonl_file(records, f)
    return records


def load_logs_from_paths(paths: list[Path]) -> list[dict]:
    """Load JSONL records from explicit files (for per-run / per-model analysis)."""
    records: list[dict] = []
    for p in paths:
        if p.is_file():
            _append_jsonl_file(records, p)
    return records


def record_is_success(r: dict) -> bool:
    """True if this execution counts as a successful attack."""
    if "success" in r:
        return bool(r["success"])
    # Legacy JSONL: type1/type2 meant attack succeeded
    t = r.get("success_type", "none")
    return t in ("type1", "type2")


def records_for_config(records: list[dict], config: str) -> list[dict]:
    return [r for r in records if r.get("config") == config]


def record_sensitive_exfiltration(response: str | None) -> bool:
    """KB / canary substring leak (same strings as experiment runner)."""
    if not response or not response.strip():
        return False
    if response.strip().startswith("[ERROR]"):
        return False
    from chatbot.evaluation_canaries import SENSITIVE_STRINGS

    return any(s in response for s in SENSITIVE_STRINGS)


def record_system_prompt_disclosure(response: str | None) -> bool:
    """Trusted system-instruction text appeared in model output (runner-aligned)."""
    if not response or not response.strip():
        return False
    if response.strip().startswith("[ERROR]"):
        return False
    from chatbot.prompt_leak_detection import response_contains_system_prompt_leak

    return response_contains_system_prompt_leak(response)


def record_outcomes_from_response(r: dict) -> tuple[bool, bool] | None:
    """
    (sensitive_exfiltration, system_prompt_disclosure) from logged response.
    None if response field is missing (cannot rescore split outcomes).
    """
    if "response" not in r:
        return None
    resp = r.get("response")
    if not isinstance(resp, str):
        return None
    return (
        record_sensitive_exfiltration(resp),
        record_system_prompt_disclosure(resp),
    )


def success_rate(records: list[dict]) -> float:
    """Proportion of records where attack succeeded; denominator = len(records) (any repeat count)."""
    if not records:
        return 0.0
    ok = sum(1 for r in records if record_is_success(r))
    return ok / len(records)


def analyze(records: list[dict]) -> dict:
    """Compute all metrics from record list."""
    if not records:
        return {}

    by_config = defaultdict(list)
    by_model = defaultdict(list)
    by_category = defaultdict(list)
    by_prompt_id = defaultdict(list)
    by_model_config = defaultdict(list)

    for r in records:
        by_config[r.get("config", "")].append(r)
        mk = record_model_key(r)
        by_model[mk].append(r)
        by_category[r.get("category", "")].append(r)
        by_prompt_id[r.get("prompt_id", "")].append(r)
        by_model_config[(mk, r.get("config", ""))].append(r)

    m1_rates = {}
    for model_key in by_model:
        m1_recs = [x for x in by_model[model_key] if x.get("config") == "M1"]
        m1_rates[model_key] = success_rate(m1_recs)

    # Per-category success rate (over all configs and LLMs, or per LLM)
    category_rates = {}
    for cat in by_category:
        category_rates[cat] = success_rate(by_category[cat])

    # Per-prompt success rate
    prompt_rates = {pid: success_rate(recs) for pid, recs in by_prompt_id.items()}

    # Per primary model overall
    model_rates = {mk: success_rate(recs) for mk, recs in by_model.items()}

    # Per-config per model (for mitigation effectiveness)
    config_rates = defaultdict(dict)
    for (model_key, config), recs in by_model_config.items():
        config_rates[model_key][config] = success_rate(recs)

    # Individual mitigation effectiveness: M1 - Mx for each primary model
    mitigation_effectiveness = defaultdict(dict)
    for model_key in by_model:
        m1 = config_rates[model_key].get("M1", 0.0)
        for config in ["M2", "M3", "M4", "M5", "M6"]:
            mx = config_rates[model_key].get(config, 0.0)
            mitigation_effectiveness[model_key][config] = m1 - mx
        mitigation_effectiveness[model_key]["M2-M6"] = m1 - config_rates[model_key].get("M2-M6", 0.0)
        mitigation_effectiveness[model_key]["M3+M6"] = m1 - config_rates[model_key].get("M3+M6", 0.0)
        mitigation_effectiveness[model_key]["M2+M5"] = m1 - config_rates[model_key].get("M2+M5", 0.0)
        mitigation_effectiveness[model_key]["M2+M3+M4"] = m1 - config_rates[model_key].get("M2+M3+M4", 0.0)

    return {
        "n_records": len(records),
        "m1_success_rate_by_model": m1_rates,
        "success_rate_by_model": model_rates,
        "success_rate_by_category": category_rates,
        "success_rate_by_prompt_id": prompt_rates,
        "success_rate_by_model_and_config": dict(config_rates),
        "mitigation_effectiveness_by_model": dict(mitigation_effectiveness),
        # Legacy keys (older JSONL used "llm" for provider only)
        "m1_success_rate_by_llm": m1_rates,
        "success_rate_by_llm": model_rates,
        "success_rate_by_llm_and_config": dict(config_rates),
        "mitigation_effectiveness_by_llm": dict(mitigation_effectiveness),
    }


def print_report(metrics: dict) -> None:
    """Pretty-print pooled metrics (all loaded records, all models mixed) to stdout."""
    if not metrics:
        print("No records to analyze.")
        return
    print(f"Total executions: {metrics['n_records']}\n")
    print("=== M1 baseline success rate (per primary model) ===")
    for mk, rate in metrics.get("m1_success_rate_by_model", metrics.get("m1_success_rate_by_llm", {})).items():
        print(f"  {mk}: {rate:.2%}")
    print("\n=== Success rate by category ===")
    for cat, rate in sorted(metrics.get("success_rate_by_category", {}).items()):
        print(f"  {cat}: {rate:.2%}")
    print("\n=== Success rate by primary model ===")
    for mk, rate in metrics.get("success_rate_by_model", metrics.get("success_rate_by_llm", {})).items():
        print(f"  {mk}: {rate:.2%}")
    print("\n=== Mitigation effectiveness (M1 - Mx) by primary model ===")
    for mk, eff in metrics.get("mitigation_effectiveness_by_model", metrics.get("mitigation_effectiveness_by_llm", {})).items():
        print(f"  {mk}:")
        for config, e in sorted(eff.items()):
            print(f"    {config}: {e:+.2%}")


def group_records_by_model(records: list[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        g[record_model_key(r)].append(r)
    return dict(g)


def build_methodology_payload(records: list[dict]) -> dict:
    """
    Split by primary model and attach analyze() output in thesis-friendly shape.
    If all records share one model, there is one entry under "models".
    """
    by_model = group_records_by_model(records)
    payload: dict = {
        "n_records_total": len(records),
        "primary_models": sorted(by_model.keys()),
        "models": {},
    }
    if len(by_model) > 1:
        payload["note"] = (
            "Multiple primary models in this input; metrics are computed separately per model. "
            "Do not pool different runs/models when reporting thesis tables unless intentional."
        )

    for mk, sub in sorted(by_model.items()):
        m = analyze(sub)
        eff = m.get("mitigation_effectiveness_by_model", {}).get(mk, {})
        cfg_rates = m.get("success_rate_by_model_and_config", {}).get(mk, {})

        payload["models"][mk] = {
            "n_records": m["n_records"],
            "attack_success_rate_by_category": m.get("success_rate_by_category", {}),
            "attack_success_rate_by_prompt_id": m.get("success_rate_by_prompt_id", {}),
            "attack_success_rate_overall": m.get("success_rate_by_model", {}).get(mk),
            "m1_baseline_attack_success_rate": m.get("m1_success_rate_by_model", {}).get(mk),
            "attack_success_rate_by_config": cfg_rates,
            "mitigation_effectiveness": {
                "individual_M2_to_M6": {k: eff[k] for k in ["M2", "M3", "M4", "M5", "M6"] if k in eff},
                "combined_M2_M6": eff.get("M2-M6"),
                "selected_M3_plus_M6": eff.get("M3+M6"),
                "selected_M2_plus_M5": eff.get("M2+M5"),
                "selected_M2_M3_M4": eff.get("M2+M3+M4"),
            },
        }
    return payload


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.2%}"


def print_methodology_report(payload: dict) -> None:
    if not payload.get("models"):
        print("No records to analyze.")
        return
    print(f"Total executions: {payload['n_records_total']}")
    if payload.get("note"):
        print(f"Note: {payload['note']}\n")
    for mk, block in payload["models"].items():
        print(f"=== Primary model: {mk} (n={block['n_records']}) ===")
        print(f"M1 baseline attack success rate: {_fmt_pct(block.get('m1_baseline_attack_success_rate'))}")
        print(
            f"Overall attack success rate (all configs): {_fmt_pct(block.get('attack_success_rate_overall'))}"
        )
        print("\nAttack success rate by category:")
        for cat, rate in sorted(block["attack_success_rate_by_category"].items()):
            print(f"  {cat}: {_fmt_pct(rate)}")
        print("\nMitigation effectiveness (M1 - Mx); higher = more effective:")
        me = block["mitigation_effectiveness"]
        ind = me.get("individual_M2_to_M6") or {}
        for k in sorted(ind.keys()):
            v = ind[k]
            print(f"  {k}: {v:+.2%}" if v is not None else f"  {k}: n/a")
        for label, key in [
            ("Combined M2-M6", "combined_M2_M6"),
            ("Selected M3+M6", "selected_M3_plus_M6"),
            ("Selected M2+M5", "selected_M2_plus_M5"),
            ("Selected M2+M3+M4", "selected_M2_M3_M4"),
        ]:
            v = me.get(key)
            if v is not None:
                print(f"  {label}: {v:+.2%}")
        print("\nAttack success rate by config:")
        for cfg, rate in sorted(block["attack_success_rate_by_config"].items()):
            print(f"  {cfg}: {_fmt_pct(rate)}")
        print("\nPer-prompt attack success rate (prompt_id -> rate):")
        for pid, rate in sorted(block["attack_success_rate_by_prompt_id"].items()):
            print(f"  {pid}: {_fmt_pct(rate)}")
        print()


def load_records_for_analysis(
    *,
    files: list[Path],
    log_dir: Path | None,
    glob_pattern: str,
    default_logs_dir: Path,
) -> list[dict]:
    """Resolve JSONL paths: explicit files, optional second directory glob, else default dir."""
    paths: list[Path] = []
    paths.extend(files)
    if log_dir is not None and log_dir.is_dir():
        pat = glob_pattern if glob_pattern else "*.jsonl"
        paths.extend(sorted(log_dir.glob(pat)))
    if paths:
        return load_logs_from_paths(paths)
    return load_logs(default_logs_dir)


def build_full_export(records: list[dict]) -> dict:
    """Thesis-oriented payload plus pooled analyze() over all loaded rows."""
    payload = build_methodology_payload(records)
    payload["flat_pooled_all_loaded_records"] = analyze(records)
    return payload


def _pretty_category(name: str) -> str:
    return str(name).replace("_", " ").title()


def _pct_cell(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.2f}%"


def model_blocks_in_file_order(paths: list[Path]) -> list[tuple[str, dict]]:
    """
    One primary model block per log file, in the same order as ``paths``.
    Raises ValueError if a model id appears twice.
    """
    seen: set[str] = set()
    ordered: list[tuple[str, dict]] = []
    for path in paths:
        recs = load_logs_from_paths([path])
        payload = build_methodology_payload(recs)
        models = payload.get("models") or {}
        if not models:
            continue
        if len(models) != 1:
            raise ValueError(
                f"{path}: expected one primary model per file for table export, found {sorted(models)}"
            )
        mk, block = next(iter(models.items()))
        if mk in seen:
            raise ValueError(
                f"Duplicate primary model {mk!r}; use one JSONL file per model or a single combined log."
            )
        seen.add(mk)
        ordered.append((mk, block))
    return ordered


def render_thesis_methodology_markdown(ordered_models: list[tuple[str, dict]]) -> str:
    """
    Markdown tables aligned with thesis methodology: attack metrics + mitigation metrics.
    Percentages match CLI output (successes / executions in each aggregate).
    """
    if not ordered_models:
        return ""

    headers = [mk for mk, _ in ordered_models]
    lines: list[str] = [
        "# Experiment results (methodology tables)",
        "",
        "Generated from JSONL logs. Mapping to methodology:",
        "",
        "- **Table 1** — Attack success rate *per prompt injection category* (successes / all executions in that category; configs pooled as in the log).",
        "- **Table 2** — Attack success rate *per LLM* (M1 baseline and overall across configs).",
        "- **Per-prompt table / CSV** — Attack success rate *per prompt* (each `prompt_id`, including paraphrases as separate ids); denominator = all logged executions for that id across configurations (e.g. one repeat per prompt–config if you use `--runs 1`).",
        "- **Tables 3–5** — Mitigation effectiveness: individual M2–M6, combined M2–M6, selected stacks, and raw success rate per configuration (M1 − Mx where applicable).",
        "",
        "**M1 baseline** = M1 configuration only. **Mitigation effectiveness** = M1 rate − Mx rate (higher = more effective).",
        "",
        "## Attack metrics",
        "",
        "### Table 1 — Attack success rate by prompt injection category",
        "",
        "Proportion of successful attack executions within each category.",
        "",
    ]

    all_cats: set[str] = set()
    for _, block in ordered_models:
        all_cats.update((block.get("attack_success_rate_by_category") or {}).keys())
    cat_rows = sorted(all_cats)

    lines.append("| Category | " + " | ".join(headers) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in headers) + " |")
    for cat in cat_rows:
        cells = [_pct_cell((block.get("attack_success_rate_by_category") or {}).get(cat)) for _, block in ordered_models]
        lines.append(f"| {_pretty_category(cat)} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.extend(
        [
            "### Table 2 — Attack success rate per LLM (overall)",
            "",
            "Separate overall rates for each primary model: baseline (M1 only) and pooled across "
            "all mitigation configurations present in the log.",
            "",
            "| Model | M1 baseline (attack success) | Overall (all configs pooled) |",
            "| --- | --- | --- |",
        ]
    )
    for mk, block in ordered_models:
        m1 = block.get("m1_baseline_attack_success_rate")
        overall = block.get("attack_success_rate_overall")
        lines.append(f"| {mk} | {_pct_cell(m1)} | {_pct_cell(overall)} |")
    lines.append("")

    lines.extend(
        [
            "## Security mitigation metrics",
            "",
            "### Table 3 — Individual control effectiveness (M1 − Mx)",
            "",
            "| Model | M2 | M3 | M4 | M5 | M6 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for mk, block in ordered_models:
        ind = (block.get("mitigation_effectiveness") or {}).get("individual_M2_to_M6") or {}
        row = [mk] + [_pct_cell(ind.get(f"M{i}")) for i in range(2, 7)]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.extend(
        [
            "### Table 4 — Combined and selected combination effectiveness (M1 − Mcombined)",
            "",
            "| Model | M2–M6 | M3+M6 | M2+M5 | M2+M3+M4 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    me_keys = [
        ("combined_M2_M6", "M2–M6"),
        ("selected_M3_plus_M6", "M3+M6"),
        ("selected_M2_plus_M5", "M2+M5"),
        ("selected_M2_M3_M4", "M2+M3+M4"),
    ]
    for mk, block in ordered_models:
        me = block.get("mitigation_effectiveness") or {}
        cells = [_pct_cell(me.get(k)) for k, _ in me_keys]
        lines.append("| " + mk + " | " + " | ".join(cells) + " |")
    lines.append("")

    lines.extend(
        [
            "### Table 5 — Attack success rate by mitigation configuration",
            "",
            "Proportion of successful attacks under each tested configuration.",
            "",
        ]
    )
    all_cfgs: set[str] = set()
    for _, block in ordered_models:
        all_cfgs.update((block.get("attack_success_rate_by_config") or {}).keys())
    cfg_rows = sorted(all_cfgs)
    lines.append("| Configuration | " + " | ".join(headers) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in headers) + " |")
    for cfg in cfg_rows:
        cells = [_pct_cell((block.get("attack_success_rate_by_config") or {}).get(cfg)) for _, block in ordered_models]
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.extend(
        [
            "## Per-prompt attack success rates",
            "",
            "Proportion of successful executions per `prompt_id` (original and paraphrases pooled across "
            "all configurations in the log). See the CSV export for the same data in long form.",
            "",
        ]
    )

    all_pids: set[str] = set()
    for _, block in ordered_models:
        all_pids.update((block.get("attack_success_rate_by_prompt_id") or {}).keys())
    pid_rows = sorted(all_pids)
    lines.append("| prompt_id | " + " | ".join(headers) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in headers) + " |")
    for pid in pid_rows:
        cells = [_pct_cell((block.get("attack_success_rate_by_prompt_id") or {}).get(pid)) for _, block in ordered_models]
        lines.append(f"| `{pid}` | " + " | ".join(cells) + " |")
    lines.append("")

    return "\n".join(lines)


def render_prompt_rates_csv(ordered_models: list[tuple[str, dict]]) -> str:
    """Long-form CSV: model, prompt_id, attack_success_rate (0–1 float for spreadsheets)."""
    import csv
    from io import StringIO

    buf = StringIO()
    w = csv.writer(buf)
    w.writerow(["model", "prompt_id", "attack_success_rate", "attack_success_rate_pct"])
    for mk, block in ordered_models:
        for pid, rate in sorted((block.get("attack_success_rate_by_prompt_id") or {}).items()):
            r = rate if rate is not None else ""
            pct_s = f"{rate * 100:.2f}" if isinstance(rate, (int, float)) else ""
            w.writerow([mk, pid, r, pct_s])
    return buf.getvalue()


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Analyze experiment JSONL logs (thesis metrics: per model, configs, prompts)."
    )
    p.add_argument(
        "--files",
        type=Path,
        nargs="*",
        default=[],
        help="Explicit JSONL file(s). If omitted (and no --log-dir), loads all *.jsonl under --logs.",
    )
    p.add_argument(
        "--logs",
        type=Path,
        default=LOGS_DIR,
        help="Directory of *.jsonl when no --files / --log-dir (default: experiment/logs)",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Extra directory of logs (combined with --files); use --glob to filter names",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="",
        help='Pattern under --log-dir (default: "*.jsonl")',
    )
    p.add_argument("--json", action="store_true", help="Print full export JSON (methodology + pooled flat)")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write same JSON as --json to this path",
    )
    p.add_argument(
        "--compact",
        action="store_true",
        help="Print short pooled summary only (no per-model thesis block, no per-prompt list)",
    )
    p.add_argument(
        "--markdown-tables",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write methodology-aligned Markdown tables; requires --files (one JSONL per primary model).",
    )
    p.add_argument(
        "--csv-prompt-rates",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write per-prompt rates as CSV (long form); use with --files (same as --markdown-tables).",
    )
    args = p.parse_args()

    export_paths = list(args.files)
    if args.markdown_tables or args.csv_prompt_rates:
        if not export_paths:
            p.error("--markdown-tables / --csv-prompt-rates require explicit --files (one JSONL per model).")
        try:
            ordered = model_blocks_in_file_order(export_paths)
        except ValueError as e:
            p.error(str(e))
        if args.markdown_tables:
            args.markdown_tables.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_tables.write_text(
                render_thesis_methodology_markdown(ordered), encoding="utf-8"
            )
        if args.csv_prompt_rates:
            args.csv_prompt_rates.parent.mkdir(parents=True, exist_ok=True)
            args.csv_prompt_rates.write_text(
                render_prompt_rates_csv(ordered), encoding="utf-8"
            )
        parts = []
        if args.markdown_tables:
            parts.append(str(args.markdown_tables))
        if args.csv_prompt_rates:
            parts.append(str(args.csv_prompt_rates))
        print(f"Wrote {' and '.join(parts)} ({len(ordered)} model(s)).")
        return

    records = load_records_for_analysis(
        files=list(args.files),
        log_dir=args.log_dir,
        glob_pattern=args.glob,
        default_logs_dir=args.logs,
    )

    if args.compact:
        metrics = analyze(records)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as fp:
                json.dump(metrics, fp, indent=2)
        if args.json:
            print(json.dumps(metrics, indent=2))
        if not args.json and not args.out:
            print_report(metrics)
        return

    export = build_full_export(records)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(export, fp, indent=2)
    if args.json:
        print(json.dumps(export, indent=2))
    elif not args.out:
        print_methodology_report(
            {k: v for k, v in export.items() if k != "flat_pooled_all_loaded_records"}
        )


if __name__ == "__main__":
    main()
