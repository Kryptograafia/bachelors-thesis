"""
Build thesis-style Markdown tables (same layout as metrics_out/thesis_repro/tables.md).

Metrics align with Thesis.txt §3.4:
  - Attack success = proportion of executions with success=true (JSONL rows).
  - Mitigation effectiveness (Table 4) = (M1 − Mx) / M1 × 100 (relative %), per §3.4.3.
  - Resilience (Table 5) = 100% − overall attack success rate, per §3.4.4.

Extra statistics (optional appendix in Markdown + optional CSV):
  - Mean and sample standard deviation of the six per-category attack success rates (same categories as Table 2).

Inputs: one JSONL file per primary model (separate runs), or a directory of such files.

Example:
  python experiment/compute_thesis_tables.py \\
    --input-dir ../Reviewed_logs \\
    --out experiment/metrics_out/thesis_repro/tables.md
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analyze_results import (  # noqa: E402
    load_logs_from_paths,
    record_is_success,
    record_model_key,
    success_rate,
)

# Category column order and abbreviations (matches thesis tables / attack suite taxonomy).
CATEGORY_ORDER: list[tuple[str, str]] = [
    ("explicit_data_exfiltration", "EX"),
    ("prompt_in_prompt", "PP"),
    ("instruction_override", "IO"),
    ("roleplay_manipulation", "RP"),
    ("obfuscated_instruction", "OB"),
    ("syntactic_separation", "SS"),
]

# Default column order for Table 2 / Table 5 (API models first, then locals); skip missing.
DEFAULT_MODEL_COLUMN_ORDER: list[str] = [
    "gpt-4o-mini",
    "gemini-2.5-flash-lite",
    "mistral",
    "llama3.1:8b",
    "gemma2:9b",
    "qwen3:8b",
]

# Tables 3–4: same default set as Table 2 (GPT, Gemini, four locals); override via --table34-models.
DEFAULT_TABLE34_MODELS: list[str] = list(DEFAULT_MODEL_COLUMN_ORDER)

TABLE4_CONFIG_ORDER: list[str] = [
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M2-M6",
    "M3+M6",
    "M2+M5",
    "M2+M3+M4",
]


def _pick_models_present(order: list[str], present: set[str]) -> list[str]:
    out = [m for m in order if m in present]
    extra = sorted(present - set(out))
    return out + extra


def _fmt_pct_cell(x: float | None, decimals: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.{decimals}f}%"


def _fmt_signed_pct_points(x: float | None, decimals: int = 1) -> str:
    """Display relative mitigation as signed percentage points, e.g. +16.7%."""
    if x is None:
        return "—"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x * 100:.{decimals}f}%"


def mitigation_effectiveness_relative(m1: float, mx: float) -> float | None:
    """(M1 - Mx) / M1 as a fraction; None if M1 == 0."""
    if m1 <= 0:
        return None
    return (m1 - mx) / m1


def records_by_model(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in records:
        mk = record_model_key(r)
        out.setdefault(mk, []).append(r)
    return out


def table2_category_matrix(
    by_model: dict[str, list[dict]], model_cols: list[str]
) -> tuple[list[tuple[str, list[float | None], float, int]], list[str]]:
    """
    Rows: category abbreviations.
    Returns list of (abbrev, [rates per model], average_fraction, rank).
    """
    rows_out: list[tuple[str, list[float | None], float, int]] = []
    full_names = [fn for fn, ab in CATEGORY_ORDER]

    for full, abbrev in CATEGORY_ORDER:
        rates: list[float | None] = []
        for mk in model_cols:
            sub = [r for r in by_model.get(mk, []) if r.get("category") == full]
            rates.append(success_rate(sub) if sub else None)
        valid = [x for x in rates if x is not None]
        avg = sum(valid) / len(valid) if valid else 0.0
        rows_out.append((abbrev, rates, avg, 0))

    # Rank categories by average attack success (desc): 1 = highest threat
    sorted_idx = sorted(range(len(rows_out)), key=lambda i: rows_out[i][2], reverse=True)
    rank_by_idx = {sorted_idx[r]: r + 1 for r in range(len(sorted_idx))}

    ranked: list[tuple[str, list[float | None], float, int]] = []
    for i, row in enumerate(rows_out):
        ranked.append((row[0], row[1], row[2], rank_by_idx[i]))
    return ranked, full_names


def table3_prompt_ranking(
    by_model: dict[str, list[dict]], model_cols: list[str]
) -> list[tuple[str, list[float | None], float]]:
    """Per prompt_id: success rate per model (all configs), then average; sort by avg desc."""
    prompt_ids: set[str] = set()
    for mk in model_cols:
        for r in by_model.get(mk, []):
            pid = r.get("prompt_id")
            if pid:
                prompt_ids.add(str(pid))

    rows: list[tuple[str, list[float | None], float]] = []
    for pid in prompt_ids:
        per_model: list[float | None] = []
        for mk in model_cols:
            sub = [
                r
                for r in by_model.get(mk, [])
                if str(r.get("prompt_id")) == pid
            ]
            per_model.append(success_rate(sub) if sub else None)
        vals = [x for x in per_model if x is not None]
        avg = sum(vals) / len(vals) if vals else 0.0
        rows.append((pid, per_model, avg))

    rows.sort(key=lambda x: x[2], reverse=True)
    return rows


def table4_mitigation_rows(
    by_model: dict[str, list[dict]], model_cols: list[str]
) -> list[tuple[str, list[float | None], float]]:
    """Configs × relative mitigation % per model + row average."""
    rows: list[tuple[str, list[float | None], float]] = []
    for cfg in TABLE4_CONFIG_ORDER:
        cells: list[float | None] = []
        for mk in model_cols:
            m1_recs = [r for r in by_model.get(mk, []) if r.get("config") == "M1"]
            mx_recs = [r for r in by_model.get(mk, []) if r.get("config") == cfg]
            m1 = success_rate(m1_recs) if m1_recs else 0.0
            mx = success_rate(mx_recs) if mx_recs else 0.0
            eff = mitigation_effectiveness_relative(m1, mx)
            cells.append(eff)
        valid = [c for c in cells if c is not None]
        row_avg = sum(valid) / len(valid) if valid else None
        rows.append((cfg, cells, row_avg))
    return rows


def table5_resilience(by_model: dict[str, list[dict]], model_cols: list[str]) -> list[tuple[str, float, int, float, int]]:
    """
    Per model: overall ASR, blocked count, resilience, n trials.
    Sorted by resilience descending (best first).
    """
    stats: list[tuple[str, float, int, float, int]] = []
    for mk in model_cols:
        recs = by_model.get(mk, [])
        n = len(recs)
        if n == 0:
            continue
        sr = success_rate(recs)
        successes = sum(1 for r in recs if record_is_success(r))
        blocked = n - successes
        resilience = 1.0 - sr
        stats.append((mk, sr, blocked, resilience, n))
    stats.sort(key=lambda x: x[3], reverse=True)
    return stats


def render_markdown(
    model_cols_t2: list[str],
    model_cols_t34: list[str],
    t2_rows: list[tuple[str, list[float | None], float, int]],
    t3_rows: list[tuple[str, list[float | None], float]],
    t4_rows: list[tuple[str, list[float | None], float]],
    t5_rows: list[tuple[str, float, int, float, int]],
) -> str:
    lines: list[str] = []

    # Table 2
    lines.append("## Table 2 (computed)")
    lines.append("")
    hdr = "| Category | " + " | ".join(model_cols_t2) + " | Average | Rank |"
    sep = "| --- | " + " | ".join(["---"] * len(model_cols_t2)) + " | --- | --- |"
    lines.append(hdr)
    lines.append(sep)
    for abbrev, rates, avg, rank in t2_rows:
        cells = [_fmt_pct_cell(x) for x in rates]
        lines.append(
            f"| {abbrev} | "
            + " | ".join(cells)
            + f" | {_fmt_pct_cell(avg)} | {rank} |"
        )
    lines.append("")

    # Table 3
    lines.append("## Table 3 (computed)")
    lines.append("")
    lines.append("| Prompt ID | " + " | ".join(model_cols_t34) + " | Average |")
    lines.append("| --- | " + " | ".join(["---"] * len(model_cols_t34)) + " | --- |")
    for pid, rates, avg in t3_rows:
        cells = [_fmt_pct_cell(x) for x in rates]
        lines.append(f"| {pid} | " + " | ".join(cells) + f" | {_fmt_pct_cell(avg)} |")
    lines.append("")

    # Table 4
    lines.append("## Table 4 (computed)")
    lines.append("")
    lines.append("| Config | " + " | ".join(model_cols_t34) + " | Average |")
    lines.append("| --- | " + " | ".join(["---"] * len(model_cols_t34)) + " | --- |")
    for cfg, cells, row_avg in t4_rows:
        pct_cells = [_fmt_signed_pct_points(c) for c in cells]
        avg_cell = _fmt_signed_pct_points(row_avg) if row_avg is not None else "—"
        lines.append(f"| {cfg} | " + " | ".join(pct_cells) + f" | {avg_cell} |")
    lines.append("")

    # Table 5
    lines.append("## Table 5 (computed)")
    lines.append("")
    lines.append("| Model | Attack success rate | Attacks blocked | Resilience | Total trials |")
    lines.append("| --- | --- | --- | --- | --- |")
    for mk, sr, blocked, res, n in t5_rows:
        lines.append(
            f"| {mk} | {_fmt_pct_cell(sr)} | {blocked} | {_fmt_pct_cell(res)} | {n} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_stats_appendix(
    by_model: dict[str, list[dict]], model_cols: list[str]
) -> str:
    """Mean and sample SD of the six per-category attack success rates (Table 2 categories)."""
    lines: list[str] = [
        "## Summary statistics (computed)",
        "",
        "**Mean** is the arithmetic mean of the six category-level attack success rates (EX, PP, IO, RP, OB, SS). "
        "**SD** is the sample standard deviation across those six rates (spread across attack categories).",
        "",
        "| Model | Mean | SD |",
        "| --- | ---: | ---: |",
    ]
    for mk in model_cols:
        recs = by_model.get(mk, [])
        if not recs:
            continue
        cat_vals: list[float] = []
        for full, _ in CATEGORY_ORDER:
            sub = [r for r in recs if r.get("category") == full]
            cat_vals.append(success_rate(sub) if sub else 0.0)
        c_mean = statistics.mean(cat_vals) if cat_vals else 0.0
        c_sd = statistics.stdev(cat_vals) if len(cat_vals) > 1 else 0.0
        lines.append(f"| {mk} | {_fmt_pct_cell(c_mean)} | {_fmt_pct_cell(c_sd)} |")
    lines.append("")
    return "\n".join(lines)


def write_stats_csv(
    path: Path, by_model: dict[str, list[dict]], model_cols: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["model", "category_rates_mean", "category_rates_sd"])
        for mk in model_cols:
            recs = by_model.get(mk, [])
            if not recs:
                continue
            cat_vals = []
            for full, _ in CATEGORY_ORDER:
                sub = [r for r in recs if r.get("category") == full]
                cat_vals.append(success_rate(sub) if sub else 0.0)
            c_mean = statistics.mean(cat_vals) if cat_vals else 0.0
            c_sd = statistics.stdev(cat_vals) if len(cat_vals) > 1 else 0.0
            w.writerow([mk, f"{c_mean:.6f}", f"{c_sd:.6f}"])


def main() -> None:
    p = argparse.ArgumentParser(description="Generate thesis-style Markdown tables from JSONL logs.")
    p.add_argument(
        "--files",
        type=Path,
        nargs="*",
        default=[],
        help="Explicit JSONL paths (one primary model per file).",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing *.jsonl (combined with --files if both set).",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="*.jsonl",
        help='Glob under --input-dir (default: "*.jsonl").',
    )
    p.add_argument(
        "--model-order",
        type=str,
        default=None,
        help="Comma-separated model ids for Table 2 / Table 5 column order (optional).",
    )
    p.add_argument(
        "--table34-models",
        type=str,
        default=None,
        help=(
            "Comma-separated model ids for Tables 3–4 (default: same as Table 2 — "
            "gpt-4o-mini, gemini-2.5-flash-lite, mistral, llama3.1:8b, "
            "gemma2:9b, qwen3:8b; missing ids skipped)."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write Markdown here (default: stdout only).",
    )
    p.add_argument(
        "--no-stats-appendix",
        action="store_true",
        help="Omit the Summary statistics section (mean / SD across six categories).",
    )
    p.add_argument(
        "--stats-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write per-model category-rate mean and SD as CSV (same metrics as appendix).",
    )
    args = p.parse_args()

    paths: list[Path] = list(args.files)
    if args.input_dir is not None:
        d = Path(args.input_dir)
        if d.is_dir():
            paths.extend(sorted(d.glob(args.glob)))

    if not paths:
        p.error("Provide --files and/or --input-dir with matching JSONL.")

    records = load_logs_from_paths(paths)
    if not records:
        p.error("No records loaded.")

    by_model = records_by_model(records)
    present = set(by_model.keys())

    if args.model_order:
        order_t2 = [x.strip() for x in args.model_order.split(",") if x.strip()]
        model_cols_t2 = _pick_models_present(order_t2, present)
    else:
        model_cols_t2 = _pick_models_present(DEFAULT_MODEL_COLUMN_ORDER, present)

    if args.table34_models:
        t34 = [x.strip() for x in args.table34_models.split(",") if x.strip()]
        model_cols_t34 = [m for m in t34 if m in present]
        missing = set(t34) - present
        if missing:
            print(f"Warning: table34-models not in data (skipped): {sorted(missing)}", file=sys.stderr)
    else:
        model_cols_t34 = [m for m in DEFAULT_TABLE34_MODELS if m in present]

    if not model_cols_t34:
        p.error("No models left for Tables 3–4; check --table34-models / logs.")

    t2_rows, _ = table2_category_matrix(by_model, model_cols_t2)
    t3_rows = table3_prompt_ranking(by_model, model_cols_t34)
    t4_rows = table4_mitigation_rows(by_model, model_cols_t34)
    t5_rows = table5_resilience(by_model, model_cols_t2)

    md = render_markdown(
        model_cols_t2,
        model_cols_t34,
        t2_rows,
        t3_rows,
        t4_rows,
        t5_rows,
    )
    if not args.no_stats_appendix:
        md += "\n" + render_stats_appendix(by_model, model_cols_t2)
    if args.stats_csv:
        write_stats_csv(args.stats_csv, by_model, model_cols_t2)
        print(f"Wrote {args.stats_csv}", file=sys.stderr)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
