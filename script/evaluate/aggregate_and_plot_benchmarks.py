#!/usr/bin/env python3
"""
Read per-run eval folders (each with ``combined_results.csv`` or ``results_*.csv``),
write ``benchmark_summary.csv``, and plot accuracy vs throughput (+ optional LLM-token share).

Typical layout (produced by ``run_aime_benchmark_suite.py``)::

    suite_dir/
      slm/combined_results.csv
      llm/combined_results.csv
      r2r_neural/combined_results.csv
      r2r_entropy_0_45/combined_results.csv
      ...

Usage::

    python script/evaluate/aggregate_and_plot_benchmarks.py --suite-dir output/eval/method_suite_20250323_120000

Hybrid runs that only have ``temp_csv/*.csv`` (no ``combined_results.csv`` yet) are merged on the fly.
Use the same Python env as training eval (e.g. conda ``r2r``) so ``matplotlib`` is available for PNGs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def _r2r_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _combine_from_temp_csv(suite_subdir: Path) -> Path | None:
    """Merge per-problem CSVs under temp_csv/ (hybrid eval layout) into combined_results.csv."""
    temp_csv = suite_subdir / "temp_csv"
    if not temp_csv.is_dir():
        return None
    files = sorted(temp_csv.glob("*.csv"))
    if not files:
        return None
    # Skip huge text columns (e.g. full_output) — only metrics columns needed for aggregation.
    _want = (
        "problem_id",
        "correct_answer",
        "has_extracted_answer",
        "predicted_answer",
        "is_correct",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "run_time",
        "speed_tokens_per_second",
        "quick_model_percentage",
        "reference_model_percentage",
        "model_agreement_percentage",
        "quick_source_agreement_percentage",
        "total_params_billions",
        "avg_params_billions",
    )

    def _read_one(f: Path) -> pd.DataFrame | None:
        try:
            header = pd.read_csv(f, nrows=0).columns.tolist()
            usecols = [c for c in _want if c in header]
            if "problem_id" not in usecols:
                return pd.read_csv(f)
            return pd.read_csv(f, usecols=usecols)
        except (OSError, pd.errors.ParserError, ValueError):
            return None

    dfs: list[pd.DataFrame] = []
    for f in files:
        one = _read_one(f)
        if one is not None and not one.empty:
            dfs.append(one)
    if not dfs:
        return None
    combined_df = pd.concat(dfs, ignore_index=True)
    if "problem_id" in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=["problem_id"], keep="last")
    out = suite_subdir / "combined_results.csv"
    combined_df.to_csv(out, index=False)
    return out


def _has_mergeable_artifacts(suite_subdir: Path) -> bool:
    """True if temp_csv/*.csv or results_*.csv exist (otherwise skip heavy hf_dataset_sglang import)."""
    tc = suite_subdir / "temp_csv"
    if tc.is_dir() and any(tc.glob("*.csv")):
        return True
    return any(suite_subdir.glob("results_*.csv"))


def ensure_combined(suite_subdir: Path) -> Path | None:
    """Return path to combined_results.csv, generating it via combine_results if needed."""
    combined = suite_subdir / "combined_results.csv"
    if combined.is_file():
        return combined

    merged = _combine_from_temp_csv(suite_subdir)
    if merged is not None:
        return merged

    if not _has_mergeable_artifacts(suite_subdir):
        return None

    # Import only when needed (pulls torch/sglang stack)
    sys.path.insert(0, str(_r2r_root() / "script"))
    from evaluate.hf_dataset_sglang import combine_results  # type: ignore

    stats = combine_results(str(suite_subdir))
    if not stats and not combined.is_file():
        return None
    return combined if combined.is_file() else None


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        return s.map(lambda x: str(x).lower() in ("true", "1", "yes"))
    return s.astype(bool)


def metrics_from_csv(combined: Path) -> dict:
    df = pd.read_csv(combined)
    row: dict[str, float | str | None] = {"n_rows": len(df)}

    if "is_correct" in df.columns:
        row["accuracy"] = float(_bool_series(df["is_correct"]).mean())
    else:
        row["accuracy"] = None

    if "output_tokens" in df.columns and "run_time" in df.columns:
        m = df["output_tokens"].notna() & df["run_time"].notna() & (df["run_time"] > 0)
        if m.any():
            row["throughput_tokens_per_s"] = float(df.loc[m, "output_tokens"].sum() / df.loc[m, "run_time"].sum())
        else:
            row["throughput_tokens_per_s"] = None
    else:
        row["throughput_tokens_per_s"] = None

    if "run_time" in df.columns and df["run_time"].notna().any():
        rt = df.loc[df["run_time"].notna() & (df["run_time"] > 0), "run_time"]
        if len(rt):
            row["avg_wall_time_s_per_problem"] = float(rt.mean())
            row["total_wall_time_s"] = float(rt.sum())
        else:
            row["avg_wall_time_s_per_problem"] = None
            row["total_wall_time_s"] = None
    else:
        row["avg_wall_time_s_per_problem"] = None
        row["total_wall_time_s"] = None

    if "reference_model_percentage" in df.columns:
        row["avg_reference_token_pct"] = float(df["reference_model_percentage"].mean())
    else:
        row["avg_reference_token_pct"] = None

    return row


def read_run_label(suite_subdir: Path, run_id: str) -> str:
    args_path = suite_subdir / "args.json"
    if args_path.is_file():
        try:
            with open(args_path) as f:
                data = json.load(f)
            router = (data.get("model_config") or {}).get("router") or {}
            strat = data.get("switching_strategy") or router.get("switching_strategy")
            th = data.get("threshold")
            if strat == "entropy_variance":
                eth = router.get("entropy_threshold")
                var = router.get("variance_threshold")
                return f"entropy+var τ={eth}, σ²={var}"
            if strat == "entropy_lookahead":
                eth = router.get("entropy_threshold")
                n = router.get("lookahead_steps")
                st = router.get("score_threshold")
                return f"entropy+LA τ={eth}, N={n}, S>{st}"
            if strat == "entropy" and th is not None:
                return f"entropy τ={th}"
            if data.get("use_hybrid"):
                return strat or "R2R neural"
            if data.get("use_model") == "quick":
                return "SLM"
            if data.get("use_model") == "reference":
                return "LLM"
        except (json.JSONDecodeError, OSError):
            pass
    if run_id.startswith("r2r_entropy_"):
        return f"entropy ({run_id.replace('r2r_entropy_', '').replace('_', '.')})"
    if run_id == "r2r_neural":
        return "R2R neural"
    if run_id == "slm":
        return "SLM"
    if run_id == "llm":
        return "LLM"
    m = re.search(r"entropy_([0-9.]+)_variance", run_id)
    if m:
        return f"entropy+var τ={m.group(1)}"
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=str, required=True, help="Directory containing per-run subfolders")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write summary CSV and PNGs (default: same as --suite-dir)",
    )
    parser.add_argument(
        "--title-tag",
        type=str,
        default="AIME",
        help="Short benchmark name for figure titles (e.g. AMC 12)",
    )
    args = parser.parse_args()

    suite = Path(args.suite_dir).resolve()
    if not suite.is_dir():
        print(f"Not a directory: {suite}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir).resolve() if args.output_dir else suite
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sub in sorted(p for p in suite.iterdir() if p.is_dir()):
        run_id = sub.name
        if run_id.startswith(".") or run_id == "logs" or run_id == out_dir.name:
            continue
        combined_path = ensure_combined(sub)
        if combined_path is None:
            print(f"[warn] skip {run_id}: no combined_results.csv")
            continue
        m = metrics_from_csv(combined_path)
        m["run_id"] = run_id
        m["label"] = read_run_label(sub, run_id)
        rows.append(m)

    if not rows:
        print("No runs with combined_results.csv found.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    # Stable display order: slm, llm, neural, then entropy-variance YAML runs by τ
    order_pref = {"slm": 0, "llm": 1, "r2r_neural": 2}

    def sort_key(rid: str) -> tuple:
        if rid in order_pref:
            return (order_pref[rid], 0.0)
        if rid.startswith("r2r_entropy_"):
            tail = rid.replace("r2r_entropy_", "").replace("_", ".")
            try:
                return (3, float(tail))
            except ValueError:
                return (3, 0.0)
        m = re.search(r"entropy_([0-9.]+)_variance", rid)
        if m:
            return (3, float(m.group(1)))
        return (9, rid)

    df["_k"] = df["run_id"].map(lambda x: sort_key(x))
    df = df.sort_values("_k").drop(columns=["_k"])

    csv_path = out_dir / "benchmark_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    if plt is None:
        print("matplotlib not installed; skip plots. pip install matplotlib")
        return

    labels = df["label"].tolist()
    x = range(len(df))

    fig1, ax1 = plt.subplots(figsize=(max(8, len(df) * 0.45), 4.2))
    acc_pct = df["accuracy"] * 100.0
    ax1.bar(x, acc_pct, color="#2a6f97")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(f"{args.title_tag} accuracy by method")
    ax1.set_ylim(0, max(100, acc_pct.max() * 1.1) if acc_pct.notna().any() else 100)
    fig1.tight_layout()
    bar_acc = out_dir / "benchmark_accuracy.png"
    fig1.savefig(bar_acc, dpi=150)
    plt.close(fig1)
    print(f"Wrote {bar_acc}")

    if df["throughput_tokens_per_s"].notna().any():
        fig2, ax2 = plt.subplots(figsize=(max(8, len(df) * 0.45), 4.2))
        tp = df["throughput_tokens_per_s"].fillna(0)
        ax2.bar(x, tp, color="#9c6644")
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(labels, rotation=35, ha="right")
        ax2.set_ylabel("Tokens/s (output)")
        ax2.set_title("Generation throughput by method")
        fig2.tight_layout()
        bar_tp = out_dir / "benchmark_throughput.png"
        fig2.savefig(bar_tp, dpi=150)
        plt.close(fig2)
        print(f"Wrote {bar_tp}")

    if df["avg_wall_time_s_per_problem"].notna().any():
        fig_w, ax_w = plt.subplots(figsize=(max(8, len(df) * 0.45), 4.2))
        wt = df["avg_wall_time_s_per_problem"].astype(float)
        ax_w.bar(x, wt, color="#bc6c25")
        ax_w.set_xticks(list(x))
        ax_w.set_xticklabels(labels, rotation=35, ha="right")
        ax_w.set_ylabel("Seconds / problem (wall)")
        ax_w.set_title("Mean wall time per problem")
        fig_w.tight_layout()
        bar_w = out_dir / "benchmark_wall_time_per_problem.png"
        fig_w.savefig(bar_w, dpi=150)
        plt.close(fig_w)
        print(f"Wrote {bar_w}")

    # Two-row overview: accuracy + throughput
    if df["accuracy"].notna().any():
        fig_ov, (ax_a, ax_t) = plt.subplots(2, 1, figsize=(max(9, len(df) * 0.5), 8.0), sharex=True)
        acc_pct = df["accuracy"] * 100.0
        ax_a.bar(x, acc_pct, color="#2a6f97")
        ax_a.set_ylabel("Accuracy (%)")
        ax_a.set_title(f"{args.title_tag}: accuracy and speed by method")
        ax_a.set_ylim(0, max(100, acc_pct.max() * 1.1) if acc_pct.notna().any() else 100)
        if df["throughput_tokens_per_s"].notna().any():
            ax_t.bar(x, df["throughput_tokens_per_s"].fillna(0), color="#9c6644")
            ax_t.set_ylabel("Output tok/s")
        elif df["avg_wall_time_s_per_problem"].notna().any():
            ax_t.bar(x, df["avg_wall_time_s_per_problem"].astype(float), color="#bc6c25")
            ax_t.set_ylabel("s / problem")
        ax_t.set_xticks(list(x))
        ax_t.set_xticklabels(labels, rotation=40, ha="right")
        fig_ov.tight_layout()
        ov_path = out_dir / "benchmark_overview_accuracy_speed.png"
        fig_ov.savefig(ov_path, dpi=150)
        plt.close(fig_ov)
        print(f"Wrote {ov_path}")

    if df["accuracy"].notna().any() and df["throughput_tokens_per_s"].notna().any():
        fig3, ax3 = plt.subplots(figsize=(7.0, 5.5))
        ax3.scatter(df["throughput_tokens_per_s"], df["accuracy"] * 100.0, s=90, c="#6a4c93", alpha=0.88, edgecolors="white", linewidths=0.5)
        for i, lab in enumerate(labels):
            ax3.annotate(
                lab,
                (df["throughput_tokens_per_s"].iloc[i], df["accuracy"].iloc[i] * 100.0),
                fontsize=7,
                xytext=(5, 4),
                textcoords="offset points",
            )
        ax3.set_xlabel("Throughput (output tokens/s)")
        ax3.set_ylabel("Accuracy (%)")
        ax3.set_title("Accuracy vs throughput")
        fig3.tight_layout()
        sc_path = out_dir / "benchmark_acc_vs_speed.png"
        fig3.savefig(sc_path, dpi=150)
        plt.close(fig3)
        print(f"Wrote {sc_path}")


if __name__ == "__main__":
    main()
