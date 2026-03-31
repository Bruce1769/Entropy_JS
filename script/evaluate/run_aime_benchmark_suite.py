#!/usr/bin/env python3
"""
Sequential evals for method comparison (AIME, AMC, MATH-500, etc.—any dataset key in
``eval_configs/dataset_configs.json``):

  - SLM only (--use_model quick)
  - LLM only (--use_model reference)
  - R2R + neural router (--use_hybrid, strategy from YAML or default neural)
  - Router sweep, either:
      * default: R2R + entropy via CLI (``--switching_strategy entropy --threshold T``), or
      * ``--entropy-variance-configs``: one ``--use_hybrid`` run per YAML (strategy/thresholds
        come only from each file, e.g. ``entropy_variance`` + ``entropy_threshold`` /
        ``variance_threshold`` — same workflow as swapping config files by hand).

Each run writes to ``--base-output/<run_id>/``. After all runs, use
``aggregate_and_plot_benchmarks.py`` on ``--base-output`` to build a table and figures.

If FlashInfer JIT fails to compile (nvcc errors in conda GCC 15 libstdc++), activate conda first
then run ``source script/evaluate/setup_flashinfer_host_compiler.sh`` so ``CC`` points at a
supported host compiler (FlashInfer passes ``CC`` to nvcc as ``-ccbin``, not ``CXX``).

Example (CLI entropy sweep on AIME)::

    cd R2R
    export CUDA_VISIBLE_DEVICES=0,1
    python script/evaluate/run_aime_benchmark_suite.py \\
        --config-path config/Qwen3-0.6B+Qwen3-8B.yaml \\
        --dataset aime \\
        --tp_size 1 --slm_tp_size 1 --llm_tp_size 1

Example (AMC 12 full set; or use ``script/evaluate/run_amc_benchmark_suite.py`` which defaults ``--dataset amc``)::

    python script/evaluate/run_aime_benchmark_suite.py \\
        --config-path config/Qwen3-0.6B+Qwen3-8B.yaml \\
        --dataset amc \\
        --tp_size 1 --slm_tp_size 1 --llm_tp_size 1

Example (one YAML per entropy+variance router, like ``config/*_entropy_*_variance_*.yaml``)::

    python script/evaluate/run_aime_benchmark_suite.py \\
        --config-path config/Qwen3-0.6B+Qwen3-8B.yaml \\
        --entropy-variance-configs \\
            config/Qwen3-0.6B+Qwen3-8B_entropy_0.45_variance_1e-5.yaml \\
            config/Qwen3-0.6B+Qwen3-8B_entropy_0.7_variance_1e-5.yaml \\
        --dataset aime

Extra flags are forwarded to ``hf_dataset_sglang.py`` (e.g. ``--num_problems 5 --debug``).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _r2r_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _entropy_thresholds() -> list[float]:
    return [0.45, 0.7, 0.95, 1.2, 1.5, 1.7, 2.0]


def _run_id_from_config_path(config_path: str) -> str:
    stem = Path(config_path).stem
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem).strip("_")[:120] or "cfg"
    return f"r2r_{safe}"


def build_run_specs(entropy_variance_configs: list[str] | None) -> list[tuple[str, list[str], str | None]]:
    """Return (run_id, extra_hf_args, config_path_override). None override uses --config-path.

    ``entropy_variance_configs``:
      - ``None`` (flag not used): append default CLI entropy-threshold sweep.
      - ``[]`` (flag used, no paths): no extra router runs after ``r2r_neural``.
      - non-empty: one ``--use_hybrid`` run per YAML (router fields from file).
    """
    specs: list[tuple[str, list[str], str | None]] = [
        ("slm", ["--use_model", "quick"], None),
        ("llm", ["--use_model", "reference"], None),
        ("r2r_neural", ["--use_hybrid"], None),
    ]
    if entropy_variance_configs is not None:
        for p in entropy_variance_configs:
            path = str(Path(p).resolve())
            specs.append((_run_id_from_config_path(path), ["--use_hybrid"], path))
    else:
        for t in _entropy_thresholds():
            tid = str(t).replace(".", "_")
            specs.append(
                (
                    f"r2r_entropy_{tid}",
                    ["--use_hybrid", "--switching_strategy", "entropy", "--threshold", str(t)],
                    None,
                )
            )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SLM / LLM / R2R / entropy-sweep evals sequentially.")
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument(
        "--entropy-variance-configs",
        nargs="*",
        default=None,
        metavar="YAML",
        help="Optional: extra router YAMLs (one eval each with --use_hybrid only). "
        "When set, replaces the default CLI entropy-threshold sweep. "
        "Use for entropy_variance configs where each file sets entropy_threshold / variance_threshold.",
    )
    parser.add_argument("--dataset", type=str, default="aime")
    parser.add_argument(
        "--base-output",
        type=str,
        default=None,
        help="Directory for per-run outputs (default: output/eval/method_suite_<timestamp>)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run if <run_id>/combined_results.csv already exists",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python binary for hf_dataset_sglang.py",
    )
    args, unknown = parser.parse_known_args()

    root = _r2r_root()
    base = Path(args.base_output) if args.base_output else root / "output" / "eval" / f"method_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base.mkdir(parents=True, exist_ok=True)

    eval_script = root / "script" / "evaluate" / "hf_dataset_sglang.py"
    if not eval_script.is_file():
        print(f"Missing {eval_script}", file=sys.stderr)
        sys.exit(1)

    def make_cmd(config_path: str) -> list[str]:
        return [
            str(args.python),
            str(eval_script),
            "--dataset",
            args.dataset,
            "--config-path",
            config_path,
            "--generator",
            "sglang",
        ]
    # Sensible defaults if user did not pass tp flags via unknown
    forwarded = list(unknown)
    if "--tp_size" not in forwarded:
        forwarded.extend(["--tp_size", "1"])
    if "--slm_tp_size" not in forwarded:
        forwarded.extend(["--slm_tp_size", "1"])
    if "--llm_tp_size" not in forwarded:
        forwarded.extend(["--llm_tp_size", "1"])

    specs = build_run_specs(args.entropy_variance_configs)
    print(f"Base output directory: {base}")
    print(f"Planned runs: {len(specs)}")
    if args.entropy_variance_configs is not None:
        print("Router sweep: per-file configs (--entropy-variance-configs), not CLI entropy thresholds.")

    for run_id, extra, config_override in specs:
        out_dir = base / run_id
        combined = out_dir / "combined_results.csv"
        if args.skip_existing and combined.is_file():
            print(f"[skip] {run_id} (found {combined})")
            continue

        cfg = config_override if config_override is not None else args.config_path
        cmd = make_cmd(cfg) + ["--output_dir", str(out_dir)] + extra + forwarded
        print("\n" + "=" * 80)
        print("RUN:", run_id)
        print("CMD:", " ".join(cmd))
        print("=" * 80)
        if args.dry_run:
            continue

        env = os.environ.copy()
        p = subprocess.run(cmd, cwd=str(root), env=env)
        if p.returncode != 0:
            print(f"[error] {run_id} exited with {p.returncode}", file=sys.stderr)
            sys.exit(p.returncode)

    print("\nDone. Aggregate and plot with:\n")
    print(f"  python script/evaluate/aggregate_and_plot_benchmarks.py --suite-dir {base}\n")


if __name__ == "__main__":
    main()
