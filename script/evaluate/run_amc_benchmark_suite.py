#!/usr/bin/env python3
"""
在 AMC（及 amc_recent）上跑与 ``run_aime_benchmark_suite.py`` 相同的方法对比流程：

  SLM / LLM / R2R-neural / 熵阈值扫（或 ``--entropy-variance-configs`` 下的 YAML）。

等价于在该脚本上默认加上 ``--dataset amc``（若命令行未显式指定 ``--dataset``）。

示例::

    cd R2R
    export CUDA_VISIBLE_DEVICES=0,1
    python script/evaluate/run_amc_benchmark_suite.py \\
        --config-path config/Qwen3-0.6B+Qwen3-8B.yaml \\
        --tp_size 1 --slm_tp_size 1 --llm_tp_size 1

近年子集（更快）::

    python script/evaluate/run_amc_benchmark_suite.py \\
        --config-path config/Qwen3-0.6B+Qwen3-8B.yaml \\
        --dataset amc_recent \\
        --num_problems 50
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> None:
    argv = sys.argv[1:]
    if "--dataset" not in argv:
        argv = ["--dataset", "amc"] + argv
    sys.argv = [sys.argv[0]] + argv

    suite_path = Path(__file__).resolve().parent / "run_aime_benchmark_suite.py"
    spec = importlib.util.spec_from_file_location("_r2r_aime_suite", suite_path)
    if spec is None or spec.loader is None:
        print(f"Cannot load {suite_path}", file=sys.stderr)
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    main()
