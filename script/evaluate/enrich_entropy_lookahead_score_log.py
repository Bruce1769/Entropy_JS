#!/usr/bin/env python3
"""Add offline-derived fields to entropy_lookahead_score_log.jsonl (no model replay).

Adds from existing ``slm_logprobs`` when present::

    sum_slm_logprobs, mean_slm_logprob, min_slm_logprob, max_slm_logprob

Adds from ``llm_logprobs`` when present::

    sum_llm_logprobs, mean_llm_logprob, min_llm_logprob

``S`` should equal sum_slm_logprobs - sum_llm_logprobs (sanity check).

Does **not** reconstruct Shannon entropy along the path (needs per-step logits).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def enrich_record(r: dict[str, Any]) -> dict[str, Any]:
    out = dict(r)
    slm = r.get("slm_logprobs")
    if isinstance(slm, list) and len(slm) > 0:
        a = np.asarray(slm, dtype=np.float64)
        out["sum_slm_logprobs"] = float(a.sum())
        out["mean_slm_logprob"] = float(a.mean())
        out["min_slm_logprob"] = float(a.min())
        out["max_slm_logprob"] = float(a.max())
    llm = r.get("llm_logprobs")
    if isinstance(llm, list) and len(llm) > 0:
        b = np.asarray(llm, dtype=np.float64)
        out["sum_llm_logprobs"] = float(b.sum())
        out["mean_llm_logprob"] = float(b.mean())
        out["min_llm_logprob"] = float(b.min())
    if (
        out.get("sum_slm_logprobs") is not None
        and out.get("sum_llm_logprobs") is not None
        and r.get("S") is not None
    ):
        recon = out["sum_slm_logprobs"] - out["sum_llm_logprobs"]
        out["S_minus_recon"] = float(r["S"]) - recon
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_jsonl", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output jsonl path")
    args = ap.parse_args()

    n_in = n_out = 0
    bad = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input_jsonl, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("event") == "entropy_lookahead_triggered":
                er = enrich_record(r)
                if er.get("S_minus_recon") is not None and abs(er["S_minus_recon"]) > 1e-3:
                    bad += 1
                fout.write(json.dumps(er, ensure_ascii=False) + "\n")
                n_out += 1
            else:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_out += 1
    print(f"Read {n_in} lines, wrote {n_out} records to {args.output}")
    if bad:
        print(f"Warning: {bad} rows where |S - (sum_slm - sum_llm)| > 1e-3")


if __name__ == "__main__":
    main()
