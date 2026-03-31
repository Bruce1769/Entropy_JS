#!/usr/bin/env python3
"""Aggregate entropy-lookahead triggers and LLM switches from entropy_lookahead_score_log.jsonl.

Each JSON line with event ``entropy_lookahead_triggered`` means: at that decode step the SLM
next-token entropy was >= ``entropy_threshold``, so lookahead (draft + LLM logprob RPC) ran.

``routed_to_llm: true`` means this step used the reference (LLM) for the actual token —
either because ``S > score_threshold`` (RPC ok), or because RPC failed and the server
falls back to LLM (``rpc_ok: false``).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", type=Path, help="Path to entropy_lookahead_score_log.jsonl")
    p.add_argument(
        "--skip-warmup-rid-max",
        type=int,
        default=None,
        metavar="N",
        help="Drop rids that are int-castable and <= N (e.g. 4 if warmup used rids 0–4).",
    )
    args = p.parse_args()

    by_rid: dict[str, dict] = defaultdict(
        lambda: {
            "triggers": 0,
            "switched": 0,
            "scored_switch": 0,
            "rpc_fail_switch": 0,
            "first_ts": None,
        }
    )
    score_threshold = None

    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("event") != "entropy_lookahead_triggered":
                continue
            rid = str(r.get("rid", ""))
            ts = r.get("ts")
            ent = by_rid[rid]
            if ent["first_ts"] is None or (ts is not None and ts < ent["first_ts"]):
                ent["first_ts"] = ts
            if score_threshold is None and "score_threshold" in r:
                score_threshold = float(r["score_threshold"])
            st = float(r.get("score_threshold", score_threshold or 0.0))

            ent["triggers"] += 1
            if r.get("routed_to_llm"):
                ent["switched"] += 1
            if r.get("rpc_ok") is False:
                ent["rpc_fail_switch"] += 1
            elif r.get("rpc_ok") and r.get("S") is not None:
                if float(r["S"]) > st:
                    ent["scored_switch"] += 1

    def rid_sort_key(k: str):
        try:
            return (0, int(k))
        except ValueError:
            return (1, k)

    def skip_rid(rid: str) -> bool:
        if args.skip_warmup_rid_max is None:
            return False
        try:
            return int(rid) <= args.skip_warmup_rid_max
        except ValueError:
            return False

    rows = [(rid, v) for rid, v in by_rid.items() if not skip_rid(rid)]
    rows.sort(key=lambda x: (x[1]["first_ts"] is None, x[1]["first_ts"] or 0, rid_sort_key(x[0])))

    print(f"file: {args.jsonl}")
    print(f"score_threshold (from log): {score_threshold}")
    print()
    print("idx\trid\tentropy_triggers\tllm_switch\tS>thr(rpc_ok)\trpc_fail->LLM")
    tot_t = tot_s = tot_sc = tot_fb = 0
    for idx, (rid, v) in enumerate(rows):
        tot_t += v["triggers"]
        tot_s += v["switched"]
        tot_sc += v["scored_switch"]
        tot_fb += v["rpc_fail_switch"]
        print(
            f"{idx}\t{rid}\t{v['triggers']}\t{v['switched']}\t{v['scored_switch']}\t{v['rpc_fail_switch']}"
        )
    print("---")
    print(
        f"rows={len(rows)}\ttriggers={tot_t}\tllm_switch={tot_s}\tS>thr={tot_sc}\trpc_fail={tot_fb}"
    )
    if tot_t:
        print(f"switch_rate_among_triggers: {tot_s / tot_t:.4f}")


if __name__ == "__main__":
    main()
