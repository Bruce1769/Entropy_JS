#!/usr/bin/env python3
import argparse
import json
import os
import time
from datetime import datetime


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _fmt(v) -> str:
    try:
        return f"{float(v):.6f}"
    except Exception:
        return "NA"


def _read_new_lines(path: str, offset: int):
    if not os.path.exists(path):
        return offset, []
    size = os.path.getsize(path)
    if size < offset:
        offset = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        lines = f.readlines()
        offset = f.tell()
    return offset, lines


def _print_runlog_lines(lines):
    for raw in lines:
        line = raw.replace("\r", "").strip()
        if not line:
            continue
        keep = (
            "[current tokens/s]" in line
            or "[quick rank" in line
            or "[evjs-" in line
            or "waiting finished reqs" in line
            or "Processing batches:" in line
        )
        if keep:
            print(f"[{_now()}] {line}", flush=True)


def _print_scorelog_lines(lines, show_all_decisions: bool):
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue

        event = rec.get("event")
        sub = rec.get("sub")
        if event != "entropy_variance_js":
            continue

        rid = rec.get("rid")
        ent = rec.get("entropy")
        ent_thr = rec.get("entropy_threshold")
        js = rec.get("js")
        js_thr = rec.get("js_threshold")

        if sub == "js_route":
            print(
                f"[{_now()}] [switch->LLM] rid={rid} "
                f"entropy={_fmt(ent)}/{_fmt(ent_thr)} js={_fmt(js)}/{_fmt(js_thr)}",
                flush=True,
            )
        elif show_all_decisions and sub in ("js_below", "rpc_fail"):
            if sub == "js_below":
                print(
                    f"[{_now()}] [keep->SLM] rid={rid} "
                    f"entropy={_fmt(ent)}/{_fmt(ent_thr)} js={_fmt(js)}/{_fmt(js_thr)}",
                    flush=True,
                )
            else:
                print(
                    f"[{_now()}] [rpc-fail] rid={rid} entropy={_fmt(ent)}/{_fmt(ent_thr)} "
                    f"error={rec.get('error', '')}",
                    flush=True,
                )


def main():
    parser = argparse.ArgumentParser(
        description="Watch R2R runtime logs (token/s, switching events, waiting progress)."
    )
    parser.add_argument("--output_dir", required=True, help="R2R eval output dir")
    parser.add_argument(
        "--score_log",
        default=None,
        help="Score log jsonl path (default: <output_dir>/entropy_variance_js_score_log.jsonl)",
    )
    parser.add_argument("--poll_s", type=float, default=1.0, help="Polling interval in seconds")
    parser.add_argument(
        "--all_decisions",
        action="store_true",
        help="Also print js_below/rpc_fail decisions (default: only switch->LLM).",
    )
    args = parser.parse_args()

    run_log = os.path.join(args.output_dir, "run.log")
    score_log = args.score_log or os.path.join(args.output_dir, "entropy_variance_js_score_log.jsonl")

    print(f"[{_now()}] watching run_log={run_log}")
    print(f"[{_now()}] watching score_log={score_log}")

    run_off = 0
    score_off = 0
    while True:
        run_off, run_lines = _read_new_lines(run_log, run_off)
        _print_runlog_lines(run_lines)

        score_off, score_lines = _read_new_lines(score_log, score_off)
        _print_scorelog_lines(score_lines, args.all_decisions)

        time.sleep(max(0.1, args.poll_s))


if __name__ == "__main__":
    main()
