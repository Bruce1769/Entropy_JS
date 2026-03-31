#!/usr/bin/env python3
"""Visualize entropy lookahead logs: H vs S, and cheap SLM-side proxies vs S.

**Current jsonl (older runs):** always has ``entropy``, ``S`` (when rpc_ok), and
``slm_logprobs`` — enough for the main H–S figure and for **derived** features
``sum / min / mean`` of SLM logprobs (no model replay).

**Newer runs** may also log ``entropy_path_sum`` / ``entropy_path`` from
``slm_server.py``; those are plotted when present.

Optional: ``enrich_entropy_lookahead_score_log.py`` writes the same derived
numbers into a new jsonl for downstream analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy import stats
except ImportError:
    stats = None


def load_scored_rows(path: Path) -> tuple[list[dict], list[float], float | None, float | None]:
    """Return (scored_rows, H_fail_list, thr_ent, thr_s)."""
    scored: list[dict] = []
    H_fail: list[float] = []
    thr_ent: float | None = None
    thr_s: float | None = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("event") != "entropy_lookahead_triggered":
                continue
            H = float(r["entropy"])
            if thr_ent is None and "entropy_threshold" in r:
                thr_ent = float(r["entropy_threshold"])
            if thr_s is None and "score_threshold" in r:
                thr_s = float(r["score_threshold"])
            if r.get("rpc_ok") and r.get("S") is not None:
                row: dict = {
                    "H": H,
                    "S": float(r["S"]),
                    "routed": bool(r.get("routed_to_llm")),
                }
                eps = r.get("entropy_path_sum")
                if eps is not None:
                    row["entropy_path_sum"] = float(eps)
                slm = r.get("slm_logprobs")
                if isinstance(slm, list) and len(slm) > 0:
                    a = np.asarray(slm, dtype=np.float64)
                    row["sum_slm_lp"] = float(a.sum())
                    row["min_slm_lp"] = float(a.min())
                    row["mean_slm_lp"] = float(a.mean())
                scored.append(row)
            else:
                H_fail.append(H)
    return scored, H_fail, thr_ent, thr_s


def _corr_txt(x: np.ndarray, y: np.ndarray) -> str:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or stats is None:
        return ""
    pr, pp = stats.pearsonr(x[m], y[m])
    sr, sp = stats.spearmanr(x[m], y[m])
    return f"Pearson r={pr:.3f}  Spearman ρ={sr:.3f}"


def plot_entropy_vs_S(
    H: np.ndarray,
    S: np.ndarray,
    routed: np.ndarray,
    H_fail: np.ndarray,
    thr_ent: float | None,
    thr_s: float | None,
    title_name: str,
    out: Path,
) -> None:
    fig = plt.figure(figsize=(11, 10), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 0.45], width_ratios=[1, 1])

    ax_hex = fig.add_subplot(gs[0, :])
    ax_bin = fig.add_subplot(gs[1, :])
    ax_hist_h = fig.add_subplot(gs[2, 0])
    ax_hist_s = fig.add_subplot(gs[2, 1])

    y_lo, y_hi = np.percentile(S, [1, 99])
    y_pad = (y_hi - y_lo) * 0.08 + 0.5
    y0, y1 = y_lo - y_pad, y_hi + y_pad
    outside = int(np.sum((S < y0) | (S > y1)))
    hb = ax_hex.hexbin(
        H,
        S,
        gridsize=(55, 45),
        extent=(float(H.min()), float(H.max()), y0, y1),
        mincnt=1,
        cmap="YlOrRd",
        linewidths=0.2,
        edgecolors="face",
    )
    ax_hex.axhline(thr_s or 1.0, color="steelblue", ls="--", lw=1.2, label=f"score_threshold={thr_s}")
    ax_hex.set_xlabel("Entropy H (nats, next-token SLM at trigger step)")
    ax_hex.set_ylabel("S = Σ(log p_SLM − log p_LLM) on draft")
    ax_hex.set_title(
        f"Entropy vs S  |  n={len(S):,} scored  |  {outside:,} pts outside y clip (1–99%)"
    )
    cb = fig.colorbar(hb, ax=ax_hex, shrink=0.75, label="count / bin")
    try:
        cb.solids.set_edgecolor("face")
    except AttributeError:
        pass

    if stats is not None and len(S) > 2:
        pr, pp = stats.pearsonr(H, S)
        sr, sp = stats.spearmanr(H, S)
        ax_hex.text(
            0.02,
            0.98,
            f"Pearson r = {pr:.3f} (p={pp:.2e})\nSpearman ρ = {sr:.3f} (p={sp:.2e})",
            transform=ax_hex.transAxes,
            va="top",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
        )

    if len(H_fail) > 0:
        y_rug = y0 - (y1 - y0) * 0.06
        ax_hex.scatter(
            H_fail,
            np.full_like(H_fail, y_rug, dtype=np.float64),
            s=4,
            c="tab:blue",
            alpha=0.25,
            marker="|",
            linewidths=0.8,
            label=f"RPC fail (no S), n={len(H_fail):,}",
        )
        ax_hex.set_ylim(y_rug - (y1 - y0) * 0.02, y1)
    ax_hex.legend(loc="upper right", fontsize=9)

    n_bins = 24
    counts, edges = np.histogram(H, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    med, q1, q3 = [], [], []
    for i in range(n_bins):
        m = (H >= edges[i]) & (H < edges[i + 1])
        if i == n_bins - 1:
            m = (H >= edges[i]) & (H <= edges[i + 1])
        if m.sum() < 5:
            med.append(np.nan)
            q1.append(np.nan)
            q3.append(np.nan)
            continue
        sb = S[m]
        med.append(np.median(sb))
        q1.append(np.percentile(sb, 25))
        q3.append(np.percentile(sb, 75))
    med, q1, q3 = map(np.asarray, (med, q1, q3))
    valid = ~np.isnan(med)
    ax_bin.fill_between(
        centers[valid], q1[valid], q3[valid], color="tab:orange", alpha=0.35, label="S IQR (25–75%)"
    )
    ax_bin.plot(centers[valid], med[valid], "o-", color="darkred", ms=4, lw=1.2, label="median S")
    ax_bin.axhline(thr_s or 1.0, color="steelblue", ls="--", lw=1.0)
    if thr_ent is not None:
        ax_bin.axvline(thr_ent, color="gray", ls=":", lw=1.0, label=f"entropy_threshold={thr_ent}")
    ax_bin.set_xlabel("Entropy H")
    ax_bin.set_ylabel("S (per H bin)")
    ax_bin.set_title("Binned: H → distribution of S")
    ax_bin.legend(loc="best", fontsize=8)

    ax_hist_h.hist(
        H[~routed],
        bins=40,
        alpha=0.55,
        color="tab:green",
        label="stay SLM (S ≤ thr)",
        density=True,
    )
    ax_hist_h.hist(
        H[routed],
        bins=40,
        alpha=0.55,
        color="tab:red",
        label="route LLM",
        density=True,
    )
    if thr_ent is not None:
        ax_hist_h.axvline(thr_ent, color="gray", ls=":", lw=1)
    ax_hist_h.set_xlabel("H")
    ax_hist_h.set_ylabel("density")
    ax_hist_h.set_title("H at triggers (by decision)")
    ax_hist_h.legend(fontsize=7)

    s_lo, s_hi = float(np.percentile(S, 0.5)), float(np.percentile(S, 99.5))
    if s_lo >= s_hi:
        s_lo, s_hi = float(S.min()), float(S.max())
    ax_hist_s.hist(
        S[~routed],
        bins=50,
        range=(s_lo, s_hi),
        alpha=0.55,
        color="tab:green",
        label="stay SLM",
        density=True,
    )
    ax_hist_s.hist(
        S[routed],
        bins=50,
        range=(s_lo, s_hi),
        alpha=0.55,
        color="tab:red",
        label="route LLM",
        density=True,
    )
    ax_hist_s.axvline(thr_s or 1.0, color="steelblue", ls="--", lw=1)
    ax_hist_s.set_xlabel("S")
    ax_hist_s.set_ylabel("density")
    ax_hist_s.set_title("S (0.5–99.5% clip)")
    ax_hist_s.legend(fontsize=7)

    fig.suptitle(title_name, fontsize=10, y=1.02)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_cheap_proxies_vs_S(
    scored: list[dict],
    thr_s: float | None,
    title_name: str,
    out: Path,
) -> None:
    S = np.array([r["S"] for r in scored], dtype=np.float64)
    sum_slm = np.array([r.get("sum_slm_lp", np.nan) for r in scored], dtype=np.float64)
    min_slm = np.array([r.get("min_slm_lp", np.nan) for r in scored], dtype=np.float64)
    mean_slm = np.array([r.get("mean_slm_lp", np.nan) for r in scored], dtype=np.float64)
    hpsum = np.array([r.get("entropy_path_sum", np.nan) for r in scored], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")
    st = thr_s or 1.0

    def one_hex(ax, x, xlab: str):
        m = np.isfinite(x) & np.isfinite(S)
        if m.sum() < 10:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(xlab)
            return
        y_lo, y_hi = np.percentile(S[m], [1, 99])
        y_pad = (y_hi - y_lo) * 0.08 + 0.5
        y0, y1 = y_lo - y_pad, y_hi + y_pad
        hb = ax.hexbin(
            x[m],
            S[m],
            gridsize=45,
            mincnt=1,
            cmap="BuPu",
            linewidths=0.15,
            edgecolors="face",
            extent=(float(np.percentile(x[m], 0.5)), float(np.percentile(x[m], 99.5)), y0, y1),
        )
        fig.colorbar(hb, ax=ax, shrink=0.7, label="count")
        ax.axhline(st, color="steelblue", ls="--", lw=1)
        ax.set_xlabel(xlab)
        ax.set_ylabel("S")
        ax.set_title(xlab + " vs S")
        t = _corr_txt(x, S)
        if t:
            ax.text(0.02, 0.98, t, transform=ax.transAxes, va="top", fontsize=8, family="monospace")

    one_hex(axes[0, 0], sum_slm, "Σ log p_SLM (draft tokens)")
    one_hex(axes[0, 1], min_slm, "min log p_SLM (draft)")
    one_hex(axes[1, 0], mean_slm, "mean log p_SLM (draft)")

    ax = axes[1, 1]
    m = np.isfinite(hpsum) & np.isfinite(S)
    if m.sum() >= 10:
        one_hex(ax, hpsum, "entropy_path_sum (Σ H along draft)")
    else:
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            "No ``entropy_path_sum`` in this log.\n"
            "It is written by newer ``slm_server.py`` when you\n"
            "re-run eval. Current file still has\n"
            "``slm_logprobs`` → see other panels.",
            ha="center",
            va="center",
            fontsize=11,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
        )
        ax.set_title("entropy_path_sum (optional)")

    fig.suptitle(
        title_name + "\nCheap SLM-side proxies from log (no LLM); not Shannon ΣH unless path sum logged",
        fontsize=10,
        y=1.02,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", type=Path)
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Main H-vs-S PNG (default: <stem>_entropy_vs_S.png)",
    )
    ap.add_argument(
        "--proxy-output",
        type=Path,
        default=None,
        help="Cheap-proxies PNG (default: <stem>_cheap_proxies_vs_S.png); use '' to skip",
    )
    ap.add_argument("--no-proxy-plot", action="store_true", help="Do not write cheap-proxies figure")
    args = ap.parse_args()

    scored, H_fail, thr_ent, thr_s = load_scored_rows(args.jsonl)
    if not scored:
        raise SystemExit("No scored rows (rpc_ok with S); nothing to plot.")

    H = np.array([r["H"] for r in scored], dtype=np.float64)
    S = np.array([r["S"] for r in scored], dtype=np.float64)
    routed = np.array([r["routed"] for r in scored], dtype=bool)
    H_fail_a = np.asarray(H_fail, dtype=np.float64)

    out_main = args.output or args.jsonl.with_name(f"{args.jsonl.stem}_entropy_vs_S.png")
    plot_entropy_vs_S(H, S, routed, H_fail_a, thr_ent, thr_s, args.jsonl.name, out_main)

    if args.no_proxy_plot:
        return
    out_px = args.proxy_output
    if out_px is not None and str(out_px) == "":
        return
    if out_px is None:
        out_px = args.jsonl.with_name(f"{args.jsonl.stem}_cheap_proxies_vs_S.png")
    plot_cheap_proxies_vs_S(scored, thr_s, args.jsonl.name, out_px)


if __name__ == "__main__":
    main()
