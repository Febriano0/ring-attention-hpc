#!/usr/bin/env python3
"""
Generator grafik publikasi (300 DPI) di results/figures/ dari CSV di results/raw/.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "results" / "raw"
FIG_DIR = ROOT / "results" / "figures"


def load_csv(name: str) -> List[Dict[str, str]]:
    with open(RAW_DIR / name, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def plot_memory_scaling() -> None:
    rows = load_csv("exp_memory.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=300)

    series_meta = [
        ("v0_naive", "V0 Naive (P=1, $O(s^2)$)", "o-", "#d62728"),
        ("v3_c1024", "V3 Ring MPI ($c=1024$ tetap)", "s--", "#1f77b4"),
        ("v3_c2048", "V3 Ring MPI ($c=2048$ tetap)", "^-", "#2ca02c"),
    ]
    for key, label, fmt, color in series_meta:
        sub = [r for r in rows if r["series"] == key]
        sub.sort(key=lambda r: int(r["s"]))
        xs = [int(r["s"]) for r in sub]
        deltas = [float(r["delta_rss_kb"]) / 1024.0 for r in sub]
        allocs = [float(r["alloc_kb"]) / 1024.0 for r in sub]
        axes[0].plot(xs, deltas, fmt, color=color, linewidth=2, markersize=6, label=label)
        axes[1].plot(xs, allocs, fmt, color=color, linewidth=2, markersize=6, label=label)

    axes[0].set_title("Delta Peak RSS Terukur per Proses")
    axes[0].set_xlabel("Panjang Sekuens Total $s$")
    axes[0].set_ylabel("Delta Peak RSS (MiB)")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].set_title("Memori Aktivasi Teoretis per Proses")
    axes[1].set_xlabel("Panjang Sekuens Total $s$")
    axes[1].set_ylabel("Memori Aktivasi (MiB)")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", linestyle=":", alpha=0.6)
    axes[1].legend()

    fig.suptitle("Skala Memori per Proses: V0 Kuadratik vs V3 Konstan terhadap $s$", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_memory_scaling_v0_vs_v3.png", bbox_inches="tight")
    plt.close(fig)


def plot_strong_scaling() -> None:
    rows = load_csv("exp_strong_scaling.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=300)

    ps = [1, 2, 4]
    axes[1].plot(ps, ps, "k--", linewidth=1.5, label="Ideal Linear")

    for ver, label, fmt, color in [
        ("v2", "V2 Blockwise + OpenMP", "o-", "#1f77b4"),
        ("v3", "V3 Ring Attention + MPI", "s-", "#ff7f0e"),
    ]:
        sub = sorted([r for r in rows if r["version"] == ver], key=lambda r: int(r["P"]))
        x = [int(r["P"]) for r in sub]
        t = [float(r["time_s"]) * 1e3 for r in sub]
        sp = [float(r["speedup"]) for r in sub]
        axes[0].plot(x, t, fmt, color=color, linewidth=2, markersize=6, label=label)
        axes[1].plot(x, sp, fmt, color=color, linewidth=2, markersize=6, label=label)

    axes[0].set_title("Waktu Eksekusi ($s=4096, d=64$)")
    axes[0].set_xlabel("Jumlah Proses / Thread $P$")
    axes[0].set_ylabel("Waktu Median (ms)")
    axes[0].set_xticks(ps)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].set_title("Speedup $S(P) = T_1 / T_P$")
    axes[1].set_xlabel("Jumlah Proses / Thread $P$")
    axes[1].set_ylabel("Speedup ($\\times$)")
    axes[1].set_xticks(ps)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    fig.suptitle("Strong Scaling: Shared-Memory OpenMP (V2) vs Ring MPI (V3)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_strong_scaling.png", bbox_inches="tight")
    plt.close(fig)


def plot_weak_scaling() -> None:
    rows = load_csv("exp_weak_scaling.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=300)
    ps = [1, 2, 4]

    for ver, label, fmt, color in [
        ("v2", "V2 Blockwise + OpenMP", "o-", "#1f77b4"),
        ("v3", "V3 Ring Attention + MPI", "s-", "#ff7f0e"),
    ]:
        sub = sorted([r for r in rows if r["version"] == ver], key=lambda r: int(r["P"]))
        x = [int(r["P"]) for r in sub]
        gflops_tot = [float(r["total_gflops"]) for r in sub]
        eff = [100.0 * float(r["work_scaled_efficiency"]) for r in sub]
        axes[0].plot(x, gflops_tot, fmt, color=color, linewidth=2, markersize=6, label=label)
        axes[1].plot(x, eff, fmt, color=color, linewidth=2, markersize=6, label=label)

    axes[0].set_title("Total Throughput ($c=1024$ tetap, $s=P \\cdot c$)")
    axes[0].set_xlabel("Jumlah Proses / Thread $P$ ($s = 1024P$)")
    axes[0].set_ylabel("Aggregate Throughput (GFLOP/s)")
    axes[0].set_xticks(ps)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].axhline(100.0, color="k", linestyle="--", linewidth=1.2, label="Ideal 100%")
    axes[1].set_title("Efisiensi Skala Kerja ($P \\cdot T_1 / T_P$)")
    axes[1].set_xlabel("Jumlah Proses / Thread $P$")
    axes[1].set_ylabel("Efisiensi (%)")
    axes[1].set_xticks(ps)
    axes[1].set_ylim(0, 120)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    fig.suptitle("Weak Scaling ($c=1024$ tetap, Panjang Sekuens $s$ Tumbuh Linear)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_weak_scaling.png", bbox_inches="tight")
    plt.close(fig)


def plot_overlap() -> None:
    rows = load_csv("exp_overlap.csv")
    rows.sort(key=lambda r: int(r["c"]))
    cs = [int(r["c"]) for r in rows]
    t_blk = [float(r["T_blk_s"]) * 1e3 for r in rows]
    t_nb = [float(r["T_nb_s"]) * 1e3 for r in rows]
    t_comp = [float(r["T_comp_s"]) * 1e3 for r in rows]
    etas = [float(r["eta"]) for r in rows]
    c_min_attn = float(rows[0]["c_min_attn"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=300)

    axes[0].plot(cs, t_blk, "o-", color="#d62728", linewidth=2, label="$T_{\\mathrm{blk}}$ (`MPI_Sendrecv`)")
    axes[0].plot(cs, t_nb, "s-", color="#1f77b4", linewidth=2, label="$T_{\\mathrm{nb}}$ (`Isend/Irecv`)")
    axes[0].plot(cs, t_comp, "^--", color="#2ca02c", linewidth=2, label="$T_{\\mathrm{comp}}$ (tanpa kom.)")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xticks(cs)
    axes[0].set_xticklabels([str(c) for c in cs])
    axes[0].set_title("Perbandingan Waktu Blocking vs Non-Blocking ($P=4$)")
    axes[0].set_xlabel("Ukuran Blok $c$")
    axes[0].set_ylabel("Waktu Median (ms)")
    axes[0].grid(True, which="both", linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].plot(cs, etas, "D-", color="#9467bd", linewidth=2, markersize=6, label="Efektivitas Overlap $\\eta$")
    axes[1].axvline(c_min_attn, color="#8c564b", linestyle="--", linewidth=1.5,
                    label=f"Prediksi $2F/B \\approx {c_min_attn:.1f}$")
    axes[1].axhline(1.0, color="k", linestyle=":", linewidth=1.2, label="Overlap Sempurna ($\\eta=1$)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(cs)
    axes[1].set_xticklabels([str(c) for c in cs])
    y_lo = min(-0.2, min(etas) - 0.15)
    y_hi = max(1.3, max(etas) + 0.15)
    axes[1].set_ylim(y_lo, y_hi)
    axes[1].set_title("Efektivitas Overlap $\\eta = (T_{\\mathrm{blk}} - T_{\\mathrm{nb}})/(T_{\\mathrm{blk}} - T_{\\mathrm{comp}})$")
    axes[1].set_xlabel("Ukuran Blok $c$")
    axes[1].set_ylabel("Rasio Overlap $\\eta$")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    fig.suptitle("Analisis Overlap Komunikasi–Komputasi Ring Attention (V3, $P=4$)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_overlap_analysis.png", bbox_inches="tight")
    plt.close(fig)


def plot_app_before_after() -> None:
    rows = load_csv("exp_app_comparison.csv")
    sub = [r for r in rows if r["sample"] == "184D_normal"]
    seq_lens = sorted({int(r["s"]) for r in sub})

    v0_rows = sorted([r for r in sub if r["version"] == "v0"], key=lambda r: int(r["s"]))
    v3_rows = sorted([r for r in sub if r["version"] == "v3"], key=lambda r: int(r["s"]))

    x = np.arange(len(seq_lens))
    width = 0.36

    v0_thr = [float(r["throughput_reads_per_s"]) for r in v0_rows]
    v3_thr = [float(r["throughput_reads_per_s"]) for r in v3_rows]
    v0_mem = [float(r["delta_rss_kb"]) / 1024.0 for r in v0_rows]
    v3_mem = [float(r["delta_rss_kb"]) / 1024.0 for r in v3_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=300)

    axes[0].bar(x - width / 2, v0_thr, width, color="#d62728", label="Sebelum: V0 Naive ($P=1$)")
    axes[0].bar(x + width / 2, v3_thr, width, color="#2ca02c", label="Sesudah: V3 Ring MPI ($P=4$)")
    axes[0].set_title("Throughput Inferensi DNA Payudara (`184D_normal`)")
    axes[0].set_xlabel("Panjang Konteks $s$ (token)")
    axes[0].set_ylabel("Throughput (read/detik)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(s) for s in seq_lens])
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].bar(x - width / 2, v0_mem, width, color="#d62728", label="Sebelum: V0 Naive ($P=1$)")
    axes[1].bar(x + width / 2, v3_mem, width, color="#2ca02c", label="Sesudah: V3 Ring MPI ($P=4$)")
    axes[1].set_title("Delta Peak RSS per Proses saat Inferensi")
    axes[1].set_xlabel("Panjang Konteks $s$ (token)")
    axes[1].set_ylabel("Delta Peak RSS (MiB)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(s) for s in seq_lens])
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.6)
    axes[1].legend()

    fig.suptitle("Evaluasi Aplikasi Sebelum vs Sesudah Optimasi (BioProject PRJNA913438)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_app_before_after_dna.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_memory_scaling()
    plot_strong_scaling()
    plot_weak_scaling()
    plot_overlap()
    plot_app_before_after()
    print(f"[done] 5 grafik disimpan di {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
