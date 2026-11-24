#!/usr/bin/env python3
"""
Skrip otomasi eksperimen lengkap (Bagian 6 Proposal):
  0. Mikro-benchmark GEMM (F) dan MPI Ping-Pong (B)
  a. Verifikasi numerik V0..V4 terhadap V0-FP64 (normal & stress logits)
  b. Profil memori puncak per proses (V0 vs V3 dengan c tetap)
  c. Strong scaling dan Weak scaling (V2 vs V3, P = 1, 2, 4)
  d. Efektivitas overlap eta dan validasi model analitik c >= 2F/B
  e. Perbandingan aplikasi sebelum-sesudah (V0 vs V3 pada data DNA PRJNA913438) + ulimit -v OOM
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent.parent
ATTN_BIN = ROOT / "build" / "attn"
BENCH_GEMM_BIN = ROOT / "build" / "bench_gemm"
BENCH_PINGPONG_BIN = ROOT / "build" / "bench_pingpong"
RAW_DIR = ROOT / "results" / "raw"
SUMMARY_DIR = ROOT / "results" / "summary"

BASE_ENV = os.environ.copy()
BASE_ENV["OPENBLAS_NUM_THREADS"] = "1"
BASE_ENV["OMP_PROC_BIND"] = "close"
BASE_ENV["OMP_PLACES"] = "cores"


def run_cmd(cmd: List[str], env: Dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, env=env or BASE_ENV, text=True, capture_output=True, check=check)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_attn_single_csv(args_list: List[str], np: int = 1, threads: int | None = None) -> Dict[str, str]:
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    tmp_path.unlink(missing_ok=True)

    env = BASE_ENV.copy()
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)

    if np > 1:
        if threads is not None and threads > 1:
            cmd = ["mpirun", "--map-by", f"slot:PE={threads}", "--bind-to", "core", "-np", str(np), str(ATTN_BIN)]
        else:
            cmd = ["mpirun", "--bind-to", "core", "-np", str(np), str(ATTN_BIN)]
    else:
        cmd = [str(ATTN_BIN)]
    cmd.extend(args_list)
    cmd.extend(["--csv", str(tmp_path)])

    try:
        run_cmd(cmd, env=env, check=True)
        rows = read_csv_rows(tmp_path)
        return rows[-1]
    finally:
        tmp_path.unlink(missing_ok=True)


def run_microbenchmarks() -> Tuple[float, Dict[int, float]]:
    print("[exp 0] Menjalankan mikro-benchmark GEMM & MPI Ping-Pong...")
    gemm_csv = RAW_DIR / "gemm_benchmark.csv"
    mpi_csv = RAW_DIR / "mpi_pingpong.csv"
    gemm_csv.unlink(missing_ok=True)
    mpi_csv.unlink(missing_ok=True)

    run_cmd([str(BENCH_GEMM_BIN), "--reps", "7", "--warmup", "2", "--csv", str(gemm_csv)])
    run_cmd(["mpirun", "--bind-to", "core", "-np", "2", str(BENCH_PINGPONG_BIN),
             "--reps", "50", "--warmup", "10", "--csv", str(mpi_csv)])

    gemm_rows = read_csv_rows(gemm_csv)
    tile64_gflops = [float(r["gflops"]) for r in gemm_rows if r["M"] == "64" and r["K"] == "64"]
    f_gemm_tile64 = statistics.mean(tile64_gflops) * 1e9

    mpi_rows = read_csv_rows(mpi_csv)
    bw_by_bytes = {int(r["bytes"]): float(r["bandwidth_bytes_per_s"]) for r in mpi_rows}
    return f_gemm_tile64, bw_by_bytes


def exp_accuracy() -> List[Dict[str, object]]:
    print("[exp a] Verifikasi numerik V0..V4 (normal & stress logits)...")
    cases = [
        ("v0", 1, 1, False), ("v1", 1, 1, False), ("v2", 1, 4, False),
        ("v3", 4, 1, False), ("v4", 2, 2, False),
        ("v0", 1, 1, True),  ("v1", 1, 1, True),  ("v2", 1, 4, True),
        ("v3", 4, 1, True),  ("v4", 2, 2, True),
    ]
    rows: List[Dict[str, object]] = []
    for ver, np, thr, stress in cases:
        cli = ["--version", ver, "--s", "1024", "--d", "64", "--c", str(1024 // np if ver in ("v3", "v4") else 256),
               "--tile", "64", "--threads", str(thr), "--seed", "1", "--verify"]
        if stress:
            cli.append("--stress-logits")
        r = run_attn_single_csv(cli, np=np, threads=thr)
        rows.append({
            "version": ver,
            "P": np,
            "threads": thr,
            "scenario": "stress_logits_std10" if stress else "standard_normal",
            "s": r["s"],
            "d": r["d"],
            "c": r["c"],
            "max_abs_err": r["max_abs_err"],
            "rel_err": r["rel_err"],
            "verified": r["verified"],
        })
    write_csv(RAW_DIR / "exp_accuracy.csv", list(rows[0].keys()), rows)
    return rows


def exp_memory(reps: int) -> List[Dict[str, object]]:
    print("[exp b] Profil memori puncak per proses (V0 vs V3 c tetap)...")
    rows: List[Dict[str, object]] = []
    for s in [1024, 2048, 4096, 8192]:
        r = run_attn_single_csv(
            ["--version", "v0", "--s", str(s), "--d", "64", "--reps", str(min(reps, 3)), "--warmup", "1"],
            np=1,
        )
        rows.append({
            "series": "v0_naive",
            "version": "v0",
            "P": 1,
            "c": s,
            "s": s,
            "peak_rss_kb": r["peak_rss_kb"],
            "delta_rss_kb": r["delta_rss_kb"],
            "alloc_kb": r["alloc_kb"],
            "time_s": r["time_s"],
        })

    for c_fixed in [1024, 2048]:
        for np in [1, 2, 4]:
            s = np * c_fixed
            r = run_attn_single_csv(
                ["--version", "v3", "--s", str(s), "--d", "64", "--c", str(c_fixed), "--tile", "64",
                 "--reps", str(reps), "--warmup", "1"],
                np=np,
            )
            rows.append({
                "series": f"v3_c{c_fixed}",
                "version": "v3",
                "P": np,
                "c": c_fixed,
                "s": s,
                "peak_rss_kb": r["peak_rss_kb"],
                "delta_rss_kb": r["delta_rss_kb"],
                "alloc_kb": r["alloc_kb"],
                "time_s": r["time_s"],
            })

    write_csv(RAW_DIR / "exp_memory.csv", list(rows[0].keys()), rows)
    return rows


def exp_scaling(reps: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    print("[exp c] Strong scaling & Weak scaling (V2 vs V3, P = 1, 2, 4)...")
    strong_rows: List[Dict[str, object]] = []
    s_strong = 4096
    for ver in ["v2", "v3"]:
        t1 = 0.0
        for p in [1, 2, 4]:
            if ver == "v2":
                r = run_attn_single_csv(
                    ["--version", "v2", "--s", str(s_strong), "--d", "64", "--c", "256", "--tile", "64",
                     "--threads", str(p), "--reps", str(reps), "--warmup", "1"],
                    np=1, threads=p,
                )
            else:
                r = run_attn_single_csv(
                    ["--version", "v3", "--s", str(s_strong), "--d", "64", "--c", str(s_strong // p), "--tile", "64",
                     "--reps", str(reps), "--warmup", "1"],
                    np=p,
                )
            t = float(r["time_s"])
            if p == 1:
                t1 = t
            speedup = t1 / max(t, 1e-12)
            eff = speedup / p
            gflops = (4.0 * s_strong * s_strong * 64.0) / max(t, 1e-12) / 1e9
            strong_rows.append({
                "version": ver, "P": p, "s": s_strong, "d": 64, "c": r["c"],
                "time_s": f"{t:.6e}", "stddev_s": r["stddev_s"],
                "gflops": f"{gflops:.3f}", "speedup": f"{speedup:.3f}", "efficiency": f"{eff:.4f}",
                "peak_rss_kb": r["peak_rss_kb"], "delta_rss_kb": r["delta_rss_kb"],
            })
    write_csv(RAW_DIR / "exp_strong_scaling.csv", list(strong_rows[0].keys()), strong_rows)

    weak_rows: List[Dict[str, object]] = []
    c_weak = 1024
    for ver in ["v2", "v3"]:
        t1 = 0.0
        for p in [1, 2, 4]:
            s = p * c_weak
            if ver == "v2":
                r = run_attn_single_csv(
                    ["--version", "v2", "--s", str(s), "--d", "64", "--c", str(c_weak), "--tile", "64",
                     "--threads", str(p), "--reps", str(reps), "--warmup", "1"],
                    np=1, threads=p,
                )
            else:
                r = run_attn_single_csv(
                    ["--version", "v3", "--s", str(s), "--d", "64", "--c", str(c_weak), "--tile", "64",
                     "--reps", str(reps), "--warmup", "1"],
                    np=p,
                )
            t = float(r["time_s"])
            if p == 1:
                t1 = t
            # Pada attention dengan c tetap (s = P*c), kerja per proses tumbuh P kali (P langkah x 4*c^2*d).
            scaled_eff = (p * t1) / max(t, 1e-12)
            total_gflops = (4.0 * s * s * 64.0) / max(t, 1e-12) / 1e9
            weak_rows.append({
                "version": ver, "P": p, "s": s, "d": 64, "c": c_weak,
                "time_s": f"{t:.6e}", "stddev_s": r["stddev_s"],
                "total_gflops": f"{total_gflops:.3f}",
                "gflops_per_proc": f"{total_gflops / p:.3f}",
                "work_scaled_efficiency": f"{scaled_eff:.4f}",
                "peak_rss_kb": r["peak_rss_kb"], "delta_rss_kb": r["delta_rss_kb"],
            })
    write_csv(RAW_DIR / "exp_weak_scaling.csv", list(weak_rows[0].keys()), weak_rows)
    return strong_rows, weak_rows


def exp_overlap(reps: int, f_gemm_tile64: float, bw_by_bytes: Dict[int, float]) -> List[Dict[str, object]]:
    print("[exp d] Efektivitas overlap (eta) dan perbandingan model c >= 2F/B...")
    rows: List[Dict[str, object]] = []
    p = 4
    d = 64
    for c in [128, 256, 512, 1024, 2048]:
        s = p * c
        r_blk = run_attn_single_csv(
            ["--version", "v3", "--s", str(s), "--d", str(d), "--c", str(c), "--tile", "64",
             "--blocking", "--reps", str(reps), "--warmup", "1"], np=p)
        r_nb = run_attn_single_csv(
            ["--version", "v3", "--s", str(s), "--d", str(d), "--c", str(c), "--tile", "64",
             "--reps", str(reps), "--warmup", "1"], np=p)
        r_comp = run_attn_single_csv(
            ["--version", "v3", "--s", str(s), "--d", str(d), "--c", str(c), "--tile", "64",
             "--no-comm", "--reps", str(reps), "--warmup", "1"], np=p)

        t_blk = float(r_blk["time_s"])
        t_nb = float(r_nb["time_s"])
        t_comp = float(r_comp["time_s"])
        t_wait = float(r_nb["wait_s"])
        denom = t_blk - t_comp
        eta = ((t_blk - t_nb) / denom) if denom > 1e-9 else 0.0

        kv_bytes = 2 * c * d * 4
        bw = bw_by_bytes.get(kv_bytes, max(bw_by_bytes.values()))
        # Total FLOPs per rank across P=4 steps = 4 * (4 * c^2 * d)
        f_attn_per_proc = (4.0 * 4.0 * c * c * d) / max(t_comp, 1e-12)
        c_min_gemm = (2.0 * f_gemm_tile64) / bw
        c_min_attn = (2.0 * f_attn_per_proc) / bw

        rows.append({
            "c": c, "s": s, "P": p, "kv_bytes": kv_bytes,
            "T_blk_s": f"{t_blk:.6e}", "T_nb_s": f"{t_nb:.6e}", "T_comp_s": f"{t_comp:.6e}",
            "T_wait_s": f"{t_wait:.6e}", "eta": f"{eta:.4f}",
            "B_gb_s": f"{bw / 1e9:.3f}",
            "F_attn_gflops": f"{f_attn_per_proc / 1e9:.2f}",
            "c_min_gemm": f"{c_min_gemm:.1f}",
            "c_min_attn": f"{c_min_attn:.1f}",
        })
    write_csv(RAW_DIR / "exp_overlap.csv", list(rows[0].keys()), rows)
    return rows


def run_triage_json(version: str, input_bin: Path, s: int, np: int, ulimit_kb: int | None = None) -> Tuple[int, Dict[str, object], str]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        jpath = Path(tmp.name)
    jpath.unlink(missing_ok=True)

    c = s // np
    cmd_parts = []
    if np > 1:
        cmd_parts.extend(["mpirun", "--bind-to", "core", "-np", str(np)])
    cmd_parts.extend([
        str(ATTN_BIN), "--version", version,
        "--weights", str(ROOT / "models" / "breast_triage_d64"),
        "--input", str(input_bin),
        "--s", str(s), "--d", "64", "--c", str(c), "--tile", "64",
        "--json-out", str(jpath),
    ])

    try:
        if ulimit_kb is not None:
            shell_cmd = f"ulimit -v {ulimit_kb} && " + " ".join(cmd_parts)
            proc = subprocess.run(["bash", "-c", shell_cmd], cwd=ROOT, env=BASE_ENV, text=True, capture_output=True)
        else:
            proc = run_cmd(cmd_parts, check=False)

        data = json.loads(jpath.read_text(encoding="utf-8")) if jpath.exists() else {}
        return proc.returncode, data, (proc.stdout + "\n" + proc.stderr).strip()
    finally:
        jpath.unlink(missing_ok=True)


def exp_application(reps: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    print("[exp e] Aplikasi sebelum-sesudah V0 vs V3 pada data DNA PRJNA913438...")
    samples = [
        ("184D_normal", ROOT / "data" / "tokens" / "184D_normal.bin"),
        ("184AA2_aig", ROOT / "data" / "tokens" / "184AA2_aig.bin"),
    ]
    app_rows: List[Dict[str, object]] = []
    for sample_name, bin_path in samples:
        for s in [1024, 2048, 4096, 8192]:
            for ver, np in [("v0", 1), ("v3", 4)]:
                # Warmup 1 + reps timed
                run_triage_json(ver, bin_path, s, np)
                times, throughputs, peaks, deltas = [], [], [], []
                last_data: Dict[str, object] = {}
                for _ in range(reps):
                    rc, data, out_txt = run_triage_json(ver, bin_path, s, np)
                    if rc != 0 or not data:
                        raise RuntimeError(f"Triage failed ({ver}, s={s}): {out_txt}")
                    last_data = data
                    times.append(float(data["time_s"]))
                    throughputs.append(float(data["throughput_reads_per_s"]))
                    peaks.append(int(data["peak_rss_kb"]))
                    deltas.append(int(data["delta_rss_kb"]))

                med_t = statistics.median(times)
                std_t = statistics.stdev(times) if len(times) > 1 else 0.0
                app_rows.append({
                    "sample": sample_name,
                    "version": ver,
                    "P": np,
                    "s": s,
                    "predicted_label": last_data["predicted_label"],
                    "confidence_pct": f"{100.0 * max(float(last_data['prob_normal']), float(last_data['prob_progression'])):.2f}",
                    "time_s": f"{med_t:.6e}",
                    "stddev_s": f"{std_t:.6e}",
                    "throughput_reads_per_s": f"{statistics.median(throughputs):.2f}",
                    "peak_rss_kb": max(peaks),
                    "delta_rss_kb": max(deltas),
                    "alloc_kb": f"{float(last_data['alloc_kb']):.2f}",
                })
    write_csv(RAW_DIR / "exp_app_comparison.csv", list(app_rows[0].keys()), app_rows)

    # Uji batas memori ulimit -v pada s = 16384 (V0 butuh ~1.05 GiB hanya untuk S)
    oom_rows: List[Dict[str, object]] = []
    ulimit_kb = 1200000
    for ver, np in [("v0", 1), ("v3", 4)]:
        rc, data, log_txt = run_triage_json(ver, samples[0][1], 16384, np, ulimit_kb=ulimit_kb)
        oom_rows.append({
            "ulimit_v_kb": ulimit_kb,
            "s": 16384,
            "version": ver,
            "P": np,
            "exit_code": rc,
            "status": "SUCCESS" if rc == 0 else "OOM_FAILED",
            "predicted_label": data.get("predicted_label", "N/A (OOM)"),
            "time_s": f"{float(data['time_s']):.5f}" if "time_s" in data else "N/A",
            "peak_rss_kb": data.get("peak_rss_kb", -1),
            "delta_rss_kb": data.get("delta_rss_kb", -1),
            "note": "Allocated O(cd) per rank" if rc == 0 else "std::bad_alloc on s x s score matrix",
        })
    write_csv(RAW_DIR / "exp_ulimit_oom.csv", list(oom_rows[0].keys()), oom_rows)
    return app_rows, oom_rows


def write_markdown_summary(
    f_gemm_tile64: float,
    acc_rows: List[Dict[str, object]],
    mem_rows: List[Dict[str, object]],
    strong_rows: List[Dict[str, object]],
    weak_rows: List[Dict[str, object]],
    overlap_rows: List[Dict[str, object]],
    app_rows: List[Dict[str, object]],
    oom_rows: List[Dict[str, object]],
) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Ringkasan Hasil Eksperimen Ring Attention HPC\n",
        f"- Laju GEMM tile 64x64x64 (`tiled_avx2_fma`): **{f_gemm_tile64 / 1e9:.2f} GFLOP/s** (1 core)\n",
        "\n## 1. Verifikasi Numerik terhadap V0-FP64 (Target `rel_err < 1e-5`)\n",
        "| Versi | P | Threads | Skenario | Max Abs Err | Rel L2 Err | Status |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for r in acc_rows:
        lines.append(f"| {r['version']} | {r['P']} | {r['threads']} | {r['scenario']} | {float(r['max_abs_err']):.3e} | {float(r['rel_err']):.3e} | {'PASS' if str(r['verified']) == '1' else 'FAIL'} |")

    lines.extend([
        "\n## 2. Profil Memori per Proses (V0 vs V3 dengan `c` Tetap)\n",
        "| Seri | P | c | s | Peak RSS (KiB) | Delta RSS (KiB) | Alloc Teoretis (KiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for r in mem_rows:
        lines.append(f"| {r['series']} | {r['P']} | {r['c']} | {r['s']} | {r['peak_rss_kb']} | {r['delta_rss_kb']} | {r['alloc_kb']} |")

    lines.extend([
        "\n## 3. Strong Scaling (`s = 4096, d = 64`)\n",
        "| Versi | P | c | Median Time (s) | GFLOP/s | Speedup | Efisiensi |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for r in strong_rows:
        lines.append(f"| {r['version']} | {r['P']} | {r['c']} | {float(r['time_s']):.5f} | {r['gflops']} | {r['speedup']}x | {100*float(r['efficiency']):.1f}% |")

    lines.extend([
        "\n## 4. Weak Scaling (`c = 1024` tetap, `s = P * c`)\n",
        "*Catatan: Pada attention dengan `c` tetap, kerja per proses tumbuh linear `O(P * c^2 * d)` sementara memori aktivasi tetap `O(c * d)`.*\n",
        "| Versi | P | s | Median Time (s) | Total GFLOP/s | GFLOP/s per Proses | Work-Scaled Eff |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for r in weak_rows:
        lines.append(f"| {r['version']} | {r['P']} | {r['s']} | {float(r['time_s']):.5f} | {r['total_gflops']} | {r['gflops_per_proc']} | {100*float(r['work_scaled_efficiency']):.1f}% |")

    lines.extend([
        "\n## 5. Efektivitas Overlap (`P = 4, d = 64`) vs Model Analitik `c >= 2F/B`\n",
        "| c | s | T_blk (s) | T_nb (s) | T_comp (s) | T_wait (s) | Overlap eta | B (GB/s) | c_min (GEMM) | c_min (Attn) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in overlap_rows:
        lines.append(
            f"| {r['c']} | {r['s']} | {float(r['T_blk_s']):.5f} | {float(r['T_nb_s']):.5f} | "
            f"{float(r['T_comp_s']):.5f} | {float(r['T_wait_s']):.5f} | {float(r['eta']):.3f} | "
            f"{r['B_gb_s']} | {r['c_min_gemm']} | {r['c_min_attn']} |"
        )

    lines.extend([
        "\n## 6. Aplikasi Triase DNA PRJNA913438: Sebelum (`V0`) vs Sesudah (`V3, P = 4`)\n",
        "| Sampel | s | Versi | Prediksi | Keyakinan | Latensi (s) | Throughput (read/s) | Delta RSS (KiB) |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ])
    for r in app_rows:
        lines.append(
            f"| {r['sample']} | {r['s']} | {r['version']} (P={r['P']}) | {r['predicted_label']} | "
            f"{r['confidence_pct']}% | {float(r['time_s']):.5f} | {r['throughput_reads_per_s']} | {r['delta_rss_kb']} |"
        )

    lines.extend([
        "\n## 7. Uji Batas Memori `ulimit -v 1200000` (`s = 16384`)\n",
        "| Versi | P | s | Status | Exit Code | Delta RSS (KiB) | Keterangan |",
        "|---|---:|---:|---|---:|---:|---|",
    ])
    for r in oom_rows:
        lines.append(f"| {r['version']} | {r['P']} | {r['s']} | {r['status']} | {r['exit_code']} | {r['delta_rss_kb']} | {r['note']} |")

    out_md = SUMMARY_DIR / "experiment_summary.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] Ringkasan tersimpan di {out_md}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all Ring Attention HPC experiments.")
    parser.add_argument("--reps", type=int, default=5, help="Repetitions per configuration (Proposal min: 5)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    f_gemm_tile64, bw_by_bytes = run_microbenchmarks()
    acc_rows = exp_accuracy()
    mem_rows = exp_memory(args.reps)
    strong_rows, weak_rows = exp_scaling(args.reps)
    overlap_rows = exp_overlap(args.reps, f_gemm_tile64, bw_by_bytes)
    app_rows, oom_rows = exp_application(args.reps)

    write_markdown_summary(
        f_gemm_tile64, acc_rows, mem_rows, strong_rows, weak_rows, overlap_rows, app_rows, oom_rows
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
