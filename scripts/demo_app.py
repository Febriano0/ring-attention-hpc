#!/usr/bin/env python3
"""
Aplikasi Demo Langsung Triase Cepat Sampel Sekuens DNA Jaringan Payudara (PRJNA913438):
Perbandingan berdampingan Baseline V0 (Naive O(s^2)) vs Ring Attention V3 (MPI O(cd)).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from download_and_tokenize import tokenize_fastq_file  # noqa: E402

ATTN_BIN = ROOT / "build" / "attn"
WEIGHTS_DIR = ROOT / "models" / "breast_triage_d64"
FASTQ_DIR = ROOT / "data" / "fastq"


def run_attention_side(
    version: str,
    tokens_bin: Path,
    s: int,
    np_ranks: int,
    ulimit_kb: int | None,
) -> Tuple[int, Dict[str, object], str]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as jtmp:
        jpath = Path(jtmp.name)
    jpath.unlink(missing_ok=True)

    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_PROC_BIND"] = "close"
    env["OMP_PLACES"] = "cores"

    c = s // np_ranks
    cmd = []
    if np_ranks > 1:
        cmd.extend(["mpirun", "--bind-to", "core", "-np", str(np_ranks)])
    cmd.extend([
        str(ATTN_BIN),
        "--version", version,
        "--weights", str(WEIGHTS_DIR),
        "--input", str(tokens_bin),
        "--s", str(s),
        "--d", "64",
        "--c", str(c),
        "--tile", "64",
        "--json-out", str(jpath),
    ])

    try:
        if ulimit_kb is not None and ulimit_kb > 0:
            shell_cmd = f"ulimit -v {ulimit_kb} && " + " ".join(cmd)
            proc = subprocess.run(["sh", "-c", shell_cmd], cwd=ROOT, env=env, text=True, capture_output=True)
        else:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)

        data = json.loads(jpath.read_text(encoding="utf-8")) if jpath.exists() else {}
        return proc.returncode, data, (proc.stdout + "\n" + proc.stderr).strip()
    finally:
        jpath.unlink(missing_ok=True)


def render_side_card(title: str, rc: int, data: Dict[str, object], logs: str) -> None:
    st.subheader(title)
    if rc != 0 or not data:
        st.error(f"Gagal / Out-Of-Memory (exit code {rc})")
        st.code(logs[-600:] if logs else "Alokasi matriks skor s x s melampaui batas ulimit -v.")
        return

    conf = 100.0 * max(float(data["prob_normal"]), float(data["prob_progression"]))
    pred = str(data["predicted_label"])
    if int(data["predicted_class"]) == 0:
        st.success(f"Prediksi: **{pred}** ({conf:.2f}%)")
    else:
        st.warning(f"Prediksi: **{pred}** ({conf:.2f}%)")

    c1, c2 = st.columns(2)
    c1.metric("Waktu Inferensi", f"{float(data['time_s']) * 1e3:.2f} ms")
    c2.metric("Throughput", f"{float(data['throughput_reads_per_s']):,.1f} read/s")
    c1.metric("Peak RSS / proses", f"{int(data['peak_rss_kb']) / 1024.0:.2f} MiB")
    c2.metric("Delta RSS aktivasi", f"{int(data['delta_rss_kb']) / 1024.0:.2f} MiB")


def main() -> None:
    st.set_page_config(page_title="Ring Attention DNA Triage Demo", layout="wide")
    st.title("Triase Cepat Sekuens DNA Jaringan Payudara (PRJNA913438)")
    st.caption(
        "Demonstrasi langsung perbandingan **Baseline V0 (Naive Attention)** vs "
        "**Ring Attention V3 (MPI + Online Softmax)** pada mesin multi-core. "
        "*Alat bantu riset untuk triase awal, bukan alat diagnosis klinis.*"
    )

    fastq_files = sorted(FASTQ_DIR.glob("*.fastq"))
    if not fastq_files:
        st.error("Tidak ada file FASTQ di data/fastq/. Jalankan scripts/download_and_tokenize.py terlebih dahulu.")
        return

    with st.sidebar:
        st.header("Parameter Demo")
        selected_name = st.selectbox("Sampel FASTQ Lokal", [p.name for p in fastq_files])
        s = st.select_slider("Panjang Konteks s (token)", options=[1024, 2048, 4096, 8192, 16384], value=4096)
        p_ranks = st.radio("Jumlah Proses MPI (P) untuk V3", options=[1, 2, 4], index=2, horizontal=True)
        use_ulimit = st.checkbox("Aktifkan Simulasi Batas Memori (ulimit -v)", value=(s == 16384))
        ulimit_mb = st.slider("Batas Virtual Memory (MiB)", min_value=800, max_value=2000, value=1172, step=64,
                              disabled=not use_ulimit)
        run_btn = st.button("Jalankan V0 vs V3", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Pilih sampel dan parameter di sidebar kiri, lalu klik **Jalankan V0 vs V3**.")
        return

    fastq_path = FASTQ_DIR / selected_name
    ulimit_kb = (ulimit_mb * 1024) if use_ulimit else None

    with st.spinner(f"Men-tokenisasi 6-mer dari {selected_name} dan menjalankan V0 vs V3 (s={s}, P={p_ranks})..."):
        tokens = tokenize_fastq_file(fastq_path, max_tokens=s)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as btmp:
            bin_path = Path(btmp.name)
        tokens.tofile(bin_path)
        try:
            rc0, d0, log0 = run_attention_side("v0", bin_path, s, 1, ulimit_kb)
            rc3, d3, log3 = run_attention_side("v3", bin_path, s, p_ranks, ulimit_kb)
        finally:
            bin_path.unlink(missing_ok=True)

    col_left, col_right = st.columns(2)
    with col_left:
        render_side_card("Sebelum: Baseline V0 (P=1)", rc0, d0, log0)
    with col_right:
        render_side_card(f"Sesudah: Ring Attention V3 (P={p_ranks})", rc3, d3, log3)

    if rc3 == 0 and d3:
        st.divider()
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=180)
        labels = ["V0 (P=1)", f"V3 (P={p_ranks})"]
        times_ms = [float(d0["time_s"]) * 1e3 if rc0 == 0 and d0 else 0.0, float(d3["time_s"]) * 1e3]
        mem_mb = [float(d0["delta_rss_kb"]) / 1024.0 if rc0 == 0 and d0 else 0.0, float(d3["delta_rss_kb"]) / 1024.0]
        colors = ["#d62728" if rc0 == 0 else "#7f7f7f", "#2ca02c"]

        axes[0].bar(labels, times_ms, color=colors)
        axes[0].set_title("Waktu Inferensi (ms) — Lebih Kecil Lebih Baik")
        axes[0].set_ylabel("ms")
        if rc0 != 0:
            axes[0].text(0, max(times_ms) * 0.15, "OOM", ha="center", fontweight="bold", color="#d62728")

        axes[1].bar(labels, mem_mb, color=colors)
        axes[1].set_title("Delta Peak RSS per Proses (MiB)")
        axes[1].set_ylabel("MiB")
        if rc0 != 0:
            axes[1].text(0, max(mem_mb) * 0.15, "OOM", ha="center", fontweight="bold", color="#d62728")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
