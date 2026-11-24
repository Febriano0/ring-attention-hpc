#!/usr/bin/env bash
set -uo pipefail

FASTQ_FILE="${1:-data/fastq/184D_normal.fastq}"
SEQ_LEN="${2:-4096}"
NP_RANKS="${3:-4}"
ULIMIT_KB="${4:-0}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ATTN_BIN="${ROOT_DIR}/build/attn"
WEIGHTS_DIR="${ROOT_DIR}/models/breast_triage_d64"

if [[ ! -x "${ATTN_BIN}" ]]; then
  echo "[error] ${ATTN_BIN} belum dikompilasi. Jalankan 'make build/attn' terlebih dahulu." >&2
  exit 1
fi

TMP_BIN="$(mktemp /tmp/dna_tokens_XXXXXX.bin)"
TMP_V0_JSON="$(mktemp /tmp/v0_out_XXXXXX.json)"
TMP_V3_JSON="$(mktemp /tmp/v3_out_XXXXXX.json)"
trap 'rm -f "${TMP_BIN}" "${TMP_V0_JSON}" "${TMP_V3_JSON}"' EXIT

python3 - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "${ROOT_DIR}/scripts")
from download_and_tokenize import tokenize_fastq_file
arr = tokenize_fastq_file(Path("${FASTQ_FILE}"), max_tokens=${SEQ_LEN})
arr.tofile("${TMP_BIN}")
print(f"[demo] Sampel=${FASTQ_FILE} | tokens={arr.size} | s=${SEQ_LEN} | P=${NP_RANKS} | ulimit_v=${ULIMIT_KB}KB")
PY

export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

C_BLOCK=$(( SEQ_LEN / NP_RANKS ))
if [[ $(( C_BLOCK * NP_RANKS )) -ne "${SEQ_LEN}" ]]; then
  echo "[error] -- Panjang konteks s=${SEQ_LEN} harus habis dibagi P=${NP_RANKS}." >&2
  exit 1
fi

run_side() {
  local cmd="$1"
  if [[ "${ULIMIT_KB}" -gt 0 ]]; then
    sh -c "ulimit -v ${ULIMIT_KB} && ${cmd}"
  else
    sh -c "${cmd}"
  fi
}

echo "--- Menjalankan Baseline V0 (P=1) ---"
run_side "${ATTN_BIN} --version v0 --weights ${WEIGHTS_DIR} --input ${TMP_BIN} --s ${SEQ_LEN} --d 64 --json-out ${TMP_V0_JSON}"
RC_V0=$?

echo "--- Menjalankan Optimasi V3 Ring MPI (P=${NP_RANKS}, c=${C_BLOCK}) ---"
run_side "mpirun --bind-to core -np ${NP_RANKS} ${ATTN_BIN} --version v3 --weights ${WEIGHTS_DIR} --input ${TMP_BIN} --s ${SEQ_LEN} --d 64 --c ${C_BLOCK} --tile 64 --json-out ${TMP_V3_JSON}"
RC_V3=$?

python3 - <<PY
import json
from pathlib import Path

def fmt(path_str, rc):
    p = Path(path_str)
    if rc != 0 or not p.exists() or p.stat().st_size == 0:
        return f"OOM / GAGAL (exit={rc})", "-", "-", "-", "-"
    d = json.loads(p.read_text())
    conf = 100.0 * max(d["prob_normal"], d["prob_progression"])
    return (
        f"{d['predicted_label']} ({conf:.2f}%)",
        f"{d['time_s']:.5f} s",
        f"{d['throughput_reads_per_s']:.1f} read/s",
        f"{d['peak_rss_kb']} KiB",
        f"{d['delta_rss_kb']} KiB",
    )

r0 = fmt("${TMP_V0_JSON}", ${RC_V0})
r3 = fmt("${TMP_V3_JSON}", ${RC_V3})
print("\n=== PERBANDINGAN TRIASE LANGSUNG (SEBELUM vs SESUDAH) ===")
print(f"{'Metrik':<22} | {'Baseline V0 (P=1)':<30} | {'Ring V3 (P=' + str(${NP_RANKS}) + ')':<30}")
print("-" * 88)
for name, a, b in zip(["Prediksi & Keyakinan", "Waktu Inferensi", "Throughput", "Peak RSS / Proses", "Delta RSS Aktivasi"], r0, r3):
    print(f"{name:<22} | {a:<30} | {b:<30}")
print("\n[Catatan] Alat bantu riset triase awal (PRJNA913438), bukan diagnosis klinis.")
PY
