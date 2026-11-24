#!/usr/bin/env bash
set -uo pipefail

FASTQ_FILE="${1:-data/fastq/184D_normal.fastq}"
SEQ_LEN="${2:-4096}"
NP_RANKS="${3:-2}"
ULIMIT_KB="${4:-0}"
VERSIONS="${5:-v0,v1,v2,v3,v4}"
OMP_THREADS="${6:-2}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ATTN_BIN="${ROOT_DIR}/build/attn"
WEIGHTS_DIR="${ROOT_DIR}/models/breast_triage_d64"
cd "${ROOT_DIR}"

usage() {
  echo "Usage: bash scripts/demo_cli.sh [FASTQ_FILE] [SEQ_LEN] [MPI_RANKS] [ULIMIT_KB] [VERSIONS] [OMP_THREADS]"
  echo "  VERSIONS: comma-separated subset of v0,v1,v2,v3,v4 (default: all)"
}

if [[ ! -x "${ATTN_BIN}" ]]; then
  echo "[error] ${ATTN_BIN} belum dikompilasi. Jalankan 'make build/attn' terlebih dahulu." >&2
  exit 1
fi
if [[ ! -f "${FASTQ_FILE}" ]]; then
  echo "[error] FASTQ tidak ditemukan: ${FASTQ_FILE}" >&2
  exit 1
fi
if [[ ! -d "${WEIGHTS_DIR}" ]]; then
  echo "[error] Direktori bobot model tidak ditemukan: ${WEIGHTS_DIR}" >&2
  exit 1
fi
if ! [[ "${SEQ_LEN}" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "${NP_RANKS}" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "${OMP_THREADS}" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "${ULIMIT_KB}" =~ ^[0-9]+$ ]]; then
  echo "[error] SEQ_LEN, MPI_RANKS, OMP_THREADS harus positif; ULIMIT_KB harus >= 0." >&2
  exit 1
fi

IFS=',' read -r -a VERSION_LIST <<< "${VERSIONS}"
if [[ "${#VERSION_LIST[@]}" -eq 0 ]]; then
  echo "[error] Daftar versi kosong." >&2
  usage >&2
  exit 1
fi
NEEDS_MPI=0
for version in "${VERSION_LIST[@]}"; do
  case "${version}" in
    v0|v1) ;;
    v2) ;;
    v3) NEEDS_MPI=1 ;;
    v4) NEEDS_MPI=1 ;;
    *)
      echo "[error] Versi tidak dikenal: ${version}. Pilih dari v0,v1,v2,v3,v4." >&2
      usage >&2
      exit 1
      ;;
  esac
done
if [[ "${NEEDS_MPI}" -eq 1 ]] && ! command -v mpirun >/dev/null 2>&1; then
  echo "[error] mpirun tidak ditemukan. Pasang OpenMPI terlebih dahulu." >&2
  exit 1
fi
if ! python3 -c 'import numpy' >/dev/null 2>&1; then
  echo "[error] NumPy belum terpasang untuk Python 3." >&2
  exit 1
fi

C_BLOCK=$(( SEQ_LEN / NP_RANKS ))
if [[ "${C_BLOCK}" -lt 1 ]]; then
  C_BLOCK="${SEQ_LEN}"
fi
if [[ "${NEEDS_MPI}" -eq 1 ]] && [[ $(( C_BLOCK * NP_RANKS )) -ne "${SEQ_LEN}" ]]; then
  echo "[error] SEQ_LEN harus habis dibagi MPI_RANKS untuk V3/V4." >&2
  exit 1
fi

export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS="${OMP_THREADS}"

TMP_BIN="$(mktemp "${TMPDIR:-/tmp}/dna_tokens_XXXXXX")"
JSON_PATHS=()
RC_CODES=()
cleanup() {
  rm -f "${TMP_BIN}" "${JSON_PATHS[@]}"
}
trap cleanup EXIT

python3 - "${FASTQ_FILE}" "${SEQ_LEN}" "${TMP_BIN}" "${ROOT_DIR}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[4])
sys.path.insert(0, str(root / "scripts"))
from download_and_tokenize import tokenize_fastq_file

fastq_path = root / sys.argv[1]
seq_len = int(sys.argv[2])
tokens_path = Path(sys.argv[3])
tokens = tokenize_fastq_file(fastq_path, max_tokens=seq_len)
tokens.tofile(tokens_path)
print(f"[demo] Sampel={fastq_path} | token tersedia={tokens.size} | s={seq_len}")
PY
if [[ "$?" -ne 0 ]]; then
  echo "[error] Gagal membaca FASTQ atau membuat token." >&2
  exit 1
fi

echo "[demo] Versi=${VERSION_LIST[*]} | MPI ranks=${NP_RANKS} | OpenMP threads=${OMP_THREADS} | ulimit_v=${ULIMIT_KB} KiB"

run_version() {
  local version="$1"
  local json_path="$2"
  local -a cmd

  cmd=("${ATTN_BIN}" --version "${version}" --weights "${WEIGHTS_DIR}" \
       --input "${TMP_BIN}" --s "${SEQ_LEN}" --d 64 --c "${C_BLOCK}" \
       --tile 64 --json-out "${json_path}")
  case "${version}" in
    v2) cmd+=(--threads "${OMP_THREADS}") ;;
    v3) cmd=(mpirun --bind-to core -np "${NP_RANKS}" "${cmd[@]}") ;;
    v4)
      cmd+=(--threads "${OMP_THREADS}")
      cmd=(mpirun --map-by "slot:PE=${OMP_THREADS}" --bind-to core \
           -np "${NP_RANKS}" "${cmd[@]}")
      ;;
  esac

  if [[ "${ULIMIT_KB}" -gt 0 ]]; then
    bash -c 'ulimit -v "$1" && shift && exec "$@"' _ "${ULIMIT_KB}" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

for version in "${VERSION_LIST[@]}"; do
  json_path="$(mktemp "${TMPDIR:-/tmp}/attn_${version}_XXXXXX")"
  JSON_PATHS+=("${json_path}")

  case "${version}" in
    v0) description="Naive baseline" ;;
    v1) description="Blockwise serial" ;;
    v2) description="Blockwise OpenMP (T=${OMP_THREADS})" ;;
    v3) description="Ring MPI (P=${NP_RANKS})" ;;
    v4) description="Hybrid MPI+OpenMP (P=${NP_RANKS}, T=${OMP_THREADS})" ;;
  esac

  echo "--- Menjalankan ${version}: ${description} ---"
  if run_version "${version}" "${json_path}"; then
    RC_CODES+=(0)
  else
    rc=$?
    RC_CODES+=("${rc}")
    echo "[warning] ${version} gagal (exit=${rc}); versi berikutnya tetap dijalankan." >&2
  fi
done

python3 - "${NP_RANKS}" "${OMP_THREADS}" "${VERSIONS}" "${JSON_PATHS[@]}" "${RC_CODES[*]}" <<'PY'
import json
import sys
from pathlib import Path

ranks = int(sys.argv[1])
threads = int(sys.argv[2])
versions = sys.argv[3].split(",")
count = len(versions)
paths = sys.argv[4 : 4 + count]
return_codes = [int(code) for code in sys.argv[4 + count].split()]

def version_label(version):
    if version == "v2":
        return f"V2 (1 rank,T={threads})"
    if version == "v3":
        return f"V3 (P={ranks})"
    if version == "v4":
        return f"V4 (P={ranks},T={threads})"
    return f"{version.upper()} (1 rank)"

columns = []
for version, path, return_code in zip(versions, paths, return_codes):
    file_path = Path(path)
    if return_code != 0 or not file_path.exists() or file_path.stat().st_size == 0:
        columns.append({
            "label": version_label(version),
            "prediction": f"GAGAL (exit={return_code})",
            "time": "-",
            "throughput": "-",
            "peak": "-",
            "delta": "-",
        })
        continue

    data = json.loads(file_path.read_text(encoding="utf-8"))
    columns.append({
        "label": version_label(version),
        "prediction": str(data["predicted_label"]),
        "time": f'{float(data["time_s"]):.5f} s',
        "throughput": f'{float(data["throughput_reads_per_s"]):.1f} read/s',
        "peak": f'{int(data["peak_rss_kb"])} KiB',
        "delta": f'{int(data["delta_rss_kb"])} KiB',
    })

row_names = [
    ("Prediksi", "prediction"),
    ("Waktu", "time"),
    ("Throughput", "throughput"),
    ("Peak RSS/proses", "peak"),
    ("Delta RSS", "delta"),
]
row_width = max(len(name) for name, _ in row_names)
col_width = max(24, *(len(column["label"]) for column in columns))
print("\n=== PERBANDINGAN INFERENSI DNA ===")
print(f"{'Metrik':<{row_width}} | " + " | ".join(f'{c["label"]:<{col_width}}' for c in columns))
print("-" * (row_width + 3 + len(columns) * (col_width + 3)))
for name, key in row_names:
    print(f"{name:<{row_width}} | " + " | ".join(f'{c[key]:<{col_width}}' for c in columns))
print("\nCatatan: V0/V1 berjalan serial, V2 memakai OpenMP, V3 memakai MPI, V4 memakai MPI+OpenMP.")
PY
