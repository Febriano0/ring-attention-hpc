#!/usr/bin/env python3
"""
Pipeline pengunduhan / pembangkitan data FASTQ BioProject PRJNA913438
dan tokenisasi non-overlapping 6-mer (k = 6, kosakata 4098).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

BASE_TO_INT: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}
INT_TO_BASE = "ACGT"
UNK_TOKEN_ID = 4096
PAD_TOKEN_ID = 4097
VOCAB_SIZE = 4098
KMER_SIZE = 6

# Sampel dari Proposal Tabel 1 (BioProject PRJNA913438)
PRJNA913438_SAMPLES: List[Tuple[str, int, str]] = [
    ("184D_normal", 0, "Normal (pra-stasis)"),
    ("184A1_immortal", 1, "Immortal (progresi)"),
    ("184AA4_immortal", 1, "Immortal (progresi)"),
    ("184B5_immortal", 1, "Immortal (progresi)"),
    ("184AA2_aig", 1, "Immortal + AIG (progresi)"),
    ("184AA3_aig", 1, "Immortal + AIG (progresi)"),
    ("184B5ME_aig", 1, "Immortal + AIG (progresi)"),
    ("184FMY2_aig", 1, "Immortal + AIG (progresi)"),
]

# Motif 6-mer pembeda kelas yang disisipkan secara biologis realistis
# (digunakan hanya pada mode --generate-synthetic-fastq untuk demo offline).
NORMAL_MOTIFS = [
    "GAATTC", "AAGCTT", "GCATGC", "TCTAGA", "GTCGAC", "CCCGGG",
    "ACTGAC", "TGTACA", "CATATG", "GGATCC", "CTCGAG", "AGATCA",
]
PROGRESSION_MOTIFS = [
    "CGGCGG", "CCGCCG", "GAGGAG", "CTCCTC", "TTTTAA", "AAAATT",
    "ACGACG", "CGTCGT", "TGGCCA", "CGTACG", "GCGATG", "TAGCTA",
]


def encode_6mer(kmer: str) -> int:
    """Encode a 6-mer string into [0..4095], or UNK_TOKEN_ID if non-ACGT appears."""
    if len(kmer) != KMER_SIZE:
        return UNK_TOKEN_ID
    val = 0
    for ch in kmer.upper():
        code = BASE_TO_INT.get(ch)
        if code is None:
            return UNK_TOKEN_ID
        val = (val << 2) | code
    return val


def tokenize_sequence(seq: str) -> List[int]:
    """Tokenize one nucleotide read using non-overlapping 6-mers."""
    n_tokens = len(seq) // KMER_SIZE
    return [encode_6mer(seq[i * KMER_SIZE : (i + 1) * KMER_SIZE]) for i in range(n_tokens)]


def tokenize_fastq_file(fastq_path: Path, max_tokens: int | None = None) -> np.ndarray:
    """Parse FASTQ and return little-endian int32 numpy array of 6-mer tokens."""
    tokens: List[int] = []
    with open(fastq_path, "r", encoding="utf-8") as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            seq = fh.readline().strip()
            plus = fh.readline()
            qual = fh.readline()
            if not plus or not qual:
                break
            tokens.extend(tokenize_sequence(seq))
            if max_tokens is not None and len(tokens) >= max_tokens:
                tokens = tokens[:max_tokens]
                break
    return np.asarray(tokens, dtype="<i4")


def generate_synthetic_fastq(
    fastq_dir: Path,
    num_reads: int = 1500,
    read_len: int = 150,
    seed: int = 2026,
) -> List[Dict[str, object]]:
    """Generate realistic 150 bp Illumina-style synthetic FASTQ datasets for offline triage."""
    fastq_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Frekuensi nukleotida mendekati genom manusia (~41% GC)
    bg_probs_normal = np.array([0.295, 0.205, 0.205, 0.295])
    bg_probs_prog = np.array([0.285, 0.215, 0.215, 0.285])

    kmers_per_read = read_len // KMER_SIZE
    records: List[Dict[str, object]] = []

    for idx, (sample_name, label, desc) in enumerate(PRJNA913438_SAMPLES):
        sample_rng = np.random.default_rng(seed + idx * 101)
        out_path = fastq_dir / f"{sample_name}.fastq"
        motif_pool = NORMAL_MOTIFS if label == 0 else PROGRESSION_MOTIFS
        bg_probs = bg_probs_normal if label == 0 else bg_probs_prog

        # Variasi ringan antar-cell-line agar skenario leave-one-line-out tetap realistis
        line_bias = 0.04 * (idx % 3)
        motif_prob = 0.28 - line_bias if label == 1 else 0.28
        line_specific_motif = PROGRESSION_MOTIFS[(idx * 3) % len(PROGRESSION_MOTIFS)]

        with open(out_path, "w", encoding="utf-8") as fh:
            for r_idx in range(num_reads):
                kmers: List[str] = []
                for _ in range(kmers_per_read):
                    u = sample_rng.random()
                    if u < motif_prob:
                        kmers.append(motif_pool[int(sample_rng.integers(0, len(motif_pool)))])
                    elif label == 1 and u < motif_prob + 0.06:
                        kmers.append(line_specific_motif)
                    else:
                        bases = sample_rng.choice(list(INT_TO_BASE), size=KMER_SIZE, p=bg_probs)
                        kmers.append("".join(bases))
                seq = "".join(kmers)
                # Mutasi titik sesekali (~0.5%) untuk mensimulasikan kesalahan base-calling Illumina
                if sample_rng.random() < 0.08:
                    pos = int(sample_rng.integers(0, read_len))
                    seq = seq[:pos] + rng.choice(["A", "C", "G", "T"]) + seq[pos + 1 :]

                quals = "".join(sample_rng.choice(["F", "G", "H", "I"], size=read_len, p=[0.1, 0.2, 0.3, 0.4]))
                fh.write(f"@PRJNA913438.{sample_name}.{r_idx + 1} length={read_len}\n")
                fh.write(f"{seq}\n+\n{quals}\n")

        records.append({
            "sample": sample_name,
            "label": label,
            "group": desc,
            "fastq_path": str(out_path),
            "num_reads": num_reads,
            "read_len": read_len,
            "source": "synthetic_prjna913438_emulation",
        })
    return records


def download_sra_subsets(sra_dir: Path, fastq_dir: Path, max_reads: int) -> bool:
    """Attempt downloading real subset FASTQ reads via SRA Toolkit if installed."""
    prefetch = shutil.which("prefetch")
    fasterq = shutil.which("fasterq-dump")
    if not prefetch or not fasterq:
        print(
            "[info] SRA Toolkit ('prefetch' / 'fasterq-dump') tidak ditemukan di PATH.\n"
            "       Untuk data BioProject PRJNA913438 offline tanpa unduhan besar,\n"
            "       jalankan: python3 scripts/download_and_tokenize.py --generate-synthetic-fastq",
            file=sys.stderr,
        )
        return False

    sra_dir.mkdir(parents=True, exist_ok=True)
    fastq_dir.mkdir(parents=True, exist_ok=True)
    # Contoh run accessions dari BioProject PRJNA913438 (dapat disesuaikan dengan SraRunTable)
    sra_runs = {
        "184D_normal": "SRR22750298",
        "184A1_immortal": "SRR22750299",
        "184B5_immortal": "SRR22750300",
        "184AA2_aig": "SRR22750301",
    }
    for name, srr in sra_runs.items():
        print(f"[sra] Mengunduh subset {srr} ({name}) maks {max_reads} reads...")
        subprocess.run([prefetch, srr, "-O", str(sra_dir)], check=True)
        subprocess.run(
            [fasterq, str(sra_dir / srr), "-O", str(fastq_dir), "--split-files", "-N", str(max_reads)],
            check=True,
        )
    return True


def tokenize_all(fastq_dir: Path, tokens_dir: Path) -> None:
    """Tokenize all FASTQ files in fastq_dir into int32 .bin files and manifest.json."""
    tokens_dir.mkdir(parents=True, exist_ok=True)
    label_map = {name: (label, desc) for name, label, desc in PRJNA913438_SAMPLES}

    manifest: List[Dict[str, object]] = []
    for fq_path in sorted(fastq_dir.glob("*.fastq")):
        stem = fq_path.stem
        arr = tokenize_fastq_file(fq_path)
        bin_path = tokens_dir / f"{stem}.bin"
        arr.tofile(bin_path)
        label, desc = label_map.get(stem, (-1, "Unknown"))
        manifest.append({
            "sample": stem,
            "label": label,
            "group": desc,
            "fastq_path": str(fq_path),
            "tokens_path": str(bin_path),
            "num_tokens": int(arr.size),
            "vocab_size": VOCAB_SIZE,
        })
        print(f"[tokenize] {fq_path.name} -> {bin_path} ({arr.size} tokens, label={label})")

    manifest_path = tokens_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[done] Manifest tersimpan di {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download/generate FASTQ PRJNA913438 & tokenize 6-mer.")
    parser.add_argument("--download-sra", action="store_true", help="Unduh subset dari NCBI SRA (butuh SRA Toolkit)")
    parser.add_argument("--generate-synthetic-fastq", action="store_true", help="Bangkitkan FASTQ sintetis offline realistis")
    parser.add_argument("--fastq-dir", type=Path, default=Path("data/fastq"))
    parser.add_argument("--tokens-dir", type=Path, default=Path("data/tokens"))
    parser.add_argument("--sra-dir", type=Path, default=Path("data/sra"))
    parser.add_argument("--num-reads", type=int, default=1500, help="Jumlah read per sampel (default: 1500 = 37500 token)")
    parser.add_argument("--max-sra-reads", type=int, default=100000)
    args = parser.parse_args()

    if args.download_sra:
        ok = download_sra_subsets(args.sra_dir, args.fastq_dir, args.max_sra_reads)
        if not ok:
            return 1
    elif args.generate_synthetic_fastq:
        generate_synthetic_fastq(args.fastq_dir, num_reads=args.num_reads)

    if not any(args.fastq_dir.glob("*.fastq")):
        print("[error] Tidak ada file .fastq di data/fastq/. Gunakan --generate-synthetic-fastq.", file=sys.stderr)
        return 1

    tokenize_all(args.fastq_dir, args.tokens_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
