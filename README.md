# Ring Attention HPC

**Implementasi dan Evaluasi Ring Attention dengan Blockwise Transformer pada Mesin Multi-core Menggunakan OpenMP dan MPI**

Proyek ini mengimplementasikan dan mengevaluasi empat versi inti forward pass attention (**V0–V3**) serta ekstensi opsional hibrida (**V4**) dalam C++17, OpenMP, dan OpenMPI, lalu menerapkannya pada studi kasus **triase cepat sampel sekuens DNA jaringan payudara (BioProject PRJNA913438)**.

---

## 1. Status Implementasi

| Versi | Implementasi | Status | Lokasi |
|---|---|---|---|
| V0 | Naive Attention (FP64 acuan & FP32 baseline $O(s^2)$) | **Implemented & Verified** | `src/v0_naive/` |
| V1 | Blockwise Attention + Tiled Online Softmax (Serial) | **Implemented & Verified** | `src/v1_blockwise/` |
| V2 | Blockwise Attention + OpenMP (Outer Q-Block) | **Implemented & Verified** | `src/v2_openmp/` |
| V3 | Ring Attention + MPI (`Isend/Irecv`, `Sendrecv`, `no-comm`) | **Implemented & Verified** | `src/v3_ring_mpi/` |
| V4 | Hybrid MPI + OpenMP (`MPI_THREAD_FUNNELED`, Optional) | **Implemented & Verified** | `src/v4_hybrid/` |
| Shared | Cache-Blocked AVX2/FMA GEMM + Online Softmax + C++ Inference | **Implemented & Verified** | `src/common/`, `src/inference/` |

---

## 2. Struktur Proyek

```text
ring-attention-hpc/
├── benchmarks/
│   ├── gemm/             # Mikro-benchmark GEMM (F) dan strong scaling V2
│   └── mpi/              # Mikro-benchmark bandwidth ping-pong MPI (B)
├── data/
│   ├── fastq/            # Sampel FASTQ siap demo (184D, 184A1, 184B5, 184AA2)
│   └── tokens/           # Token biner 6-mer (int32) + manifest.json
├── docs/
│   ├── architecture.md
│   ├── common-api.md
│   └── experiment-plan.md
├── models/
│   └── breast_triage_d64/ # Bobot Transformer FP32 hasil pelatihan PyTorch offline
├── results/
│   ├── figures/          # Grafik publikasi PNG (300 DPI)
│   ├── raw/              # Data mentah CSV hasil eksperimen
│   └── summary/          # Ringkasan tabel markdown (experiment_summary.md)
├── scripts/
│   ├── download_and_tokenize.py
│   ├── train_offline_pytorch.py
│   ├── run_experiments.py
│   ├── plot_results.py
│   ├── demo_app.py       # Antarmuka web Streamlit (V0 vs V3 real-time)
│   └── demo_cli.sh       # Fallback demo CLI satu perintah
├── src/
│   ├── common/           # Kernel GEMM ter-tile, Online Softmax, utilitas memori
│   ├── inference/        # Pipeline inferensi DNA end-to-end
│   ├── v0_naive/
│   ├── v1_blockwise/
│   ├── v2_openmp/
│   ├── v3_ring_mpi/
│   ├── v4_hybrid/
│   └── main.cpp          # Biner CLI terpadu (./build/attn)
├── tests/                # Unit test GEMM, V2, dan V4 terhadap acuan FP64
├── Makefile
└── README.md
```

---

## 3. Cara Build & Menjalankan Pengujian (Zsh)

Seluruh kompilasi menggunakan `-O3 -march=native -Wall -Wextra -std=c++17 -fopenmp` (tanpa `-ffast-math` demi menjaga presisi `exp()`).

```zsh
# Kompilasi biner utama ./build/attn, seluruh unit test, dan mikro-benchmark
make clean && make setup && make all

# Jalankan seluruh unit test (GEMM, V2 OpenMP, V4 Hybrid, dan verifikasi CLI V0..V4)
make test-all
```

Menjalankan `./build/attn` secara langsung sesuai antarmuka Proposal Subbab 6.4:

```zsh
OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \
  mpirun --bind-to core -np 4 ./build/attn \
  --version v3 --s 8192 --d 64 --c 2048 --tile 64 --seed 1 --verify \
  --csv results/raw/v3_s8192.csv
```

---

## 4. Pipeline Data DNA (PRJNA913438) & Pelatihan Offline PyTorch

```zsh
# 1. Bangkitkan sampel FASTQ 150 bp realistis & tokenisasi 6-mer (kosakata 4098)
#    (atau gunakan --download-sra jika SRA Toolkit tersedia)
python3 scripts/download_and_tokenize.py --generate-synthetic-fastq

# 2. Latih model Transformer minimal 1-layer di CPU dan ekspor bobot FP32
python3 scripts/train_offline_pytorch.py --data-dir data/tokens --out-dir models/breast_triage_d64

# 3. Uji inferensi langsung dari C++/MPI
OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \
  mpirun --bind-to core -np 4 ./build/attn \
  --version v3 --weights models/breast_triage_d64 --input data/tokens/184D_normal.bin \
  --s 4096 --d 64 --c 1024 --tile 64
```

> **Catatan Keterbatasan Data:** Subset data PRJNA913438 hanya berasal dari satu sistem sel epitel payudara (1 lini normal `184D`, 7 lini progresi). Aplikasi ini berfungsi sebagai alat bantu triase riset untuk demonstrasi skalabilitas Ring Attention, **bukan alat diagnosis klinis**.

---

## 5. Menjalankan Eksperimen Penuh & Membuat Grafik

```zsh
# Menjalankan seluruh skenario Bagian 6 (>= 5 pengulangan per konfigurasi)
python3 scripts/run_experiments.py --reps 5

# Membuat 5 grafik PNG (300 DPI) di results/figures/
python3 scripts/plot_results.py
```

### Ringkasan Hasil Eksperimen Terverifikasi (Intel i7-8665U, 4 Core Fisik)
- **Kebenaran Numerik (`V0..V4` vs `V0-FP64`):** seluruh versi lolos target `< 1e-5` dengan galat relatif L2 **`3.24e-07`–`3.31e-07`** ($\mathcal{N}(0, 1)$) dan **`1.225e-06`–`1.227e-06`** pada uji stres logit $\mathcal{N}(0, 10^2)$.
- **Skala Memori per Proses:** `delta_rss` pada **V0** tumbuh kuadratik dari `5.39 MiB` ($s=1024$) ke **`270.4 MiB`** ($s=8192$), sedangkan **V3** ($c=2048$ tetap) konstan di **`4.29–4.59 MiB`** ($s=2048\dots 8192$).
- **Strong Scaling ($s=4096, d=64, P=4$):** V2 mencapai speedup **`4.194x`** (`64.82 GFLOP/s`), V3 mencapai **`4.098x`** (`60.31 GFLOP/s`).
- **Overlap (`V3, P=4, c=2048`):** rasio overlap $\eta = \mathbf{0.811}$ (`T_blk = 0.27936 s`, `T_nb = 0.27440 s`, `T_comp = 0.27324 s`).
- **Demo `ulimit -v 1200000` ($s=16384$):** **V0 gagal (`std::bad_alloc`, exit 2)**, sedangkan **V3 ($P=4, c=4096$) sukses** dengan `delta_rss = 11.3 MiB`.

---

## 6. Menjalankan Aplikasi Demo Langsung

**Mode Web Interaktif (Streamlit):**

```zsh
streamlit run scripts/demo_app.py
```

**Mode Cadangan CLI Satu Perintah:**

```zsh
# Perbandingan normal (s=4096, P=4)
./scripts/demo_cli.sh data/fastq/184D_normal.fastq 4096 4

# Simulasi batas memori ulimit -v 1200000 KiB pada s=16384 (V0 OOM vs V3 sukses)
./scripts/demo_cli.sh data/fastq/184AA2_aig.fastq 16384 4 1200000
```

---

## 7. Pembagian Tugas

| Anggota | Tanggung Jawab |
|---|---|
| Dhimas Early Oceandy | V0, V1, generator data, dan validasi numerik |
| Mahardika Ramadhana | Kernel tiled, V2 OpenMP, dan V4 opsional |
| Thufail Bahir Al Bariq | V3 Ring MPI dan koordinasi repository |
| Kevin Febriano | Benchmark, eksperimen, pengukuran memori, dan analisis |

---

## 8. Referensi

1. Liu, H., Zaharia, M., & Abbeel, P. (2023). *Ring Attention with Blockwise Transformers for Near-Infinite Context*. arXiv:2310.01889.
2. Liu, H., & Abbeel, P. (2023). *Blockwise Parallel Transformer for Large Context Models*.
3. Dao, T., et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*.
