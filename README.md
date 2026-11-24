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
│   ├── demo_app.py       # Antarmuka web Streamlit (perbandingan V0 vs V3)
│   └── demo_cli.sh       # Demo CLI untuk V0, V1, V2, V3, dan V4
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

## 3. Platform dan Persiapan

### Linux / mesin HPC

Linux adalah lingkungan utama yang digunakan proyek. Siapkan compiler C++17, GNU Make, OpenMPI, OpenMP, dan Python 3. Untuk pipeline data/demo Python, paket yang digunakan adalah NumPy, PyTorch, Matplotlib, dan Streamlit.

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y build-essential g++ make openmpi-bin libopenmpi-dev \
  python3 python3-venv python3-pip

mkdir -p "$HOME/venvs"
python3 -m venv "$HOME/venvs/ring-attention-hpc"
source "$HOME/venvs/ring-attention-hpc/bin/activate"
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install numpy
# Opsional: untuk melatih ulang model, membuat grafik, atau menjalankan demo web
python -m pip install torch matplotlib streamlit
```

Simpan virtual environment di filesystem Linux WSL (`$HOME`), bukan di folder proyek yang berada di `/mnt/c` atau `/mnt/d`. `numpy` diperlukan untuk tokenisasi demo CLI; `torch` hanya diperlukan untuk melatih ulang model, sedangkan `matplotlib` dan `streamlit` hanya diperlukan untuk grafik dan demo web. Bobot model contoh sudah tersedia di `models/breast_triage_d64/`, sehingga pelatihan ulang tidak wajib. Pasang PyTorch sesuai petunjuk resminya bila membutuhkan build CPU/GPU tertentu.

Jika `python -m pip` gagal dengan pesan seperti `No module named pip._internal.utils`, virtual environment yang sedang aktif kemungkinan tidak lengkap. Jangan hapus environment lama dulu; buat environment Linux baru di home WSL:

```bash
deactivate
mkdir -p "$HOME/venvs"
python3 -m venv "$HOME/venvs/ring-attention-hpc-clean"
source "$HOME/venvs/ring-attention-hpc-clean/bin/activate"
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install numpy
```

### Windows

Proyek dapat dijalankan dari Windows **melalui WSL2 (Ubuntu)**. Jalankan perintah berikut di PowerShell sebagai Administrator, lalu buka terminal Ubuntu:

```powershell
wsl --install -d Ubuntu
```

Setelah Ubuntu terbuka, ikuti langkah Linux di atas. Simpan/clone repository di filesystem WSL (misalnya `~/ring-attention-hpc`), bukan di `/mnt/c` atau `/mnt/d`, agar build dan eksperimen I/O tidak terhambat akses filesystem lintas Windows/Linux. Virtual environment tetap disimpan di `$HOME/venvs/`, terpisah dari lokasi repository.

**Build native Windows (PowerShell/Visual Studio) belum didukung atau diuji.** Proyek bergantung pada Makefile dan skrip shell POSIX, OpenMPI, serta metrik memori Linux (`/proc` dan `sys/resource.h`). WSL2 memungkinkan menjalankan toolchain Linux, tetapi tetap menguji proses MPI pada satu mesin WSL; untuk eksperimen MPI multi-node gunakan cluster Linux.

---

## 4. Cara Build & Menjalankan Pengujian

Seluruh kompilasi menggunakan `-O3 -march=native -Wall -Wextra -std=c++17 -fopenmp` (tanpa `-ffast-math` demi menjaga presisi `exp()`).

```bash
# Kompilasi biner utama ./build/attn, seluruh unit test, dan mikro-benchmark
make setup
make all

# Jalankan seluruh unit test (GEMM, V2 OpenMP, V4 Hybrid, dan verifikasi CLI V0..V4)
make test-all
```

Contoh menjalankan `./build/attn` secara langsung:

```bash
OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \
  mpirun --bind-to core -np 4 ./build/attn \
  --version v3 --s 8192 --d 64 --c 2048 --tile 64 --seed 1 --verify \
  --csv results/raw/v3_s8192.csv
```

Pastikan `s = P * c` untuk V3/V4; contoh di atas memakai `s=8192`, `P=4`, dan `c=2048`. `--verify` mengukur galat numerik terhadap acuan FP64; ini adalah pemeriksaan implementasi numerik, bukan evaluasi biologis.

---

## 5. Perbandingan Performa Serial vs MPI

Untuk mengukur dampak distribusi MPI dengan perubahan algoritma seminimal mungkin, bandingkan V1 serial dengan V3 MPI memakai seed, panjang konteks, dimensi, ukuran blok, dan tile yang sama:

```bash
mkdir -p results/raw
OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./build/attn --version v1 --s 4096 --d 64 --c 1024 --tile 64 \
  --seed 1 --warmup 1 --reps 5 --csv results/raw/v1_serial.csv

OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \
  mpirun --bind-to core -np 4 ./build/attn \
  --version v3 --s 4096 --d 64 --c 1024 --tile 64 \
  --seed 1 --warmup 1 --reps 5 --csv results/raw/v3_mpi.csv
```

Speedup dihitung sebagai `waktu V1 serial / waktu V3 MPI` untuk konfigurasi yang sama. Jalankan pengukuran pada mesin dan kondisi yang sama, gunakan beberapa pengulangan, dan jangan bandingkan hasil WSL2 dengan Linux native seolah-olah keduanya lingkungan identik.

Untuk menjalankan seluruh rangkaian benchmark yang sudah disediakan:

```bash
python3 scripts/run_experiments.py --reps 5
python3 scripts/plot_results.py
```

Skrip menyimpan CSV di `results/raw/` dan grafik di `results/figures/`. Rangkaian ini mencakup beberapa eksperimen (termasuk scaling dan aplikasi V0 vs V3), bukan hanya satu perbandingan serial-vs-MPI. Lihat [docs/experiment-plan.md](docs/experiment-plan.md) untuk interpretasi metrik dan batas perbandingan.

---

## 6. Input DNA untuk Demo Aplikasi (Opsional)

Input DNA tidak diperlukan untuk menguji kernel attention atau menjalankan `scripts/run_experiments.py`; benchmark kernel menggunakan input numerik yang dibangkitkan oleh program. FASTQ/token dan model hanya diperlukan untuk demo pipeline inferensi aplikasi.

```bash
# Bangkitkan data sintetis offline dan tokenisasi 6-mer
python3 scripts/download_and_tokenize.py --generate-synthetic-fastq

# Opsional: latih ulang model dan ekspor bobot FP32
python3 scripts/train_offline_pytorch.py --data-dir data/tokens --out-dir models/breast_triage_d64

# Jalankan inferensi aplikasi menggunakan token dan bobot
OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \
  mpirun --bind-to core -np 4 ./build/attn \
  --version v3 --weights models/breast_triage_d64 --input data/tokens/184D_normal.bin \
  --s 4096 --d 64 --c 1024 --tile 64
```

Data sintetis sengaja disediakan untuk demo offline yang dapat diulang. Opsi `--download-sra` merupakan jalur tambahan dan belum menjadi jalur benchmark utama; sebelum dipakai, pastikan run accession, nama berkas FASTQ, label manifest, dan batas jumlah reads sudah dipetakan dengan benar. Untuk tujuan proyek ini, data sintetis atau input tetap yang sama sudah cukup untuk membandingkan kinerja komputasi.

---

## 7. Ringkasan Hasil Eksperimen Tersimpan (Intel i7-8665U, 4 Core Fisik)

- **Kebenaran Numerik (`V0..V4` vs `V0-FP64`):** seluruh versi lolos target `< 1e-5` dengan galat relatif L2 **`3.24e-07`–`3.31e-07`** ($\mathcal{N}(0, 1)$) dan **`1.225e-06`–`1.227e-06`** pada uji stres logit $\mathcal{N}(0, 10^2)$.
- **Skala Memori per Proses:** `delta_rss` pada **V0** tumbuh kuadratik dari `5.39 MiB` ($s=1024$) ke **`270.4 MiB`** ($s=8192$), sedangkan **V3** ($c=2048$ tetap) konstan di **`4.29–4.59 MiB`** ($s=2048\dots 8192$).
- **Strong Scaling ($s=4096, d=64, P=4$):** V2 mencapai speedup **`4.194x`** (`64.82 GFLOP/s`), V3 mencapai **`4.098x`** (`60.31 GFLOP/s`).
- **Overlap (`V3, P=4, c=2048`):** rasio overlap $\eta = \mathbf{0.811}$ (`T_blk = 0.27936 s`, `T_nb = 0.27440 s`, `T_comp = 0.27324 s`).
- **Demo `ulimit -v 1200000` ($s=16384$):** **V0 gagal (`std::bad_alloc`, exit 2)**, sedangkan **V3 ($P=4, c=4096$) sukses** dengan `delta_rss = 11.3 MiB`.

---

## 8. Menjalankan Aplikasi Demo Langsung

**Mode Web Interaktif (Streamlit):**

```bash
streamlit run scripts/demo_app.py
```

**Mode Cadangan CLI Satu Perintah:**

```bash
# Jalankan semua versi (s=4096, MPI P=2, OpenMP T=2)
bash scripts/demo_cli.sh data/fastq/184D_normal.fastq 4096 2 0 v0,v1,v2,v3,v4 2

# Pilih subset, misalnya hanya baseline serial dan Ring MPI
bash scripts/demo_cli.sh data/fastq/184D_normal.fastq 4096 4 0 v1,v3

# Simulasi batas memori ulimit -v 1200000 KiB
bash scripts/demo_cli.sh data/fastq/184AA2_aig.fastq 16384 4 1200000 v0,v1,v2,v3,v4
```

Argumen CLI berurutan: `FASTQ_FILE SEQ_LEN MPI_RANKS ULIMIT_KB VERSIONS OMP_THREADS`. `VERSIONS` default-nya semua versi; `OMP_THREADS` dipakai oleh V2 dan V4. V0/V1 berjalan serial, V2 memakai OpenMP, V3 memakai MPI, dan V4 memakai MPI+OpenMP. Untuk V4, siapkan kira-kira `MPI_RANKS × OMP_THREADS` core agar thread tidak oversubscribe. Demo web Streamlit tetap berfokus pada V0 vs V3 sebagai perbandingan baseline naive dengan versi Ring MPI.

---

## 9. Pembagian Tugas

| Anggota | Tanggung Jawab |
|---|---|
| Dhimas Early Oceandy | V0, V1, generator data, dan validasi numerik |
| Mahardika Ramadhana | Kernel tiled, V2 OpenMP, dan V4 opsional |
| Thufail Bahir Al Bariq | V3 Ring MPI dan koordinasi repository |
| Kevin Febriano | Benchmark, eksperimen, pengukuran memori, dan analisis |

---

## 10. Referensi

1. Liu, H., Zaharia, M., & Abbeel, P. (2023). *Ring Attention with Blockwise Transformers for Near-Infinite Context*. arXiv:2310.01889.
2. Liu, H., & Abbeel, P. (2023). *Blockwise Parallel Transformer for Large Context Models*.
3. Dao, T., et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*.
