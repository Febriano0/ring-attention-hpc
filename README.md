# Ring Attention HPC

**Implementasi dan Evaluasi Ring Attention dengan Blockwise Transformer pada Mesin Multi-core Menggunakan OpenMP dan MPI**

Proyek ini merupakan implementasi dan evaluasi performa Ring Attention dengan pendekatan Blockwise Transformer pada mesin multi-core menggunakan C/C++, OpenMP, dan MPI.

## 1. Deskripsi Proyek

Self-attention standar pada Transformer membutuhkan memori yang tumbuh secara kuadratik terhadap panjang sekuens. Proyek ini mempelajari implementasi Blockwise Attention dan Ring Attention untuk mengurangi kebutuhan memori per proses melalui pembagian blok dan komunikasi antarproses.

Setiap proses MPI merepresentasikan satu perangkat yang menjadi bagian dari topologi ring. Blok Key dan Value dipertukarkan antarproses, sementara komputasi attention dilakukan menggunakan mekanisme online softmax.

Eksperimen dilakukan pada satu mesin multi-core dengan komunikasi antarproses menggunakan MPI.

## 2. Tujuan

1. Mengimplementasikan attention naif dan blockwise dengan online softmax.
2. Mengimplementasikan paralelisasi menggunakan OpenMP.
3. Mengimplementasikan Ring Attention menggunakan MPI.
4. Memverifikasi kebenaran numerik hasil implementasi.
5. Mengukur penggunaan memori, waktu eksekusi, speedup, dan efisiensi paralel.
6. Mengevaluasi efektivitas overlap komunikasi dan komputasi.

## 3. Implementasi

| Versi | Implementasi | Status |
|---|---|---|
| V0 | Naive Attention | Planned |
| V1 | Blockwise Attention + Online Softmax | Planned |
| V2 | Blockwise Attention + OpenMP | Planned |
| V3 | Ring Attention + MPI | Planned |
| V4 | Hybrid MPI + OpenMP | Optional |

V0 sampai V3 merupakan implementasi utama. V4 merupakan pengembangan opsional.

## 4. Teknologi

- C/C++: Implementasi algoritma.
- OpenMP: Shared-memory parallelism.
- MPI (OpenMPI/MPICH): Distributed-process communication.
- Python: Analisis dan visualisasi hasil eksperimen.
- NumPy dan Matplotlib: Pengolahan data dan grafik.
- Git dan GitHub: Version control dan kolaborasi.

## 5. Struktur Proyek

```text
ring-attention-hpc/
├── docs/
├── src/
│   ├── common/
│   ├── v0_naive/
│   ├── v1_blockwise/
│   ├── v2_openmp/
│   ├── v3_ring_mpi/
│   └── v4_hybrid/
├── benchmarks/
│   ├── gemm/
│   └── mpi/
├── scripts/
├── tests/
├── results/
│   ├── raw/
│   └── figures/
├── README.md
├── Makefile
└── .gitignore
```

## 6. Pembagian Tugas

| Anggota | Tanggung Jawab |
|---|---|
| Dhimas Early Oceandy | V0, V1, generator data, dan validasi numerik |
| Mahardika Ramadhana | Kernel tiled, V2 OpenMP, dan V4 opsional |
| Thufail Bahir Al Bariq | V3 Ring MPI dan koordinasi repository |
| Kevin Febriano | Benchmark, eksperimen, pengukuran memori, dan analisis |

## 7. Rancangan Eksperimen

Parameter eksperimen utama:

- Dimensi head: 64 (opsional 128).
- Ukuran blok: 128, 256, 512, 1024.
- Jumlah proses/thread: 1, 2, 4, dan 8 jika core fisik mencukupi.
- Presisi utama: FP32.
- Acuan kebenaran: FP64.
- Pengulangan: minimal 5 kali setiap konfigurasi.

Metrik evaluasi:

1. Galat numerik terhadap acuan FP64.
2. Peak memory per process.
3. Execution time.
4. Parallel speedup dan efficiency.
5. Strong scaling dan weak scaling.
6. Communication-computation overlap effectiveness.

## 8. Lingkungan Pengembangan

Proyek dirancang untuk berjalan pada Linux atau Windows menggunakan WSL2.

Kebutuhan perangkat lunak:

- GCC/G++
- OpenMP
- OpenMPI atau MPICH
- Python 3
- NumPy
- Matplotlib

## 9. Status Pengembangan

Proyek saat ini berada pada tahap persiapan repository dan environment pengembangan.

Implementasi dan eksperimen akan dilakukan secara bertahap berdasarkan prioritas V0 sampai V3.

## 10. Referensi

1. Liu, H., Zaharia, M., & Abbeel, P. (2023). *Ring Attention with Blockwise Transformers for Near-Infinite Context*. arXiv:2310.01889.
2. Liu, H., & Abbeel, P. (2023). *Blockwise Parallel Transformer for Large Context Models*.
3. Dao, T., et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*.

---

**Universitas Gadjah Mada**

High Performance Computing Project
