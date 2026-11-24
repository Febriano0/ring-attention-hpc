# Ringkasan Hasil Eksperimen Ring Attention HPC

- Laju GEMM tile 64x64x64 (`tiled_avx2_fma`): **38.11 GFLOP/s** (1 core)


## 1. Verifikasi Numerik terhadap V0-FP64 (Target `rel_err < 1e-5`)

| Versi | P | Threads | Skenario | Max Abs Err | Rel L2 Err | Status |
|---|---:|---:|---|---:|---:|---|
| v0 | 1 | 1 | standard_normal | 1.991e-07 | 3.241e-07 | PASS |
| v1 | 1 | 1 | standard_normal | 2.374e-07 | 3.310e-07 | PASS |
| v2 | 1 | 4 | standard_normal | 2.374e-07 | 3.310e-07 | PASS |
| v3 | 4 | 1 | standard_normal | 2.140e-07 | 3.301e-07 | PASS |
| v4 | 2 | 2 | standard_normal | 2.374e-07 | 3.305e-07 | PASS |
| v0 | 1 | 1 | stress_logits_std10 | 1.529e-05 | 1.226e-06 | PASS |
| v1 | 1 | 1 | stress_logits_std10 | 1.514e-05 | 1.226e-06 | PASS |
| v2 | 1 | 4 | stress_logits_std10 | 1.514e-05 | 1.226e-06 | PASS |
| v3 | 4 | 1 | stress_logits_std10 | 1.505e-05 | 1.225e-06 | PASS |
| v4 | 2 | 2 | stress_logits_std10 | 1.514e-05 | 1.227e-06 | PASS |

## 2. Profil Memori per Proses (V0 vs V3 dengan `c` Tetap)

| Seri | P | c | s | Peak RSS (KiB) | Delta RSS (KiB) | Alloc Teoretis (KiB) |
|---|---:|---:|---:|---:|---:|---:|
| v0_naive | 1 | 1024 | 1024 | 178112 | 5388 | 5120.00 |
| v0_naive | 1 | 2048 | 2048 | 191332 | 18756 | 18432.00 |
| v0_naive | 1 | 4096 | 4096 | 242436 | 69848 | 69632.00 |
| v0_naive | 1 | 8192 | 8192 | 443212 | 270436 | 270336.00 |
| v3_c1024 | 1 | 1024 | 1024 | 174872 | 2212 | 1560.00 |
| v3_c1024 | 2 | 1024 | 2048 | 175316 | 2516 | 1560.00 |
| v3_c1024 | 4 | 1024 | 4096 | 175280 | 2428 | 1560.00 |
| v3_c2048 | 1 | 2048 | 2048 | 176832 | 4288 | 3104.00 |
| v3_c2048 | 2 | 2048 | 4096 | 177336 | 4588 | 3104.00 |
| v3_c2048 | 4 | 2048 | 8192 | 177232 | 4400 | 3104.00 |

## 3. Strong Scaling (`s = 4096, d = 64`)

| Versi | P | c | Median Time (s) | GFLOP/s | Speedup | Efisiensi |
|---|---:|---:|---:|---:|---:|---:|
| v2 | 1 | 256 | 0.27789 | 15.455 | 1.000x | 100.0% |
| v2 | 2 | 256 | 0.12888 | 33.324 | 2.156x | 107.8% |
| v2 | 4 | 256 | 0.06626 | 64.819 | 4.194x | 104.8% |
| v3 | 1 | 4096 | 0.29181 | 14.718 | 1.000x | 100.0% |
| v3 | 2 | 2048 | 0.13786 | 31.155 | 2.117x | 105.8% |
| v3 | 4 | 1024 | 0.07121 | 60.314 | 4.098x | 102.5% |

## 4. Weak Scaling (`c = 1024` tetap, `s = P * c`)

*Catatan: Pada attention dengan `c` tetap, kerja per proses tumbuh linear `O(P * c^2 * d)` sementara memori aktivasi tetap `O(c * d)`.*

| Versi | P | s | Median Time (s) | Total GFLOP/s | GFLOP/s per Proses | Work-Scaled Eff |
|---|---:|---:|---:|---:|---:|---:|
| v2 | 1 | 1024 | 0.01719 | 15.615 | 15.615 | 100.0% |
| v2 | 2 | 2048 | 0.03326 | 32.287 | 16.144 | 103.4% |
| v2 | 4 | 4096 | 0.06610 | 64.974 | 16.244 | 104.0% |
| v3 | 1 | 1024 | 0.01700 | 15.788 | 15.788 | 100.0% |
| v3 | 2 | 2048 | 0.06011 | 17.862 | 8.931 | 56.6% |
| v3 | 4 | 4096 | 0.06715 | 63.963 | 15.991 | 101.3% |

## 5. Efektivitas Overlap (`P = 4, d = 64`) vs Model Analitik `c >= 2F/B`

| c | s | T_blk (s) | T_nb (s) | T_comp (s) | T_wait (s) | Overlap eta | B (GB/s) | c_min (GEMM) | c_min (Attn) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 512 | 0.00329 | 0.00464 | 0.00277 | 0.00090 | -2.628 | 1.992 | 38.3 | 6.1 |
| 256 | 1024 | 0.01171 | 0.01149 | 0.01117 | 0.00029 | 0.411 | 2.228 | 34.2 | 5.4 |
| 512 | 2048 | 0.04474 | 0.04451 | 0.04016 | 0.00026 | 0.050 | 2.393 | 31.9 | 5.6 |
| 1024 | 4096 | 0.06809 | 0.07085 | 0.06681 | 0.00659 | -2.151 | 2.889 | 26.4 | 11.1 |
| 2048 | 8192 | 0.27936 | 0.27440 | 0.27324 | 0.00485 | 0.811 | 2.942 | 25.9 | 10.7 |

## 6. Aplikasi Triase DNA PRJNA913438: Sebelum (`V0`) vs Sesudah (`V3, P = 4`)

| Sampel | s | Versi | Prediksi | Keyakinan | Latensi (s) | Throughput (read/s) | Delta RSS (KiB) |
|---|---:|---|---|---:|---:|---:|---:|
| 184D_normal | 1024 | v0 (P=1) | Normal-like | 100.00% | 0.04712 | 869.24 | 7028 |
| 184D_normal | 1024 | v3 (P=4) | Normal-like | 100.00% | 0.01283 | 3191.76 | 2088 |
| 184D_normal | 2048 | v0 (P=1) | Normal-like | 100.00% | 0.14537 | 563.51 | 20792 |
| 184D_normal | 2048 | v3 (P=4) | Normal-like | 100.00% | 0.04595 | 1782.64 | 2764 |
| 184D_normal | 4096 | v0 (P=1) | Normal-like | 100.00% | 0.37195 | 440.49 | 73076 |
| 184D_normal | 4096 | v3 (P=4) | Normal-like | 100.00% | 0.11012 | 1487.88 | 4076 |
| 184D_normal | 8192 | v0 (P=1) | Normal-like | 100.00% | 1.21938 | 268.73 | 275736 |
| 184D_normal | 8192 | v3 (P=4) | Normal-like | 100.00% | 0.41477 | 790.04 | 6528 |
| 184AA2_aig | 1024 | v0 (P=1) | Progresi Karsinogenesis | 100.00% | 0.04322 | 947.81 | 6952 |
| 184AA2_aig | 1024 | v3 (P=4) | Progresi Karsinogenesis | 100.00% | 0.01261 | 3248.73 | 2080 |
| 184AA2_aig | 2048 | v0 (P=1) | Progresi Karsinogenesis | 100.00% | 0.14027 | 584.03 | 20872 |
| 184AA2_aig | 2048 | v3 (P=4) | Progresi Karsinogenesis | 100.00% | 0.04664 | 1756.45 | 2732 |
| 184AA2_aig | 4096 | v0 (P=1) | Progresi Karsinogenesis | 100.00% | 0.39922 | 410.40 | 72964 |
| 184AA2_aig | 4096 | v3 (P=4) | Progresi Karsinogenesis | 100.00% | 0.17816 | 919.62 | 3996 |
| 184AA2_aig | 8192 | v0 (P=1) | Progresi Karsinogenesis | 100.00% | 1.34374 | 243.86 | 275864 |
| 184AA2_aig | 8192 | v3 (P=4) | Progresi Karsinogenesis | 100.00% | 0.42438 | 772.15 | 6496 |

## 7. Uji Batas Memori `ulimit -v 1200000` (`s = 16384`)

| Versi | P | s | Status | Exit Code | Delta RSS (KiB) | Keterangan |
|---|---:|---:|---|---:|---:|---|
| v0 | 1 | 16384 | OOM_FAILED | 2 | -1 | std::bad_alloc on s x s score matrix |
| v3 | 4 | 16384 | SUCCESS | 0 | 11616 | Allocated O(cd) per rank |
