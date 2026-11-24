# Kontrak API Bersama (`src/common/`)

Dokumen ini menjelaskan antarmuka bersama di direktori `src/common/` yang menjadi fondasi kernel komputasi dan Online Softmax untuk seluruh versi implementasi (**V0**, **V1**, **V2**, **V3**, dan **V4**). Dengan memakai header yang sama, laju komputasi $F$ tetap seragam di semua versi dan integrasi antar-anggota tim bebas konflik.

---

## 1. Konvensi Memori & Kompilasi

- **Layout matriks:** *Row-major* kontigu (`float` FP32 untuk V0–V4, `double` FP64 khusus acuan V0-FP64).
- **Faktor skala attention:** $\text{scale} = \frac{1}{\sqrt{d}}$.
- **Flag kompilasi wajib (Proposal Subbab 6.5):**
  - Tambahkan `-Isrc -O3 -march=native -Wall -Wextra -std=c++17` (dan `-fopenmp` untuk V2/V4).
  - **Dilarang memakai `-ffast-math`** karena mengurangi akurasi `exp()` pada Online Softmax.
- **Integrasi OpenBLAS opsional:** Kompilasi dengan `-DUSE_OPENBLAS -lopenblas` dan jalankan dengan `OPENBLAS_NUM_THREADS=1` agar tidak terjadi *thread oversubscription*.

---

## 2. Kernel GEMM Ter-tile (`src/common/gemm.hpp`)

```cpp
namespace ring_attn {

const char* gemm_backend_name() noexcept;

// C[M x N] = alpha * A[M x K] * B^T[K x N] + beta * C[M x N]
// B disimpan [N x K] dengan row stride ldb.
void gemm_nt_fp32(int M, int N, int K,
                  float alpha,
                  const float* A, int lda,
                  const float* B, int ldb,
                  float beta,
                  float* C, int ldc);

// C[M x N] = alpha * A[M x K] * B[K x N] + beta * C[M x N]
// B disimpan [K x N] dengan row stride ldb.
void gemm_nn_fp32(int M, int N, int K,
                  float alpha,
                  const float* A, int lda,
                  const float* B, int ldb,
                  float beta,
                  float* C, int ldc);

} // namespace ring_attn
```

### Detail Penting
- **Semantik `beta == 0.0f`:** Aman dari `NaN`/`Inf` sisa pada buffer `C` — implementasi tidak membaca `C` sama sekali saat `beta == 0.0f`.
- **Akumulasi (`beta == 1.0f`):** Gunakan `beta = 0.0f` saat menghitung skor tile $S = \frac{1}{\sqrt{d}} Q K^T$ (`gemm_nt_fp32`), dan `beta = 1.0f` saat mengakumulasikan keluaran $O \mathrel{+}= P V$ (`gemm_nn_fp32`).
- **Backend default (`tiled_avx2_fma`):** *Cache-blocked* (`48 x 64 x 64`, muat di L1d 32 KiB) dengan *micro-kernel* AVX2 + FMA $6 \times 16$.

---

## 3. Tipe Konfigurasi & Metrik Galat (`src/common/attention_types.hpp`)

```cpp
namespace ring_attn {

struct AttentionConfig {
    int s = 1024;           // Panjang sekuens total
    int d = 64;             // Dimensi head
    int c = 256;            // Ukuran blok (baris per blok Q / KV)
    int bq = 64;            // Ukuran tile query internal
    int bk = 64;            // Ukuran tile key internal
    int num_threads = 1;    // Jumlah thread OpenMP (V2/V4)
    bool causal = false;    // Opsional: causal mask
};

struct ErrorMetrics {
    double max_abs_err = 0.0;
    double rel_l2_err = 0.0;
    bool all_finite = true;
};

void validate_attention_config(const AttentionConfig& cfg);
ErrorMetrics compute_error_metrics(const float* test_out, const double* ref_out, std::size_t n);

} // namespace ring_attn
```

---

## 4. Primitif Blok Online Softmax (`src/common/block_attention.hpp`)

```cpp
namespace ring_attn {

// Inisialisasi O_acc = 0, m_state = -INF, l_state = 0 untuk q_rows baris.
void init_block_state(float* O_acc, float* m_state, float* l_state, int q_rows, int d);

// Hitung attention satu pasangan (Q_block, K_block, V_block) secara bertile (bq x bk)
// dan gabungkan ke statistik berjalan (O_acc, m_state, l_state).
void attend_block_online_softmax(
    const float* Q_block, int q_rows, int q_offset,
    const float* K_block, const float* V_block, int k_rows, int k_offset,
    int d, int bq, int bk, float scale, bool causal,
    float* O_acc, float* m_state, float* l_state, float* tile_buf);

// Normalisasi akhir setelah seluruh blok KV selesai: O_acc[r, :] /= l_state[r].
void finalize_block_output(float* O_acc, const float* l_state, int q_rows, int d);

} // namespace ring_attn
```

### Aturan Parameter & Scratchpad
- `q_offset` dan `k_offset` adalah indeks baris global awal masing-masing blok (`0` sampai `s - 1`), dipakai saat `causal == true` (untuk non-causal, nilai offset apa pun aman).
- `tile_buf` harus dialokasikan oleh pemanggil (dan **bersifat privat per thread** pada OpenMP) dengan kapasitas minimal $\min(b_q, q_{\text{rows}}) \times \min(b_k, k_{\text{rows}})$ elemen `float`.

---

## 5. Contoh Pemanggilan Minimal untuk Anggota Tim

### A. Untuk Dhimas — V0 (Naive FP32) & V1 (Blockwise Serial)

**V0 (menggunakan kernel GEMM yang sama dengan V1–V4):**

```cpp
#include "common/gemm.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

void attention_v0_fp32(const float* Q, const float* K, const float* V,
                       float* O, int s, int d) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    std::vector<float> S(static_cast<std::size_t>(s) * s);

    // 1. S[s x s] = (1/sqrt(d)) * Q * K^T
    ring_attn::gemm_nt_fp32(s, s, d, scale, Q, d, K, d, 0.0f, S.data(), s);

    // 2. Softmax stabil per baris secara in-place
    for (int i = 0; i < s; ++i) {
        float* row = S.data() + static_cast<std::size_t>(i) * s;
        float m = *std::max_element(row, row + s);
        double l = 0.0;
        for (int j = 0; j < s; ++j) {
            row[j] = std::exp(row[j] - m);
            l += static_cast<double>(row[j]);
        }
        const float inv_l = 1.0f / static_cast<float>(l);
        for (int j = 0; j < s; ++j) {
            row[j] *= inv_l;
        }
    }

    // 3. O[s x d] = P * V
    ring_attn::gemm_nn_fp32(s, d, s, 1.0f, S.data(), s, V, d, 0.0f, O, d);
}
```

**V1 (Blockwise Serial dengan Online Softmax):**

```cpp
#include "common/attention_types.hpp"
#include "common/block_attention.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

void attention_v1_serial(const float* Q, const float* K, const float* V,
                         float* O, const ring_attn::AttentionConfig& cfg) {
    ring_attn::validate_attention_config(cfg);
    const int s = cfg.s, d = cfg.d, c = cfg.c;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    std::vector<float> m_state(c), l_state(c);
    std::vector<float> tile_buf(static_cast<std::size_t>(std::min(cfg.bq, c)) * std::min(cfg.bk, c));

    for (int q0 = 0; q0 < s; q0 += c) {
        const int q_rows = std::min(c, s - q0);
        float* O_blk = O + static_cast<std::size_t>(q0) * d;
        ring_attn::init_block_state(O_blk, m_state.data(), l_state.data(), q_rows, d);

        for (int k0 = 0; k0 < s; k0 += c) {
            const int k_rows = std::min(c, s - k0);
            ring_attn::attend_block_online_softmax(
                Q + static_cast<std::size_t>(q0) * d, q_rows, q0,
                K + static_cast<std::size_t>(k0) * d,
                V + static_cast<std::size_t>(k0) * d, k_rows, k0,
                d, cfg.bq, cfg.bk, scale, cfg.causal,
                O_blk, m_state.data(), l_state.data(), tile_buf.data());
        }
        ring_attn::finalize_block_output(O_blk, l_state.data(), q_rows, d);
    }
}
```

---

### B. Untuk Thufail — V3 (Ring Attention MPI + Buffer Gabungan `[K|V]`)

Buffer gabungan `KVcur` menyimpan `[K_block | V_block]` kontigu berukuran `2 * c * d`. Cukup kirim pointer `KVcur.data()` dan `KVcur.data() + c * d` tanpa penyalinan tambahan:

```cpp
#include "common/block_attention.hpp"
#include <algorithm>
#include <cmath>
#include <mpi.h>
#include <vector>

void attention_v3_ring_mpi(const float* Q_local,
                           const float* K_local,
                           const float* V_local,
                           float* O_local,
                           int c, int d, int bq, int bk, bool causal,
                           MPI_Comm comm) {
    int rank = 0, P = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    const int prev = (rank - 1 + P) % P;
    const int next = (rank + 1) % P;
    const std::size_t block_elems = static_cast<std::size_t>(c) * d;
    const int kv_count = static_cast<int>(2 * block_elems);
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    // Double buffer gabungan [K | V]
    std::vector<float> KVcur(2 * block_elems), KVnext(2 * block_elems);
    std::copy_n(K_local, block_elems, KVcur.data());
    std::copy_n(V_local, block_elems, KVcur.data() + block_elems);

    std::vector<float> m_state(c), l_state(c);
    std::vector<float> tile_buf(static_cast<std::size_t>(std::min(bq, c)) * std::min(bk, c));
    ring_attn::init_block_state(O_local, m_state.data(), l_state.data(), c, d);

    for (int step = 0; step < P; ++step) {
        MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        if (step < P - 1) {
            MPI_Irecv(KVnext.data(), kv_count, MPI_FLOAT, prev, step, comm, &reqs[0]);
            MPI_Isend(KVcur.data(),  kv_count, MPI_FLOAT, next, step, comm, &reqs[1]);
        }

        const int kv_owner = (rank - step + P) % P;
        const float* K_cur = KVcur.data();
        const float* V_cur = KVcur.data() + block_elems;

        ring_attn::attend_block_online_softmax(
            Q_local, c, rank * c,
            K_cur, V_cur, c, kv_owner * c,
            d, bq, bk, scale, causal,
            O_local, m_state.data(), l_state.data(), tile_buf.data());

        if (step < P - 1) {
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
            KVcur.swap(KVnext);
        }
    }

    ring_attn::finalize_block_output(O_local, l_state.data(), c, d);
}
```

---

## 6. Perintah Kompilasi & Pengujian Cepat (Zsh)

```zsh
# Uji kernel GEMM dan V2 OpenMP
make clean && make setup && make test

# Uji V4 Hybrid MPI + OpenMP (2 rank x 2 thread = 4 core fisik)
make test-v4
```
