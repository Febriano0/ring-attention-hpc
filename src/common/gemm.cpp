#include "common/gemm.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace ring_attn {
namespace {

constexpr int kTileMC = 48; // Multiple of MR = 6
constexpr int kTileNC = 64; // Multiple of NR = 16
constexpr int kTileKC = 64; // Fits packed panels comfortably inside 32 KiB L1d

bool validate_and_check_nonempty(
    int M, int N, int K,
    int lda, int ldb, int ldc,
    int min_lda, int min_ldb,
    const float* A, const float* B, const float* C) {
    if (M < 0 || N < 0 || K < 0) {
        throw std::invalid_argument("GEMM dimensions M, N, K must be non-negative");
    }
    if (M == 0 || N == 0) {
        return false;
    }
    if (C == nullptr || ldc < N) {
        throw std::invalid_argument("Invalid output matrix C or stride ldc");
    }
    if (K > 0) {
        if (A == nullptr || lda < min_lda) {
            throw std::invalid_argument("Invalid input matrix A or stride lda");
        }
        if (B == nullptr || ldb < min_ldb) {
            throw std::invalid_argument("Invalid input matrix B or stride ldb");
        }
    }
    return true;
}

void apply_beta_scale(int M, int N, float beta, float* C, int ldc) noexcept {
    if (beta == 0.0f) {
        // Never read C when beta == 0.0f so pre-existing NaN/Inf is overwritten cleanly.
        for (int i = 0; i < M; ++i) {
            std::fill_n(C + static_cast<std::size_t>(i) * ldc, N, 0.0f);
        }
        return;
    }
    if (beta == 1.0f) {
        return;
    }
    for (int i = 0; i < M; ++i) {
        float* c_row = C + static_cast<std::size_t>(i) * ldc;
        for (int j = 0; j < N; ++j) {
            c_row[j] *= beta;
        }
    }
}

#ifndef USE_OPENBLAS

void pack_b_nn(int kb, int nb, const float* B, int ldb, float* B_pack) noexcept {
    for (int k = 0; k < kb; ++k) {
        const float* src = B + static_cast<std::size_t>(k) * ldb;
        float* dst = B_pack + static_cast<std::size_t>(k) * nb;
        std::copy_n(src, nb, dst);
    }
}

void pack_b_nt(int nb, int kb, const float* B, int ldb, float* B_pack) noexcept {
    // Transpose [nb x kb] block into contiguous [kb x nb] panel so NT reuses the NN micro-kernel.
    for (int j = 0; j < nb; ++j) {
        const float* src = B + static_cast<std::size_t>(j) * ldb;
        for (int k = 0; k < kb; ++k) {
            B_pack[static_cast<std::size_t>(k) * nb + j] = src[k];
        }
    }
}

#if defined(__AVX2__) && defined(__FMA__)
inline void micro_kernel_6x16_avx2(
    int kb, float alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc) noexcept {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

    const float* a0 = A;
    const float* a1 = a0 + lda;
    const float* a2 = a1 + lda;
    const float* a3 = a2 + lda;
    const float* a4 = a3 + lda;
    const float* a5 = a4 + lda;
    const float* bptr = B;

    for (int k = 0; k < kb; ++k) {
        const __m256 b0 = _mm256_loadu_ps(bptr);
        const __m256 b1 = _mm256_loadu_ps(bptr + 8);
        __m256 va;
        va = _mm256_set1_ps(a0[k]); c00 = _mm256_fmadd_ps(va, b0, c00); c01 = _mm256_fmadd_ps(va, b1, c01);
        va = _mm256_set1_ps(a1[k]); c10 = _mm256_fmadd_ps(va, b0, c10); c11 = _mm256_fmadd_ps(va, b1, c11);
        va = _mm256_set1_ps(a2[k]); c20 = _mm256_fmadd_ps(va, b0, c20); c21 = _mm256_fmadd_ps(va, b1, c21);
        va = _mm256_set1_ps(a3[k]); c30 = _mm256_fmadd_ps(va, b0, c30); c31 = _mm256_fmadd_ps(va, b1, c31);
        va = _mm256_set1_ps(a4[k]); c40 = _mm256_fmadd_ps(va, b0, c40); c41 = _mm256_fmadd_ps(va, b1, c41);
        va = _mm256_set1_ps(a5[k]); c50 = _mm256_fmadd_ps(va, b0, c50); c51 = _mm256_fmadd_ps(va, b1, c51);
        bptr += ldb;
    }

    const __m256 valpha = _mm256_set1_ps(alpha);
    auto store_row = [&](float* row, __m256 lo, __m256 hi) noexcept {
        _mm256_storeu_ps(row, _mm256_fmadd_ps(valpha, lo, _mm256_loadu_ps(row)));
        _mm256_storeu_ps(row + 8, _mm256_fmadd_ps(valpha, hi, _mm256_loadu_ps(row + 8)));
    };
    store_row(C + 0 * ldc, c00, c01);
    store_row(C + 1 * ldc, c10, c11);
    store_row(C + 2 * ldc, c20, c21);
    store_row(C + 3 * ldc, c30, c31);
    store_row(C + 4 * ldc, c40, c41);
    store_row(C + 5 * ldc, c50, c51);
}
#endif

void accumulate_tail_rows(
    int mr, int nb, int kb, float alpha,
    const float* A, int lda,
    const float* B_panel, int ldb_panel,
    float* C, int ldc) noexcept {
    for (int i = 0; i < mr; ++i) {
        const float* a_row = A + static_cast<std::size_t>(i) * lda;
        float* c_row = C + static_cast<std::size_t>(i) * ldc;
        int j = 0;
#if defined(__AVX2__) && defined(__FMA__)
        const __m256 valpha = _mm256_set1_ps(alpha);
        for (; j + 8 <= nb; j += 8) {
            __m256 acc = _mm256_setzero_ps();
            const float* bptr = B_panel + j;
            for (int k = 0; k < kb; ++k) {
                acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[k]), _mm256_loadu_ps(bptr), acc);
                bptr += ldb_panel;
            }
            _mm256_storeu_ps(c_row + j, _mm256_fmadd_ps(valpha, acc, _mm256_loadu_ps(c_row + j)));
        }
#endif
        for (; j < nb; ++j) {
            float acc = 0.0f;
            const float* bptr = B_panel + j;
            for (int k = 0; k < kb; ++k) {
                acc += a_row[k] * (*bptr);
                bptr += ldb_panel;
            }
            c_row[j] += alpha * acc;
        }
    }
}

void accumulate_tile_nn(
    int mb, int nb, int kb, float alpha,
    const float* A, int lda,
    const float* B_panel, int ldb_panel,
    float* C, int ldc) noexcept {
    int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    for (; i + 6 <= mb; i += 6) {
        const float* a_blk = A + static_cast<std::size_t>(i) * lda;
        float* c_blk = C + static_cast<std::size_t>(i) * ldc;
        int j = 0;
        for (; j + 16 <= nb; j += 16) {
            micro_kernel_6x16_avx2(kb, alpha, a_blk, lda, B_panel + j, ldb_panel, c_blk + j, ldc);
        }
        if (j < nb) {
            accumulate_tail_rows(6, nb - j, kb, alpha, a_blk, lda, B_panel + j, ldb_panel, c_blk + j, ldc);
        }
    }
#endif
    if (i < mb) {
        accumulate_tail_rows(mb - i, nb, kb, alpha,
                             A + static_cast<std::size_t>(i) * lda, lda,
                             B_panel, ldb_panel,
                             C + static_cast<std::size_t>(i) * ldc, ldc);
    }
}

template <bool TransB>
void gemm_blocked_impl(
    int M, int N, int K,
    float alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float beta,
    float* C, int ldc) {
    apply_beta_scale(M, N, beta, C, ldc);
    if (K == 0 || alpha == 0.0f) {
        return;
    }

    alignas(64) float B_pack[kTileKC * kTileNC];

    for (int j0 = 0; j0 < N; j0 += kTileNC) {
        const int nb = std::min(kTileNC, N - j0);
        for (int k0 = 0; k0 < K; k0 += kTileKC) {
            const int kb = std::min(kTileKC, K - k0);
            if constexpr (TransB) {
                pack_b_nt(nb, kb, B + static_cast<std::size_t>(j0) * ldb + k0, ldb, B_pack);
            } else {
                pack_b_nn(kb, nb, B + static_cast<std::size_t>(k0) * ldb + j0, ldb, B_pack);
            }
            for (int i0 = 0; i0 < M; i0 += kTileMC) {
                const int mb = std::min(kTileMC, M - i0);
                accumulate_tile_nn(
                    mb, nb, kb, alpha,
                    A + static_cast<std::size_t>(i0) * lda + k0, lda,
                    B_pack, nb,
                    C + static_cast<std::size_t>(i0) * ldc + j0, ldc);
            }
        }
    }
}

#endif // !USE_OPENBLAS

} // namespace

const char* gemm_backend_name() noexcept {
#ifdef USE_OPENBLAS
    return "openblas_cblas";
#elif defined(__AVX2__) && defined(__FMA__)
    return "tiled_avx2_fma";
#else
    return "tiled_scalar";
#endif
}

void gemm_nt_fp32(int M, int N, int K,
                  float alpha,
                  const float* A, int lda,
                  const float* B, int ldb,
                  float beta,
                  float* C, int ldc) {
    if (!validate_and_check_nonempty(M, N, K, lda, ldb, ldc, K, K, A, B, C)) {
        return;
    }
#ifdef USE_OPENBLAS
    if (K == 0 || alpha == 0.0f) {
        apply_beta_scale(M, N, beta, C, ldc);
        return;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#else
    gemm_blocked_impl<true>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

void gemm_nn_fp32(int M, int N, int K,
                  float alpha,
                  const float* A, int lda,
                  const float* B, int ldb,
                  float beta,
                  float* C, int ldc) {
    if (!validate_and_check_nonempty(M, N, K, lda, ldb, ldc, K, N, A, B, C)) {
        return;
    }
#ifdef USE_OPENBLAS
    if (K == 0 || alpha == 0.0f) {
        apply_beta_scale(M, N, beta, C, ldc);
        return;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#else
    gemm_blocked_impl<false>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

} // namespace ring_attn
