#ifndef RING_ATTN_COMMON_GEMM_HPP
#define RING_ATTN_COMMON_GEMM_HPP

namespace ring_attn {

/// Returns the active GEMM backend identifier string.
const char* gemm_backend_name() noexcept;

/// Computes C[M x N] = alpha * A[M x K] * B^T[K x N] + beta * C[M x N] in row-major order.
/// B has logical shape [N x K] with row stride ldb.
/// Never reads C when beta == 0.0f.
void gemm_nt_fp32(int M, int N, int K,
                  float alpha,
                  const float* A, int lda,
                  const float* B, int ldb,
                  float beta,
                  float* C, int ldc);

/// Computes C[M x N] = alpha * A[M x K] * B[K x N] + beta * C[M x N] in row-major order.
/// B has logical shape [K x N] with row stride ldb.
/// Never reads C when beta == 0.0f.
void gemm_nn_fp32(int M, int N, int K,
                  float alpha,
                  const float* A, int lda,
                  const float* B, int ldb,
                  float beta,
                  float* C, int ldc);

} // namespace ring_attn

#endif // RING_ATTN_COMMON_GEMM_HPP
