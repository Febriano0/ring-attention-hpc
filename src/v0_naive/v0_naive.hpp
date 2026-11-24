#ifndef RING_ATTN_V0_NAIVE_HPP
#define RING_ATTN_V0_NAIVE_HPP

namespace ring_attn {

/// Ground-truth FP64 single-head self-attention reference.
/// Q, K, V, and O have row-major shape [s x d].
void attention_v0_fp64(const double* Q,
                       const double* K,
                       const double* V,
                       double* O,
                       int s,
                       int d,
                       bool causal = false);

/// Baseline naive FP32 self-attention with full O(s^2) score matrix S materialization.
/// Uses the shared gemm_nt_fp32 and gemm_nn_fp32 compute kernels.
void attention_v0_fp32(const float* Q,
                       const float* K,
                       const float* V,
                       float* O,
                       int s,
                       int d,
                       bool causal = false);

} // namespace ring_attn

#endif // RING_ATTN_V0_NAIVE_HPP
