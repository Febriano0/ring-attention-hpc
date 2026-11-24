#ifndef RING_ATTN_V2_OPENMP_HPP
#define RING_ATTN_V2_OPENMP_HPP

#include "common/attention_types.hpp"

namespace ring_attn {

/// Computes single-head self-attention using Blockwise Online Softmax parallelized
/// across outer query blocks with OpenMP (#pragma omp parallel for).
/// Q, K, V, and O have row-major shape [cfg.s x cfg.d].
void attention_v2_openmp(const float* Q,
                         const float* K,
                         const float* V,
                         float* O,
                         const AttentionConfig& cfg);

} // namespace ring_attn

#endif // RING_ATTN_V2_OPENMP_HPP
