#ifndef RING_ATTN_V1_BLOCKWISE_HPP
#define RING_ATTN_V1_BLOCKWISE_HPP

#include "common/attention_types.hpp"

namespace ring_attn {

/// Serial Blockwise Attention with tiled Online Softmax (O(s*d + bq*bk) memory).
/// Q, K, V, and O have row-major shape [cfg.s x cfg.d].
void attention_v1_blockwise(const float* Q,
                            const float* K,
                            const float* V,
                            float* O,
                            const AttentionConfig& cfg);

} // namespace ring_attn

#endif // RING_ATTN_V1_BLOCKWISE_HPP
