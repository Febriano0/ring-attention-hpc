#ifndef RING_ATTN_COMMON_BLOCK_ATTENTION_HPP
#define RING_ATTN_COMMON_BLOCK_ATTENTION_HPP

namespace ring_attn {

/// Initializes running Online Softmax state (O_acc = 0, m = -inf, l = 0) for a query block.
void init_block_state(float* O_acc, float* m_state, float* l_state, int q_rows, int d);

/// Accumulates one (Q_block, K_block, V_block) interaction into (O_acc, m_state, l_state)
/// using bq x bk internal tiles and numerically stable Online Softmax.
/// tile_buf must be caller-owned thread-private storage of at least min(bq, q_rows) * min(bk, k_rows) floats.
void attend_block_online_softmax(
    const float* Q_block, int q_rows, int q_offset,
    const float* K_block, const float* V_block, int k_rows, int k_offset,
    int d, int bq, int bk, float scale, bool causal,
    float* O_acc, float* m_state, float* l_state, float* tile_buf);

/// Normalizes accumulated block output: O_acc[r, :] /= l_state[r].
void finalize_block_output(float* O_acc, const float* l_state, int q_rows, int d);

} // namespace ring_attn

#endif // RING_ATTN_COMMON_BLOCK_ATTENTION_HPP
