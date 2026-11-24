#include "v1_blockwise/v1_blockwise.hpp"
#include "common/block_attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ring_attn {

void attention_v1_blockwise(const float* Q,
                            const float* K,
                            const float* V,
                            float* O,
                            const AttentionConfig& cfg) {
    validate_attention_config(cfg);
    if (Q == nullptr || K == nullptr || V == nullptr || O == nullptr) {
        throw std::invalid_argument("attention_v1_blockwise received null tensor pointer");
    }

    const int s = cfg.s;
    const int d = cfg.d;
    const int c = cfg.c;
    const int bq = cfg.bq;
    const int bk = cfg.bk;
    const bool causal = cfg.causal;

    const int max_q_rows = std::min(c, s);
    const int eff_bq = std::min(bq, max_q_rows);
    const int eff_bk = std::min(bk, max_q_rows);
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    std::vector<float> m_state(static_cast<std::size_t>(max_q_rows));
    std::vector<float> l_state(static_cast<std::size_t>(max_q_rows));
    std::vector<float> tile_buf(static_cast<std::size_t>(eff_bq) * eff_bk);

    for (int q0 = 0; q0 < s; q0 += c) {
        const int q_rows = std::min(c, s - q0);
        const float* Q_blk = Q + static_cast<std::size_t>(q0) * d;
        float* O_blk = O + static_cast<std::size_t>(q0) * d;

        init_block_state(O_blk, m_state.data(), l_state.data(), q_rows, d);

        for (int k0 = 0; k0 < s; k0 += c) {
            if (causal && k0 > q0 + q_rows - 1) {
                break;
            }
            const int k_rows = std::min(c, s - k0);
            const float* K_blk = K + static_cast<std::size_t>(k0) * d;
            const float* V_blk = V + static_cast<std::size_t>(k0) * d;

            attend_block_online_softmax(
                Q_blk, q_rows, q0,
                K_blk, V_blk, k_rows, k0,
                d, bq, bk, scale, causal,
                O_blk, m_state.data(), l_state.data(), tile_buf.data());
        }

        finalize_block_output(O_blk, l_state.data(), q_rows, d);
    }
}

} // namespace ring_attn
