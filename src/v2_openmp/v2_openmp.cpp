#include "v2_openmp/v2_openmp.hpp"
#include "common/block_attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <omp.h>
#include <stdexcept>
#include <vector>

namespace ring_attn {

void attention_v2_openmp(const float* Q,
                         const float* K,
                         const float* V,
                         float* O,
                         const AttentionConfig& cfg) {
    validate_attention_config(cfg);
    if (Q == nullptr || K == nullptr || V == nullptr || O == nullptr) {
        throw std::invalid_argument("attention_v2_openmp received null tensor pointer");
    }

    const int s = cfg.s;
    const int d = cfg.d;
    const int c = cfg.c;
    const int bq = cfg.bq;
    const int bk = cfg.bk;
    const bool causal = cfg.causal;

    const int num_q_blocks = (s + c - 1) / c;
    const int num_kv_blocks = (s + c - 1) / c;
    const int max_q_rows = std::min(c, s);
    const int eff_bq = std::min(bq, max_q_rows);
    const int eff_bk = std::min(bk, max_q_rows);
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    const int threads = (cfg.num_threads > 0) ? cfg.num_threads : omp_get_max_threads();

#pragma omp parallel num_threads(threads)
    {
        // Thread-private state and scratchpad; outer Q-block partitioning makes O writes disjoint.
        std::vector<float> m_state(static_cast<std::size_t>(max_q_rows));
        std::vector<float> l_state(static_cast<std::size_t>(max_q_rows));
        std::vector<float> tile_buf(static_cast<std::size_t>(eff_bq) * eff_bk);

        auto run_q_block = [&](int qi_blk) {
            const int q_start = qi_blk * c;
            const int q_rows = std::min(c, s - q_start);
            const float* Q_blk = Q + static_cast<std::size_t>(q_start) * d;
            float* O_blk = O + static_cast<std::size_t>(q_start) * d;

            init_block_state(O_blk, m_state.data(), l_state.data(), q_rows, d);

            for (int kj_blk = 0; kj_blk < num_kv_blocks; ++kj_blk) {
                const int k_start = kj_blk * c;
                if (causal && k_start > q_start + q_rows - 1) {
                    break;
                }
                const int k_rows = std::min(c, s - k_start);
                const float* K_blk = K + static_cast<std::size_t>(k_start) * d;
                const float* V_blk = V + static_cast<std::size_t>(k_start) * d;

                attend_block_online_softmax(
                    Q_blk, q_rows, q_start,
                    K_blk, V_blk, k_rows, k_start,
                    d, bq, bk, scale, causal,
                    O_blk, m_state.data(), l_state.data(), tile_buf.data());
            }

            finalize_block_output(O_blk, l_state.data(), q_rows, d);
        };

        if (causal) {
#pragma omp for schedule(dynamic, 1)
            for (int qi_blk = 0; qi_blk < num_q_blocks; ++qi_blk) {
                run_q_block(qi_blk);
            }
        } else {
#pragma omp for schedule(static)
            for (int qi_blk = 0; qi_blk < num_q_blocks; ++qi_blk) {
                run_q_block(qi_blk);
            }
        }
    }
}

} // namespace ring_attn
