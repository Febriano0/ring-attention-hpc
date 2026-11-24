#include "common/block_attention.hpp"
#include "common/gemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ring_attn {
namespace {

void apply_causal_mask_tile(
    float* tile_buf, int qm, int kn, int q_base, int k_base) noexcept {
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();
    for (int r = 0; r < qm; ++r) {
        const int global_q = q_base + r;
        float* row = tile_buf + static_cast<std::size_t>(r) * kn;
        for (int col = 0; col < kn; ++col) {
            if (k_base + col > global_q) {
                row[col] = kNegInf;
            }
        }
    }
}

void update_online_softmax_row(
    float* tile_row, int kn, int d,
    float* o_row, float& m_val, float& l_val) noexcept {
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();

    float row_max = kNegInf;
    for (int col = 0; col < kn; ++col) {
        row_max = std::max(row_max, tile_row[col]);
    }
    if (row_max == kNegInf) {
        std::fill_n(tile_row, kn, 0.0f);
        return;
    }

    const float m_old = m_val;
    const float m_new = std::max(m_old, row_max);
    const float alpha = (m_old == kNegInf) ? 0.0f : std::exp(m_old - m_new);

    if (alpha == 0.0f) {
        std::fill_n(o_row, d, 0.0f);
    } else if (alpha != 1.0f) {
        for (int idx = 0; idx < d; ++idx) {
            o_row[idx] *= alpha;
        }
    }

    double row_sum = 0.0;
    for (int col = 0; col < kn; ++col) {
        const float score = tile_row[col];
        const float p = (score == kNegInf) ? 0.0f : std::exp(score - m_new);
        tile_row[col] = p;
        row_sum += static_cast<double>(p);
    }

    m_val = m_new;
    l_val = static_cast<float>(static_cast<double>(alpha) * static_cast<double>(l_val) + row_sum);
}

} // namespace

void init_block_state(float* O_acc, float* m_state, float* l_state, int q_rows, int d) {
    if (q_rows <= 0 || d <= 0 || O_acc == nullptr || m_state == nullptr || l_state == nullptr) {
        throw std::invalid_argument("init_block_state received invalid dimensions or null pointer");
    }
    std::fill_n(O_acc, static_cast<std::size_t>(q_rows) * d, 0.0f);
    std::fill_n(m_state, q_rows, -std::numeric_limits<float>::infinity());
    std::fill_n(l_state, q_rows, 0.0f);
}

void attend_block_online_softmax(
    const float* Q_block, int q_rows, int q_offset,
    const float* K_block, const float* V_block, int k_rows, int k_offset,
    int d, int bq, int bk, float scale, bool causal,
    float* O_acc, float* m_state, float* l_state, float* tile_buf) {
    if (q_rows <= 0 || k_rows <= 0 || d <= 0 || bq <= 0 || bk <= 0 ||
        Q_block == nullptr || K_block == nullptr || V_block == nullptr ||
        O_acc == nullptr || m_state == nullptr || l_state == nullptr || tile_buf == nullptr) {
        throw std::invalid_argument("attend_block_online_softmax received invalid arguments");
    }

    for (int qi = 0; qi < q_rows; qi += bq) {
        const int qm = std::min(bq, q_rows - qi);
        const float* Q_tile = Q_block + static_cast<std::size_t>(qi) * d;
        float* O_tile = O_acc + static_cast<std::size_t>(qi) * d;

        for (int kj = 0; kj < k_rows; kj += bk) {
            const int kn = std::min(bk, k_rows - kj);
            if (causal && (k_offset + kj > q_offset + qi + qm - 1)) {
                continue;
            }

            const float* K_tile = K_block + static_cast<std::size_t>(kj) * d;
            const float* V_tile = V_block + static_cast<std::size_t>(kj) * d;

            // 1. Tile score matrix: S_tile[qm x kn] = scale * Q_tile * K_tile^T
            gemm_nt_fp32(qm, kn, d, scale, Q_tile, d, K_tile, d, 0.0f, tile_buf, kn);

            if (causal) {
                apply_causal_mask_tile(tile_buf, qm, kn, q_offset + qi, k_offset + kj);
            }

            // 2. Per-row Online Softmax update & rescaling of O_acc
            for (int r = 0; r < qm; ++r) {
                update_online_softmax_row(
                    tile_buf + static_cast<std::size_t>(r) * kn,
                    kn, d,
                    O_tile + static_cast<std::size_t>(r) * d,
                    m_state[qi + r],
                    l_state[qi + r]);
            }

            // 3. Output tile accumulation: O_tile += P_tile * V_tile
            gemm_nn_fp32(qm, d, kn, 1.0f, tile_buf, kn, V_tile, d, 1.0f, O_tile, d);
        }
    }
}

void finalize_block_output(float* O_acc, const float* l_state, int q_rows, int d) {
    if (q_rows <= 0 || d <= 0 || O_acc == nullptr || l_state == nullptr) {
        throw std::invalid_argument("finalize_block_output received invalid dimensions or null pointer");
    }
    for (int r = 0; r < q_rows; ++r) {
        const float denom = l_state[r];
        float* row = O_acc + static_cast<std::size_t>(r) * d;
        if (denom > 0.0f) {
            const float inv_l = 1.0f / denom;
            for (int col = 0; col < d; ++col) {
                row[col] *= inv_l;
            }
        } else {
            std::fill_n(row, d, 0.0f);
        }
    }
}

} // namespace ring_attn
