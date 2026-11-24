#include "v0_naive/v0_naive.hpp"
#include "common/gemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ring_attn {
namespace {

void validate_v0_args(const void* Q, const void* K, const void* V, const void* O, int s, int d) {
    if (s <= 0 || d <= 0) {
        throw std::invalid_argument("V0 dimensions s and d must be positive");
    }
    if (Q == nullptr || K == nullptr || V == nullptr || O == nullptr) {
        throw std::invalid_argument("V0 received null tensor pointer");
    }
}

} // namespace

void attention_v0_fp64(const double* Q,
                       const double* K,
                       const double* V,
                       double* O,
                       int s,
                       int d,
                       bool causal) {
    validate_v0_args(Q, K, V, O, s, d);

    const double scale = 1.0 / std::sqrt(static_cast<double>(d));
    std::vector<double> scores(static_cast<std::size_t>(s));
    std::vector<double> probs(static_cast<std::size_t>(s));

    for (int i = 0; i < s; ++i) {
        const int max_j = causal ? (i + 1) : s;
        const double* q_row = Q + static_cast<std::size_t>(i) * d;
        double max_score = -std::numeric_limits<double>::infinity();

        for (int j = 0; j < max_j; ++j) {
            const double* k_row = K + static_cast<std::size_t>(j) * d;
            double dot = 0.0;
            for (int t = 0; t < d; ++t) {
                dot += q_row[t] * k_row[t];
            }
            const double sc = dot * scale;
            scores[static_cast<std::size_t>(j)] = sc;
            max_score = std::max(max_score, sc);
        }

        double sum_exp = 0.0;
        for (int j = 0; j < max_j; ++j) {
            const double p = std::exp(scores[static_cast<std::size_t>(j)] - max_score);
            probs[static_cast<std::size_t>(j)] = p;
            sum_exp += p;
        }

        const double inv_sum = 1.0 / sum_exp;
        double* o_row = O + static_cast<std::size_t>(i) * d;
        std::fill_n(o_row, d, 0.0);

        for (int j = 0; j < max_j; ++j) {
            const double w = probs[static_cast<std::size_t>(j)] * inv_sum;
            const double* v_row = V + static_cast<std::size_t>(j) * d;
            for (int t = 0; t < d; ++t) {
                o_row[t] += w * v_row[t];
            }
        }
    }
}

void attention_v0_fp32(const float* Q,
                       const float* K,
                       const float* V,
                       float* O,
                       int s,
                       int d,
                       bool causal) {
    validate_v0_args(Q, K, V, O, s, d);

    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    const std::size_t s_sz = static_cast<std::size_t>(s);
    // Explicit O(s^2) score matrix materialization (baseline bottleneck).
    std::vector<float> S(s_sz * s_sz);

    gemm_nt_fp32(s, s, d, scale, Q, d, K, d, 0.0f, S.data(), s);

    for (int i = 0; i < s; ++i) {
        float* row = S.data() + static_cast<std::size_t>(i) * s_sz;
        const int max_j = causal ? (i + 1) : s;
        float row_max = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < max_j; ++j) {
            row_max = std::max(row_max, row[j]);
        }

        double row_sum = 0.0;
        for (int j = 0; j < max_j; ++j) {
            const float p = std::exp(row[j] - row_max);
            row[j] = p;
            row_sum += static_cast<double>(p);
        }
        const float inv_sum = 1.0f / static_cast<float>(row_sum);
        for (int j = 0; j < max_j; ++j) {
            row[j] *= inv_sum;
        }
        if (causal && max_j < s) {
            std::fill(row + max_j, row + s, 0.0f);
        }
    }

    gemm_nn_fp32(s, d, s, 1.0f, S.data(), s, V, d, 0.0f, O, d);
}

} // namespace ring_attn
