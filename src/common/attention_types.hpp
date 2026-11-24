#ifndef RING_ATTN_COMMON_ATTENTION_TYPES_HPP
#define RING_ATTN_COMMON_ATTENTION_TYPES_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ring_attn {

struct AttentionConfig {
    int s = 1024;           // Total sequence length
    int d = 64;             // Head dimension
    int c = 256;            // Block size (rows per Q/KV block)
    int bq = 64;            // Internal query tile size
    int bk = 64;            // Internal key tile size
    int num_threads = 1;    // OpenMP thread count (0 = runtime default)
    bool causal = false;    // Optional causal mask
};

struct ErrorMetrics {
    double max_abs_err = 0.0;
    double rel_l2_err = 0.0;
    bool all_finite = true;
};

inline void validate_attention_config(const AttentionConfig& cfg) {
    if (cfg.s <= 0 || cfg.d <= 0 || cfg.c <= 0 || cfg.bq <= 0 || cfg.bk <= 0) {
        throw std::invalid_argument("AttentionConfig dimensions s, d, c, bq, bk must be positive");
    }
    if (cfg.num_threads < 0) {
        throw std::invalid_argument("AttentionConfig num_threads must be non-negative");
    }
}

inline ErrorMetrics compute_error_metrics(
    const float* test_out,
    const double* ref_out,
    std::size_t n) {
    if (n > 0 && (test_out == nullptr || ref_out == nullptr)) {
        throw std::invalid_argument("compute_error_metrics received null pointer");
    }
    ErrorMetrics metrics{};
    double diff_sq = 0.0;
    double ref_sq = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        const float val = test_out[i];
        if (!std::isfinite(val)) {
            metrics.all_finite = false;
            metrics.max_abs_err = std::numeric_limits<double>::infinity();
            metrics.rel_l2_err = std::numeric_limits<double>::infinity();
            return metrics;
        }
        const double diff = static_cast<double>(val) - ref_out[i];
        metrics.max_abs_err = std::max(metrics.max_abs_err, std::abs(diff));
        diff_sq += diff * diff;
        ref_sq += ref_out[i] * ref_out[i];
    }

    const double denom = std::max(std::sqrt(ref_sq), 1e-12);
    metrics.rel_l2_err = std::sqrt(diff_sq) / denom;
    return metrics;
}

} // namespace ring_attn

#endif // RING_ATTN_COMMON_ATTENTION_TYPES_HPP
