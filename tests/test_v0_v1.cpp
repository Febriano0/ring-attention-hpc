#include "common/attention_types.hpp"
#include "v0_naive/v0_naive.hpp"
#include "v1_blockwise/v1_blockwise.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

void fill_tensors(
    int s, int d, float qk_std, std::uint64_t seed,
    std::vector<float>& Q, std::vector<float>& K, std::vector<float>& V,
    std::vector<double>& Q64, std::vector<double>& K64, std::vector<double>& V64) {
    const std::size_t n = static_cast<std::size_t>(s) * d;
    Q.resize(n); K.resize(n); V.resize(n);
    Q64.resize(n); K64.resize(n); V64.resize(n);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> qk(0.0f, qk_std);
    std::normal_distribution<float> vv(0.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i) {
        Q[i] = qk(rng); K[i] = qk(rng); V[i] = vv(rng);
        Q64[i] = Q[i]; K64[i] = K[i]; V64[i] = V[i];
    }
}

bool run_case(const std::string& name, int s, int d, int c, int tile,
              float qk_std, std::uint64_t seed, bool causal) {
    std::vector<float> Q, K, V, O0fp32, O1;
    std::vector<double> Q64, K64, V64, Oref;
    fill_tensors(s, d, qk_std, seed, Q, K, V, Q64, K64, V64);
    const std::size_t n = static_cast<std::size_t>(s) * d;
    O0fp32.resize(n, 0.0f); O1.resize(n, 0.0f); Oref.resize(n, 0.0);

    ring_attn::attention_v0_fp64(Q64.data(), K64.data(), V64.data(), Oref.data(), s, d, causal);
    ring_attn::attention_v0_fp32(Q.data(), K.data(), V.data(), O0fp32.data(), s, d, causal);
    ring_attn::AttentionConfig cfg{s, d, c, tile, tile, 1, causal};
    ring_attn::attention_v1_blockwise(Q.data(), K.data(), V.data(), O1.data(), cfg);

    const ring_attn::ErrorMetrics e0 =
        ring_attn::compute_error_metrics(O0fp32.data(), Oref.data(), n);
    const ring_attn::ErrorMetrics e1 =
        ring_attn::compute_error_metrics(O1.data(), Oref.data(), n);
    const bool ok = e0.all_finite && e1.all_finite &&
                    e0.rel_l2_err < 1e-5 && e1.rel_l2_err < 1e-5;

    std::cout << std::left << std::setw(24) << name
              << " | s=" << s << " d=" << d << " c=" << c
              << " tile=" << tile << " | v0 rel=" << std::scientific << e0.rel_l2_err
              << " | v1 rel=" << e1.rel_l2_err
              << " | " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

} // namespace

int main() {
    std::cout << "=== test_v0_v1 (reference: V0-FP64) ===\n";
    bool ok = true;
    ok &= run_case("standard_normal", 256, 64, 64, 64, 1.0f, 101ULL, false);
    ok &= run_case("non_divisible", 197, 67, 64, 32, 1.0f, 102ULL, false);
    ok &= run_case("stress_logits_std10", 512, 64, 128, 64, std::sqrt(10.0f), 103ULL, false);
    ok &= run_case("stress_logits_std100", 512, 64, 128, 64, 10.0f, 104ULL, false);
    ok &= run_case("causal", 256, 64, 64, 64, 1.0f, 105ULL, true);
    std::cout << (ok ? "ALL V0/V1 TESTS PASSED\n" : "SOME V0/V1 TESTS FAILED\n");
    return ok ? 0 : 1;
}
