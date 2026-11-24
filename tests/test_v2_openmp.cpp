#include "common/attention_types.hpp"
#include "common/gemm.hpp"
#include "v2_openmp/v2_openmp.hpp"

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

void reference_attention_fp64(
    const std::vector<double>& Q,
    const std::vector<double>& K,
    const std::vector<double>& V,
    std::vector<double>& O,
    int s, int d, bool causal) {
    const double scale = 1.0 / std::sqrt(static_cast<double>(d));
    std::vector<double> scores(static_cast<std::size_t>(s));
    std::vector<double> probs(static_cast<std::size_t>(s));

    for (int i = 0; i < s; ++i) {
        const int max_j = causal ? (i + 1) : s;
        double max_score = -std::numeric_limits<double>::infinity();
        const double* q_row = Q.data() + static_cast<std::size_t>(i) * d;

        for (int j = 0; j < max_j; ++j) {
            const double* k_row = K.data() + static_cast<std::size_t>(j) * d;
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
        double* o_row = O.data() + static_cast<std::size_t>(i) * d;
        std::fill_n(o_row, d, 0.0);

        for (int j = 0; j < max_j; ++j) {
            const double weight = probs[static_cast<std::size_t>(j)] * inv_sum;
            const double* v_row = V.data() + static_cast<std::size_t>(j) * d;
            for (int t = 0; t < d; ++t) {
                o_row[t] += weight * v_row[t];
            }
        }
    }
}

void generate_tensors(
    int s, int d, float qk_std, float v_std, std::uint64_t seed,
    std::vector<float>& Q, std::vector<float>& K, std::vector<float>& V,
    std::vector<double>& Q64, std::vector<double>& K64, std::vector<double>& V64) {
    const std::size_t n = static_cast<std::size_t>(s) * d;
    Q.resize(n);
    K.resize(n);
    V.resize(n);
    Q64.resize(n);
    K64.resize(n);
    V64.resize(n);

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist_qk(0.0f, qk_std);
    std::normal_distribution<float> dist_v(0.0f, v_std);

    for (std::size_t i = 0; i < n; ++i) {
        Q[i] = dist_qk(rng);
        K[i] = dist_qk(rng);
        V[i] = dist_v(rng);
        Q64[i] = static_cast<double>(Q[i]);
        K64[i] = static_cast<double>(K[i]);
        V64[i] = static_cast<double>(V[i]);
    }
}

bool run_accuracy_case(
    const std::string& name,
    const ring_attn::AttentionConfig& cfg,
    float qk_std, float v_std, std::uint64_t seed,
    double rel_tol = 1e-5) {
    std::vector<float> Q, K, V;
    std::vector<double> Q64, K64, V64;
    generate_tensors(cfg.s, cfg.d, qk_std, v_std, seed, Q, K, V, Q64, K64, V64);

    const std::size_t n = static_cast<std::size_t>(cfg.s) * cfg.d;
    std::vector<float> O_v2(n, 0.0f);
    std::vector<double> O_ref(n, 0.0);

    ring_attn::attention_v2_openmp(Q.data(), K.data(), V.data(), O_v2.data(), cfg);
    reference_attention_fp64(Q64, K64, V64, O_ref, cfg.s, cfg.d, cfg.causal);

    const ring_attn::ErrorMetrics m = ring_attn::compute_error_metrics(O_v2.data(), O_ref.data(), n);
    const bool ok = m.all_finite && (m.rel_l2_err < rel_tol);

    std::cout << std::left << std::setw(26) << name
              << " | s=" << std::setw(4) << cfg.s
              << " d=" << std::setw(3) << cfg.d
              << " c=" << std::setw(3) << cfg.c
              << " tile=" << std::setw(3) << cfg.bq << "x" << std::setw(3) << cfg.bk
              << " P=" << cfg.num_threads
              << (cfg.causal ? " C" : "  ")
              << " | max_abs=" << std::scientific << std::setprecision(3) << m.max_abs_err
              << " | rel_l2=" << m.rel_l2_err
              << " | " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool run_thread_determinism_check() {
    ring_attn::AttentionConfig cfg;
    cfg.s = 512;
    cfg.d = 64;
    cfg.c = 128;
    cfg.bq = 64;
    cfg.bk = 64;

    std::vector<float> Q, K, V;
    std::vector<double> Q64, K64, V64;
    generate_tensors(cfg.s, cfg.d, 1.0f, 1.0f, 777ULL, Q, K, V, Q64, K64, V64);

    const std::size_t n = static_cast<std::size_t>(cfg.s) * cfg.d;
    std::vector<float> O_p1(n), O_p2(n), O_p4(n);

    cfg.num_threads = 1;
    ring_attn::attention_v2_openmp(Q.data(), K.data(), V.data(), O_p1.data(), cfg);
    cfg.num_threads = 2;
    ring_attn::attention_v2_openmp(Q.data(), K.data(), V.data(), O_p2.data(), cfg);
    cfg.num_threads = 4;
    ring_attn::attention_v2_openmp(Q.data(), K.data(), V.data(), O_p4.data(), cfg);

    const bool exact_1_2 = (O_p1 == O_p2);
    const bool exact_1_4 = (O_p1 == O_p4);
    const bool ok = exact_1_2 && exact_1_4;

    std::cout << std::left << std::setw(26) << "thread_determinism_p1_2_4"
              << " | s=512  d=64  c=128 tile=64 x64  P=1,2,4"
              << " | bit_exact_p1_p2=" << (exact_1_2 ? "yes" : "no")
              << " bit_exact_p1_p4=" << (exact_1_4 ? "yes" : "no")
              << " | " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

} // namespace

int main() {
    std::cout << "=== test_v2_openmp (GEMM backend: " << ring_attn::gemm_backend_name() << ") ===\n";
    bool all_ok = true;

    // 1. Standard normal N(0, 1) configurations
    all_ok &= run_accuracy_case("std_normal_s256_d64",  {256, 64, 64, 64, 64, 4, false},   1.0f, 1.0f, 101ULL);
    all_ok &= run_accuracy_case("std_normal_s512_d64",  {512, 64, 128, 64, 64, 4, false},  1.0f, 1.0f, 102ULL);
    all_ok &= run_accuracy_case("std_normal_s512_d128", {512, 128, 128, 64, 64, 4, false}, 1.0f, 1.0f, 103ULL);
    all_ok &= run_accuracy_case("std_normal_s1024_t128",{1024, 64, 256, 128, 128, 4, false},1.0f, 1.0f, 104ULL);

    // 2. Non-divisible sequence, head, block, and tile dimensions
    all_ok &= run_accuracy_case("non_div_s300_d64",     {300, 64, 128, 64, 64, 4, false},  1.0f, 1.0f, 201ULL);
    all_ok &= run_accuracy_case("non_div_s197_d67",     {197, 67, 64, 32, 48, 4, false},   1.0f, 1.0f, 202ULL);

    // 3. Thread-count bit-exactness (verifies race-free parallelization)
    all_ok &= run_thread_determinism_check();

    // 4. Large-logit stress tests (Proposal Bag. 6.2)
    // Logits ~ N(0, 10^2) when Q, K ~ N(0, sqrt(10)^2)
    all_ok &= run_accuracy_case("stress_logits_std10",  {512, 64, 128, 64, 64, 4, false},  std::sqrt(10.0f), 1.0f, 301ULL);
    // Extreme logits: Q, K ~ N(0, 10^2) -> logits std ~ 100
    all_ok &= run_accuracy_case("stress_qk_std10",      {512, 64, 128, 64, 64, 4, false},  10.0f, 1.0f, 302ULL);

    // 5. Optional causal mask check
    all_ok &= run_accuracy_case("causal_s256_d64",      {256, 64, 64, 64, 64, 4, true},    1.0f, 1.0f, 401ULL);

    std::cout << (all_ok ? "ALL V2 OPENMP TESTS PASSED\n" : "SOME V2 OPENMP TESTS FAILED\n");
    return all_ok ? 0 : 1;
}
