#include "common/gemm.hpp"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct GemmError {
    double max_abs = 0.0;
    double rel_l2 = 0.0;
    bool all_finite = true;
};

void ref_gemm_fp64(bool trans_b, int M, int N, int K,
                   double alpha,
                   const float* A, int lda,
                   const float* B, int ldb,
                   double beta,
                   std::vector<double>& C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            const std::size_t idx = static_cast<std::size_t>(i) * ldc + j;
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                const double a = static_cast<double>(A[static_cast<std::size_t>(i) * lda + k]);
                const double b = trans_b
                    ? static_cast<double>(B[static_cast<std::size_t>(j) * ldb + k])
                    : static_cast<double>(B[static_cast<std::size_t>(k) * ldb + j]);
                sum += a * b;
            }
            const double prev = (beta == 0.0) ? 0.0 : beta * C[idx];
            C[idx] = alpha * sum + prev;
        }
    }
}

GemmError measure_error(const std::vector<float>& got, const std::vector<double>& ref,
                        int M, int N, int ldc) {
    GemmError err{};
    double diff_sq = 0.0;
    double ref_sq = 0.0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            const std::size_t idx = static_cast<std::size_t>(i) * ldc + j;
            const float v = got[idx];
            if (!std::isfinite(v)) {
                err.all_finite = false;
            }
            const double d = static_cast<double>(v) - ref[idx];
            err.max_abs = std::max(err.max_abs, std::abs(d));
            diff_sq += d * d;
            ref_sq += ref[idx] * ref[idx];
        }
    }
    const double denom = std::max(std::sqrt(ref_sq), 1e-12);
    err.rel_l2 = std::sqrt(diff_sq) / denom;
    return err;
}

bool run_case(const std::string& label, bool trans_b,
              int M, int N, int K,
              int lda, int ldb, int ldc,
              float alpha, float beta,
              bool poison_c_with_nan,
              std::mt19937_64& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);

    const int b_rows = trans_b ? N : K;
    std::vector<float> A(static_cast<std::size_t>(M) * lda, 0.0f);
    std::vector<float> B(static_cast<std::size_t>(b_rows) * ldb, 0.0f);
    std::vector<float> C_got(static_cast<std::size_t>(M) * ldc, 0.0f);
    std::vector<double> C_ref(static_cast<std::size_t>(M) * ldc, 0.0);

    for (float& x : A) x = dist(rng);
    for (float& x : B) x = dist(rng);
    for (std::size_t idx = 0; idx < C_got.size(); ++idx) {
        if (poison_c_with_nan) {
            C_got[idx] = (idx % 2 == 0)
                ? std::numeric_limits<float>::quiet_NaN()
                : std::numeric_limits<float>::infinity();
            C_ref[idx] = 0.0;
        } else {
            const float val = dist(rng);
            C_got[idx] = val;
            C_ref[idx] = static_cast<double>(val);
        }
    }

    if (trans_b) {
        ring_attn::gemm_nt_fp32(M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C_got.data(), ldc);
    } else {
        ring_attn::gemm_nn_fp32(M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C_got.data(), ldc);
    }

    ref_gemm_fp64(trans_b, M, N, K, static_cast<double>(alpha),
                  A.data(), lda, B.data(), ldb, static_cast<double>(beta), C_ref, ldc);

    const GemmError err = measure_error(C_got, C_ref, M, N, ldc);
    const bool ok = err.all_finite && err.rel_l2 < 1e-5;

    std::cout << std::left << std::setw(30) << label
              << " | " << (trans_b ? "NT" : "NN")
              << " | M=" << std::setw(3) << M
              << " N=" << std::setw(3) << N
              << " K=" << std::setw(3) << K
              << " | max_abs=" << std::scientific << std::setprecision(2) << err.max_abs
              << " | rel_l2=" << err.rel_l2
              << " | " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool test_invalid_arguments() {
    float buf[16] = {};
    auto expect_throw = [](auto&& fn) {
        try {
            fn();
            return false;
        } catch (const std::invalid_argument&) {
            return true;
        }
    };

    bool ok = true;
    ok &= expect_throw([&] { ring_attn::gemm_nt_fp32(-1, 4, 4, 1.0f, buf, 4, buf, 4, 0.0f, buf, 4); });
    ok &= expect_throw([&] { ring_attn::gemm_nn_fp32(4, 4, 4, 1.0f, buf, 3, buf, 4, 0.0f, buf, 4); });
    ok &= expect_throw([&] { ring_attn::gemm_nt_fp32(4, 4, 4, 1.0f, nullptr, 4, buf, 4, 0.0f, buf, 4); });
    ok &= expect_throw([&] { ring_attn::gemm_nn_fp32(4, 4, 4, 1.0f, buf, 4, buf, 4, 0.0f, buf, 2); });

    std::cout << std::left << std::setw(30) << "invalid_arguments"
              << " | -- | checks=4          | max_abs=--       | rel_l2=--       | "
              << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

} // namespace

int main() {
    std::cout << "=== test_gemm (backend: " << ring_attn::gemm_backend_name() << ") ===\n";
    std::mt19937_64 rng(20261124ULL);
    bool all_ok = true;

    const float inv_sqrt_64 = 1.0f / std::sqrt(64.0f);

    // Attention tile multiples
    all_ok &= run_case("tile_64x64x64_nt", true, 64, 64, 64, 64, 64, 64, inv_sqrt_64, 0.0f, false, rng);
    all_ok &= run_case("tile_64x64x64_nn", false, 64, 64, 64, 64, 64, 64, 1.0f, 1.0f, false, rng);
    all_ok &= run_case("tile_128x128x64_nt", true, 128, 128, 64, 64, 64, 128, inv_sqrt_64, 0.0f, false, rng);
    all_ok &= run_case("tile_128x64x128_nn", false, 128, 64, 128, 128, 64, 64, 1.0f, 1.0f, false, rng);

    // Awkward / tail dimensions
    all_ok &= run_case("tail_1x1x1_nt", true, 1, 1, 1, 1, 1, 1, 1.0f, 0.0f, false, rng);
    all_ok &= run_case("tail_5x7x3_nn", false, 5, 7, 3, 3, 7, 7, 1.0f, 0.5f, false, rng);
    all_ok &= run_case("tail_13x19x17_nt", true, 13, 19, 17, 17, 17, 19, -0.75f, 0.25f, false, rng);
    all_ok &= run_case("tail_67x71x65_nn", false, 67, 71, 65, 65, 71, 71, 1.0f, 1.0f, false, rng);
    all_ok &= run_case("tail_131x63x97_nt", true, 131, 63, 97, 97, 97, 63, 0.5f, 0.0f, false, rng);

    // Strided submatrices
    all_ok &= run_case("strided_nt", true, 67, 73, 70, 77, 75, 84, inv_sqrt_64, 0.35f, false, rng);
    all_ok &= run_case("strided_nn", false, 67, 73, 70, 77, 78, 84, 1.0f, 0.35f, false, rng);

    // beta=0 with NaN/Inf in C, and alpha=0
    all_ok &= run_case("beta0_nan_inf_nt", true, 65, 65, 64, 64, 64, 65, 1.0f, 0.0f, true, rng);
    all_ok &= run_case("beta0_nan_inf_nn", false, 65, 65, 64, 64, 65, 65, 1.0f, 0.0f, true, rng);
    all_ok &= run_case("alpha0_beta_scale", false, 33, 29, 16, 16, 29, 29, 0.0f, 0.5f, false, rng);

    // Invalid argument handling
    all_ok &= test_invalid_arguments();

    std::cout << (all_ok ? "ALL GEMM TESTS PASSED\n" : "SOME GEMM TESTS FAILED\n");
    return all_ok ? 0 : 1;
}
