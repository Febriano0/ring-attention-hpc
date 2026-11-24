#include "common/attention_types.hpp"
#include "common/gemm.hpp"
#include "v4_hybrid/v4_hybrid.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <omp.h>
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

bool run_hybrid_case(
    const std::string& name,
    const ring_attn::HybridRingConfig& cfg,
    float qk_std, float v_std, std::uint64_t seed,
    int rank, int world_size,
    double rel_tol = 1e-5) {
    const int s = world_size * cfg.c;
    const std::size_t local_elems = static_cast<std::size_t>(cfg.c) * cfg.d;
    const int mpi_chunk = static_cast<int>(local_elems);

    std::vector<float> Q_full, K_full, V_full, O_full;
    std::vector<double> Q64, K64, V64, O_ref;

    if (rank == 0) {
        generate_tensors(s, cfg.d, qk_std, v_std, seed, Q_full, K_full, V_full, Q64, K64, V64);
        O_full.resize(static_cast<std::size_t>(s) * cfg.d, 0.0f);
        O_ref.resize(static_cast<std::size_t>(s) * cfg.d, 0.0);
    }

    std::vector<float> Q_local(local_elems);
    std::vector<float> K_local(local_elems);
    std::vector<float> V_local(local_elems);
    std::vector<float> O_local(local_elems, 0.0f);

    MPI_Scatter(rank == 0 ? Q_full.data() : nullptr, mpi_chunk, MPI_FLOAT,
                Q_local.data(), mpi_chunk, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(rank == 0 ? K_full.data() : nullptr, mpi_chunk, MPI_FLOAT,
                K_local.data(), mpi_chunk, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(rank == 0 ? V_full.data() : nullptr, mpi_chunk, MPI_FLOAT,
                V_local.data(), mpi_chunk, MPI_FLOAT, 0, MPI_COMM_WORLD);

    ring_attn::attention_v4_hybrid(
        Q_local.data(), K_local.data(), V_local.data(), O_local.data(), cfg, MPI_COMM_WORLD);

    MPI_Gather(O_local.data(), mpi_chunk, MPI_FLOAT,
               rank == 0 ? O_full.data() : nullptr, mpi_chunk, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int ok_int = 1;
    if (rank == 0) {
        reference_attention_fp64(Q64, K64, V64, O_ref, s, cfg.d, cfg.causal);
        const ring_attn::ErrorMetrics m = ring_attn::compute_error_metrics(
            O_full.data(), O_ref.data(), static_cast<std::size_t>(s) * cfg.d);
        const bool ok = m.all_finite && (m.rel_l2_err < rel_tol);
        ok_int = ok ? 1 : 0;

        const int threads = (cfg.num_threads > 0) ? cfg.num_threads : omp_get_max_threads();
        std::cout << std::left << std::setw(22) << name
                  << " | s=" << std::setw(4) << s
                  << " c=" << std::setw(3) << cfg.c
                  << " d=" << std::setw(3) << cfg.d
                  << " np=" << world_size
                  << " thr=" << threads
                  << (cfg.causal ? " C" : "  ")
                  << " | max_abs=" << std::scientific << std::setprecision(3) << m.max_abs_err
                  << " | rel_l2=" << m.rel_l2_err
                  << " | " << (ok ? "PASS" : "FAIL") << "\n";
    }
    MPI_Bcast(&ok_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return ok_int == 1;
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::cerr << "MPI implementation does not support MPI_THREAD_FUNNELED (got "
                      << provided << ")\n";
        }
        MPI_Finalize();
        return 1;
    }

    const int threads = (omp_get_max_threads() >= 2) ? 2 : 1;
    if (rank == 0) {
        std::cout << "=== test_v4_hybrid (GEMM backend: " << ring_attn::gemm_backend_name()
                  << ", ranks=" << world_size << ", omp_threads=" << threads
                  << ", mpi_thread_level=" << provided << ") ===\n";
    }

    bool all_ok = true;
    all_ok &= run_hybrid_case("std_normal_c128_d64",  {128, 64, 64, 64, 64, threads, false},  1.0f, 1.0f, 501ULL, rank, world_size);
    all_ok &= run_hybrid_case("std_normal_c256_d128", {256, 128, 64, 64, 64, threads, false}, 1.0f, 1.0f, 502ULL, rank, world_size);
    all_ok &= run_hybrid_case("non_div_c150_d67",     {150, 67, 48, 32, 48, threads, false},  1.0f, 1.0f, 503ULL, rank, world_size);
    all_ok &= run_hybrid_case("stress_logits_std10",  {256, 64, 64, 64, 64, threads, false},  std::sqrt(10.0f), 1.0f, 504ULL, rank, world_size);
    all_ok &= run_hybrid_case("stress_qk_std10",      {256, 64, 64, 64, 64, threads, false},  10.0f, 1.0f, 505ULL, rank, world_size);
    all_ok &= run_hybrid_case("causal_c128_d64",      {128, 64, 64, 64, 64, threads, true},   1.0f, 1.0f, 506ULL, rank, world_size);

    if (rank == 0) {
        std::cout << (all_ok ? "ALL V4 HYBRID TESTS PASSED\n" : "SOME V4 HYBRID TESTS FAILED\n");
    }

    MPI_Finalize();
    return all_ok ? 0 : 1;
}
