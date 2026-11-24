#include "common/gemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct TimingStats {
    double median_s = 0.0;
    double stddev_s = 0.0;
};

struct BenchArgs {
    int reps = 7;
    int warmup = 2;
    std::string csv_path;
};

BenchArgs parse_args(int argc, char** argv) {
    BenchArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto need_val = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };
        if (flag == "--reps") {
            args.reps = std::stoi(need_val("--reps"));
        } else if (flag == "--warmup") {
            args.warmup = std::stoi(need_val("--warmup"));
        } else if (flag == "--csv") {
            args.csv_path = need_val("--csv");
        } else {
            throw std::invalid_argument("Unknown option: " + flag);
        }
    }
    if (args.reps <= 0 || args.warmup < 0) {
        throw std::invalid_argument("reps must be > 0 and warmup >= 0");
    }
    return args;
}

TimingStats summarize_times(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const std::size_t n = samples.size();
    TimingStats st;
    st.median_s = (n % 2 == 1)
        ? samples[n / 2]
        : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);

    if (n > 1) {
        double mean = 0.0;
        for (double t : samples) mean += t;
        mean /= static_cast<double>(n);
        double sq = 0.0;
        for (double t : samples) {
            const double d = t - mean;
            sq += d * d;
        }
        st.stddev_s = std::sqrt(sq / static_cast<double>(n - 1));
    }
    return st;
}

void write_csv_row(
    const std::string& path,
    const char* backend,
    const char* op,
    int M, int N, int K,
    int reps,
    double median_s,
    double stddev_s,
    double gflops) {
    if (path.empty()) return;

    std::ifstream check_in(path);
    const bool needs_header = !check_in.good() || check_in.peek() == std::ifstream::traits_type::eof();
    check_in.close();

    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open CSV path: " + path);
    }
    if (needs_header) {
        out << "backend,op,M,N,K,reps,median_s,stddev_s,gflops\n";
    }
    out << backend << "," << op << "," << M << "," << N << "," << K << "," << reps << ","
        << std::scientific << std::setprecision(6) << median_s << ","
        << stddev_s << "," << std::fixed << std::setprecision(3) << gflops << "\n";
}

void bench_shape(
    bool trans_b, int M, int N, int K,
    const BenchArgs& args, std::mt19937_64& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> A(static_cast<std::size_t>(M) * K);
    std::vector<float> B(static_cast<std::size_t>(trans_b ? N : K) * (trans_b ? K : N));
    std::vector<float> C(static_cast<std::size_t>(M) * N, 0.0f);

    for (float& x : A) x = dist(rng);
    for (float& x : B) x = dist(rng);

    auto invoke = [&]() {
        if (trans_b) {
            ring_attn::gemm_nt_fp32(M, N, K, 1.0f, A.data(), K, B.data(), K, 0.0f, C.data(), N);
        } else {
            ring_attn::gemm_nn_fp32(M, N, K, 1.0f, A.data(), K, B.data(), N, 1.0f, C.data(), N);
        }
    };

    for (int w = 0; w < args.warmup; ++w) {
        invoke();
    }

    std::vector<double> samples(static_cast<std::size_t>(args.reps));
    for (int r = 0; r < args.reps; ++r) {
        const double t0 = omp_get_wtime();
        invoke();
        const double t1 = omp_get_wtime();
        samples[static_cast<std::size_t>(r)] = t1 - t0;
    }

    const TimingStats st = summarize_times(samples);
    const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    const double gflops = flops / std::max(st.median_s, 1e-12) / 1e9;
    const char* op = trans_b ? "NT" : "NN";

    std::cout << std::left << std::setw(4) << op
              << " | M=" << std::setw(4) << M
              << " N=" << std::setw(4) << N
              << " K=" << std::setw(3) << K
              << " | median=" << std::fixed << std::setprecision(4) << std::setw(9) << (st.median_s * 1e3) << " ms"
              << " | stddev=" << std::setw(8) << (st.stddev_s * 1e3) << " ms"
              << " | F=" << std::setprecision(2) << std::setw(7) << gflops << " GFLOP/s\n";

    write_csv_row(args.csv_path, ring_attn::gemm_backend_name(), op, M, N, K,
                  args.reps, st.median_s, st.stddev_s, gflops);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const BenchArgs args = parse_args(argc, argv);
        std::cout << "=== bench_gemm (backend: " << ring_attn::gemm_backend_name()
                  << ", reps=" << args.reps << ", warmup=" << args.warmup << ") ===\n";

        struct Shape { int M, N, K; };
        const Shape shapes[] = {
            {64, 64, 64},
            {128, 128, 64},
            {256, 256, 64},
            {512, 512, 64},
            {1024, 1024, 64},
            {512, 512, 128},
        };

        std::mt19937_64 rng(42ULL);
        for (const Shape& sh : shapes) {
            bench_shape(true, sh.M, sh.N, sh.K, args, rng);
            bench_shape(false, sh.M, sh.N, sh.K, args, rng);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "bench_gemm error: " << ex.what() << "\n";
        return 1;
    }
}
