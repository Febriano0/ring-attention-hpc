#include "common/attention_types.hpp"
#include "common/gemm.hpp"
#include "v2_openmp/v2_openmp.hpp"

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
#include <sys/resource.h>
#include <vector>

namespace {

struct ScalingArgs {
    int s = 4096;
    int d = 64;
    int c = 256;
    int bq = 64;
    int bk = 64;
    int reps = 5;
    int warmup = 1;
    std::uint64_t seed = 42ULL;
    std::string csv_path;
};

struct TimingStats {
    double median_s = 0.0;
    double stddev_s = 0.0;
};

ScalingArgs parse_args(int argc, char** argv) {
    ScalingArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto need_val = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };
        if (flag == "--s") args.s = std::stoi(need_val("--s"));
        else if (flag == "--d") args.d = std::stoi(need_val("--d"));
        else if (flag == "--c") args.c = std::stoi(need_val("--c"));
        else if (flag == "--bq") args.bq = std::stoi(need_val("--bq"));
        else if (flag == "--bk") args.bk = std::stoi(need_val("--bk"));
        else if (flag == "--reps") args.reps = std::stoi(need_val("--reps"));
        else if (flag == "--warmup") args.warmup = std::stoi(need_val("--warmup"));
        else if (flag == "--seed") args.seed = std::stoull(need_val("--seed"));
        else if (flag == "--csv") args.csv_path = need_val("--csv");
        else throw std::invalid_argument("Unknown option: " + flag);
    }
    if (args.s <= 0 || args.d <= 0 || args.c <= 0 || args.bq <= 0 || args.bk <= 0 ||
        args.reps <= 0 || args.warmup < 0) {
        throw std::invalid_argument("Invalid positive dimension or repetition argument");
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

long read_peak_rss_kb() {
    struct rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) != 0) {
        return -1;
    }
    return ru.ru_maxrss;
}

void write_csv_row(
    const std::string& path,
    const ScalingArgs& args,
    int P,
    double time_s,
    double stddev_s,
    double gflops,
    double speedup,
    double efficiency,
    long peak_rss_kb) {
    if (path.empty()) return;

    std::ifstream check_in(path);
    const bool needs_header = !check_in.good() || check_in.peek() == std::ifstream::traits_type::eof();
    check_in.close();

    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open CSV path: " + path);
    }
    if (needs_header) {
        out << "version,s,d,c,bq,bk,P,seed,time_s,stddev_s,gflops,speedup,efficiency,peak_rss_kb\n";
    }
    out << "v2," << args.s << "," << args.d << "," << args.c << ","
        << args.bq << "," << args.bk << "," << P << "," << args.seed << ","
        << std::scientific << std::setprecision(6) << time_s << ","
        << stddev_s << ","
        << std::fixed << std::setprecision(3) << gflops << ","
        << speedup << "," << efficiency << "," << peak_rss_kb << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const ScalingArgs args = parse_args(argc, argv);
        const std::size_t n = static_cast<std::size_t>(args.s) * args.d;

        std::vector<float> Q(n), K(n), V(n), O(n, 0.0f);
        std::mt19937_64 rng(args.seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (std::size_t i = 0; i < n; ++i) {
            Q[i] = dist(rng);
            K[i] = dist(rng);
            V[i] = dist(rng);
        }

        std::cout << "=== bench_v2_scaling (GEMM backend: " << ring_attn::gemm_backend_name()
                  << ", s=" << args.s << ", d=" << args.d << ", c=" << args.c
                  << ", tile=" << args.bq << "x" << args.bk
                  << ", reps=" << args.reps << ", warmup=" << args.warmup << ") ===\n";
        std::cout << "P |   median_s |   stddev_s |  GFLOP/s | Speedup | Efficiency | peak_rss_kb\n";
        std::cout << "--+------------+------------+----------+---------+------------+------------\n";

        const int thread_counts[] = {1, 2, 4};
        double t1_s = 0.0;
        const double total_flops = 4.0 * static_cast<double>(args.s) * static_cast<double>(args.s)
                                   * static_cast<double>(args.d);

        for (int P : thread_counts) {
            ring_attn::AttentionConfig cfg;
            cfg.s = args.s;
            cfg.d = args.d;
            cfg.c = args.c;
            cfg.bq = args.bq;
            cfg.bk = args.bk;
            cfg.num_threads = P;
            cfg.causal = false;

            for (int w = 0; w < args.warmup; ++w) {
                ring_attn::attention_v2_openmp(Q.data(), K.data(), V.data(), O.data(), cfg);
            }

            std::vector<double> samples(static_cast<std::size_t>(args.reps));
            for (int r = 0; r < args.reps; ++r) {
                const double t0 = omp_get_wtime();
                ring_attn::attention_v2_openmp(Q.data(), K.data(), V.data(), O.data(), cfg);
                const double t1 = omp_get_wtime();
                samples[static_cast<std::size_t>(r)] = t1 - t0;
            }

            const TimingStats st = summarize_times(samples);
            if (P == 1) {
                t1_s = st.median_s;
            }
            const double gflops = total_flops / std::max(st.median_s, 1e-12) / 1e9;
            const double speedup = t1_s / std::max(st.median_s, 1e-12);
            const double efficiency = speedup / static_cast<double>(P);
            const long peak_rss_kb = read_peak_rss_kb();

            std::cout << std::setw(1) << P << " | "
                      << std::fixed << std::setprecision(5) << std::setw(10) << st.median_s << " | "
                      << std::setw(10) << st.stddev_s << " | "
                      << std::setprecision(2) << std::setw(8) << gflops << " | "
                      << std::setw(6) << speedup << "x | "
                      << std::setprecision(1) << std::setw(9) << (efficiency * 100.0) << "% | "
                      << std::setw(11) << peak_rss_kb << "\n";

            write_csv_row(args.csv_path, args, P, st.median_s, st.stddev_s,
                          gflops, speedup, efficiency, peak_rss_kb);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "bench_v2_scaling error: " << ex.what() << "\n";
        return 1;
    }
}
