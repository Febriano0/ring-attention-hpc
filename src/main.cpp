#include "common/attention_types.hpp"
#include "common/gemm.hpp"
#include "common/memory_utils.hpp"
#include "v0_naive/v0_naive.hpp"
#include "v1_blockwise/v1_blockwise.hpp"
#include "v2_openmp/v2_openmp.hpp"
#include "v3_ring_mpi/v3_ring_mpi.hpp"
#include "v4_hybrid/v4_hybrid.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <new>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliArgs {
    std::string version = "v3";
    int s = 1024;
    bool s_set = false;
    int d = 64;
    int c = 256;
    bool c_set = false;
    int bq = 64;
    int bk = 64;
    int sub_block_q = 64;
    int threads = 0;
    std::uint64_t seed = 1ULL;
    int reps = 1;
    int warmup = 0;
    bool blocking = false;
    bool no_comm = false;
    bool causal = false;
    bool stress_logits = false;
    bool verify = false;
    std::string csv_path;
    std::string weights_dir;
    std::string input_path;
    std::string json_out;
};

struct TimingStats {
    double median_s = 0.0;
    double stddev_s = 0.0;
};

void print_usage() {
    std::cout << "Usage: ./build/attn [options]\n"
              << "  --version <v0_fp64|v0|v1|v2|v3|v4>\n"
              << "  --s <int>             Total sequence length\n"
              << "  --d <int>             Head dimension (default: 64)\n"
              << "  --c <int>             Block size / local rank rows (default: 256)\n"
              << "  --tile <int>          Square internal tile size bq=bk (default: 64)\n"
              << "  --bq <int>            Internal query tile size\n"
              << "  --bk <int>            Internal key tile size\n"
              << "  --sub-block-q <int>   Intra-rank OpenMP sub-block for v4 (default: 64)\n"
              << "  --threads <int>       OpenMP threads for v2/v4\n"
              << "  --seed <uint64>       RNG seed (default: 1)\n"
              << "  --reps <int>          Timed repetitions (default: 1)\n"
              << "  --warmup <int>        Warmup repetitions (default: 0)\n"
              << "  --blocking            Use MPI_Sendrecv for v3 (T_blk)\n"
              << "  --no-comm             Disable MPI comm for v3 (T_comp)\n"
              << "  --causal              Enable causal mask\n"
              << "  --stress-logits       Generate N(0, sqrt(10)^2) Q/K -> N(0, 10^2) logits\n"
              << "  --verify              Verify numerical accuracy against v0_fp64 (< 1e-5)\n"
              << "  --csv <path>          Append result row to CSV file\n"
              << "  --weights <dir>       Weight directory for DNA triage inference\n"
              << "  --input <tokens.bin>  Tokenized DNA binary file for inference\n"
              << "  --json-out <path>     Write inference summary JSON\n";
}

CliArgs parse_cli(int argc, char** argv) {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto need_val = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };
        if (flag == "--help" || flag == "-h") {
            print_usage();
            std::exit(0);
        } else if (flag == "--version") {
            args.version = need_val("--version");
        } else if (flag == "--s") {
            args.s = std::stoi(need_val("--s"));
            args.s_set = true;
        } else if (flag == "--d") {
            args.d = std::stoi(need_val("--d"));
        } else if (flag == "--c") {
            args.c = std::stoi(need_val("--c"));
            args.c_set = true;
        } else if (flag == "--tile") {
            const int t = std::stoi(need_val("--tile"));
            args.bq = t;
            args.bk = t;
        } else if (flag == "--bq") {
            args.bq = std::stoi(need_val("--bq"));
        } else if (flag == "--bk") {
            args.bk = std::stoi(need_val("--bk"));
        } else if (flag == "--sub-block-q") {
            args.sub_block_q = std::stoi(need_val("--sub-block-q"));
        } else if (flag == "--threads") {
            args.threads = std::stoi(need_val("--threads"));
        } else if (flag == "--seed") {
            args.seed = std::stoull(need_val("--seed"));
        } else if (flag == "--reps") {
            args.reps = std::stoi(need_val("--reps"));
        } else if (flag == "--warmup") {
            args.warmup = std::stoi(need_val("--warmup"));
        } else if (flag == "--blocking") {
            args.blocking = true;
        } else if (flag == "--no-comm") {
            args.no_comm = true;
        } else if (flag == "--causal") {
            args.causal = true;
        } else if (flag == "--stress-logits") {
            args.stress_logits = true;
        } else if (flag == "--verify") {
            args.verify = true;
        } else if (flag == "--csv") {
            args.csv_path = need_val("--csv");
        } else if (flag == "--weights") {
            args.weights_dir = need_val("--weights");
        } else if (flag == "--input") {
            args.input_path = need_val("--input");
        } else if (flag == "--json-out") {
            args.json_out = need_val("--json-out");
        } else {
            throw std::invalid_argument("Unknown CLI flag: " + flag);
        }
    }
    return args;
}

void resolve_and_validate_args(CliArgs& args, int world_size) {
    const bool is_dist = (args.version == "v3" || args.version == "v4");
    const bool valid_ver = is_dist || args.version == "v0_fp64" || args.version == "v0" ||
                           args.version == "v1" || args.version == "v2";
    if (!valid_ver) {
        throw std::invalid_argument("Unsupported --version: " + args.version);
    }
    if (args.d <= 0 || args.bq <= 0 || args.bk <= 0 || args.sub_block_q <= 0 ||
        args.reps <= 0 || args.warmup < 0 || args.threads < 0) {
        throw std::invalid_argument("Invalid positive dimension, thread, or repetition parameter");
    }
    if (args.blocking && args.no_comm) {
        throw std::invalid_argument("--blocking and --no-comm are mutually exclusive");
    }
    if (!is_dist && world_size != 1) {
        throw std::invalid_argument(args.version + " must run with 1 MPI rank");
    }
    if (is_dist) {
        if (args.c_set && !args.s_set) {
            args.s = world_size * args.c;
        } else if (!args.c_set) {
            if (args.s <= 0 || (args.s % world_size) != 0) {
                throw std::invalid_argument("Total s must be positive and divisible by P for v3/v4");
            }
            args.c = args.s / world_size;
        } else {
            if (args.c <= 0 || args.s != world_size * args.c) {
                throw std::invalid_argument("For v3/v4, s must equal P * c");
            }
        }
        if (args.no_comm && args.verify && world_size > 1) {
            throw std::invalid_argument("--verify cannot be used with --no-comm when P > 1");
        }
    } else if (args.s <= 0 || args.c <= 0) {
        throw std::invalid_argument("s and c must be positive");
    }
}

TimingStats summarize_times(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const std::size_t n = samples.size();
    TimingStats st{};
    st.median_s = (n % 2 == 1)
        ? samples[n / 2]
        : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
    if (n > 1) {
        double mean = 0.0;
        for (double v : samples) mean += v;
        mean /= static_cast<double>(n);
        double sq = 0.0;
        for (double v : samples) {
            const double diff = v - mean;
            sq += diff * diff;
        }
        st.stddev_s = std::sqrt(sq / static_cast<double>(n - 1));
    }
    return st;
}

void fill_normal_fp32(float* dst, std::size_t count, float stddev, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, stddev);
    for (std::size_t i = 0; i < count; ++i) {
        dst[i] = dist(rng);
    }
}

double compute_theoretical_alloc_kb(const CliArgs& args, int effective_threads) {
    const double c = static_cast<double>(std::min(args.c, args.s));
    const double s = static_cast<double>(args.s);
    const double d = static_cast<double>(args.d);
    const double bq = static_cast<double>(std::min(args.bq, static_cast<int>(c)));
    const double bk = static_cast<double>(std::min(args.bk, static_cast<int>(c)));

    double bytes = 0.0;
    if (args.version == "v0_fp64") {
        bytes = (4.0 * s * d + 2.0 * s) * 8.0;
    } else if (args.version == "v0") {
        bytes = (4.0 * s * d + s * s) * 4.0;
    } else if (args.version == "v1") {
        bytes = (4.0 * s * d + 2.0 * c + bq * bk) * 4.0;
    } else if (args.version == "v2") {
        bytes = (4.0 * s * d + static_cast<double>(effective_threads) * (2.0 * c + bq * bk)) * 4.0;
    } else if (args.version == "v3") {
        bytes = (6.0 * c * d + 2.0 * c + bq * bk) * 4.0;
    } else {
        const double sub = static_cast<double>(std::min(args.sub_block_q, args.c));
        const double bq4 = static_cast<double>(std::min(args.bq, static_cast<int>(sub)));
        bytes = (6.0 * c * d + 2.0 * c + static_cast<double>(effective_threads) * bq4 * bk) * 4.0;
    }
    return bytes / 1024.0;
}

void append_csv(
    const CliArgs& args,
    const std::string& comm_mode,
    int world_size,
    int effective_threads,
    const TimingStats& st,
    double wait_s,
    long peak_rss_kb,
    long delta_rss_kb,
    double alloc_kb,
    const ring_attn::ErrorMetrics& err,
    int verified) {
    if (args.csv_path.empty()) {
        return;
    }
    std::ifstream check_in(args.csv_path);
    const bool needs_header = !check_in.good() || check_in.peek() == std::ifstream::traits_type::eof();
    check_in.close();

    std::ofstream out(args.csv_path, std::ios::app);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open CSV path: " + args.csv_path);
    }
    if (needs_header) {
        out << "version,comm_mode,s,d,c,tile,P,threads,seed,time_s,stddev_s,wait_s,"
               "peak_rss_kb,delta_rss_kb,alloc_kb,max_abs_err,rel_err,verified\n";
    }
    out << args.version << "," << comm_mode << "," << args.s << "," << args.d << "," << args.c << ","
        << args.bq << "," << world_size << "," << effective_threads << "," << args.seed << ","
        << std::scientific << std::setprecision(6) << st.median_s << "," << st.stddev_s << ","
        << wait_s << "," << peak_rss_kb << "," << delta_rss_kb << ","
        << std::fixed << std::setprecision(2) << alloc_kb << ","
        << std::scientific << std::setprecision(6) << err.max_abs_err << ","
        << err.rel_l2_err << "," << verified << "\n";
}

int run_synthetic_benchmark(const CliArgs& args, int rank, int world_size, long rss_baseline_kb) {
    const float qk_std = args.stress_logits ? std::sqrt(10.0f) : 1.0f;
    const bool is_dist = (args.version == "v3" || args.version == "v4");
    const int eff_threads = (args.version == "v2" || args.version == "v4")
        ? ((args.threads > 0) ? args.threads : omp_get_max_threads())
        : 1;

    const ring_attn::RingCommMode ring_mode = args.no_comm
        ? ring_attn::RingCommMode::NoComm
        : (args.blocking ? ring_attn::RingCommMode::Blocking : ring_attn::RingCommMode::NonBlocking);
    const std::string comm_mode_str = (args.version == "v3")
        ? (args.no_comm ? "no_comm" : (args.blocking ? "blocking" : "non_blocking"))
        : ((args.version == "v4") ? "non_blocking" : "none");

    const std::size_t full_elems = static_cast<std::size_t>(args.s) * static_cast<std::size_t>(args.d);
    const std::size_t local_elems = static_cast<std::size_t>(args.c) * static_cast<std::size_t>(args.d);

    std::vector<float> Q_full, K_full, V_full, O_full;
    std::vector<double> Q64, K64, V64, O64;
    std::vector<float> Q_loc, K_loc, V_loc, O_loc;

    if (!is_dist) {
        if (args.version == "v0_fp64") {
            std::vector<float> tmp_q(full_elems), tmp_k(full_elems), tmp_v(full_elems);
            fill_normal_fp32(tmp_q.data(), full_elems, qk_std, args.seed);
            fill_normal_fp32(tmp_k.data(), full_elems, qk_std, args.seed + 1ULL);
            fill_normal_fp32(tmp_v.data(), full_elems, 1.0f, args.seed + 2ULL);
            Q64.resize(full_elems); K64.resize(full_elems); V64.resize(full_elems); O64.resize(full_elems);
            for (std::size_t i = 0; i < full_elems; ++i) {
                Q64[i] = static_cast<double>(tmp_q[i]);
                K64[i] = static_cast<double>(tmp_k[i]);
                V64[i] = static_cast<double>(tmp_v[i]);
            }
        } else {
            Q_full.resize(full_elems); K_full.resize(full_elems);
            V_full.resize(full_elems); O_full.resize(full_elems, 0.0f);
            fill_normal_fp32(Q_full.data(), full_elems, qk_std, args.seed);
            fill_normal_fp32(K_full.data(), full_elems, qk_std, args.seed + 1ULL);
            fill_normal_fp32(V_full.data(), full_elems, 1.0f, args.seed + 2ULL);
        }
    } else {
        Q_loc.resize(local_elems); K_loc.resize(local_elems);
        V_loc.resize(local_elems); O_loc.resize(local_elems, 0.0f);
        if (args.verify) {
            if (rank == 0) {
                Q_full.resize(full_elems); K_full.resize(full_elems);
                V_full.resize(full_elems); O_full.resize(full_elems, 0.0f);
                fill_normal_fp32(Q_full.data(), full_elems, qk_std, args.seed);
                fill_normal_fp32(K_full.data(), full_elems, qk_std, args.seed + 1ULL);
                fill_normal_fp32(V_full.data(), full_elems, 1.0f, args.seed + 2ULL);
            }
            const int chunk = static_cast<int>(local_elems);
            MPI_Scatter(rank == 0 ? Q_full.data() : nullptr, chunk, MPI_FLOAT,
                        Q_loc.data(), chunk, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Scatter(rank == 0 ? K_full.data() : nullptr, chunk, MPI_FLOAT,
                        K_loc.data(), chunk, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Scatter(rank == 0 ? V_full.data() : nullptr, chunk, MPI_FLOAT,
                        V_loc.data(), chunk, MPI_FLOAT, 0, MPI_COMM_WORLD);
        } else {
            // Non-verify mode: strictly O(c*d) allocation per rank (no full s*d tensor on rank 0).
            const std::uint64_t r_off = static_cast<std::uint64_t>(rank) * 1315423911ULL;
            fill_normal_fp32(Q_loc.data(), local_elems, qk_std, args.seed + r_off);
            fill_normal_fp32(K_loc.data(), local_elems, qk_std, args.seed + 1ULL + r_off);
            fill_normal_fp32(V_loc.data(), local_elems, 1.0f, args.seed + 2ULL + r_off);
        }
    }

    double last_wait_s = 0.0;
    auto invoke_once = [&]() {
        if (args.version == "v0_fp64") {
            ring_attn::attention_v0_fp64(Q64.data(), K64.data(), V64.data(), O64.data(), args.s, args.d, args.causal);
        } else if (args.version == "v0") {
            ring_attn::attention_v0_fp32(Q_full.data(), K_full.data(), V_full.data(), O_full.data(), args.s, args.d, args.causal);
        } else if (args.version == "v1") {
            ring_attn::AttentionConfig cfg{args.s, args.d, args.c, args.bq, args.bk, 1, args.causal};
            ring_attn::attention_v1_blockwise(Q_full.data(), K_full.data(), V_full.data(), O_full.data(), cfg);
        } else if (args.version == "v2") {
            ring_attn::AttentionConfig cfg{args.s, args.d, args.c, args.bq, args.bk, eff_threads, args.causal};
            ring_attn::attention_v2_openmp(Q_full.data(), K_full.data(), V_full.data(), O_full.data(), cfg);
        } else if (args.version == "v3") {
            ring_attn::RingConfig rcfg{args.c, args.d, args.bq, args.bk, args.causal, ring_mode};
            const ring_attn::RingStats st = ring_attn::attention_v3_ring_mpi(
                Q_loc.data(), K_loc.data(), V_loc.data(), O_loc.data(), rcfg, MPI_COMM_WORLD);
            last_wait_s = st.wait_s;
        } else {
            ring_attn::HybridRingConfig hcfg{args.c, args.d, args.sub_block_q, args.bq, args.bk, eff_threads, args.causal};
            ring_attn::attention_v4_hybrid(
                Q_loc.data(), K_loc.data(), V_loc.data(), O_loc.data(), hcfg, MPI_COMM_WORLD);
        }
    };

    for (int w = 0; w < args.warmup; ++w) {
        invoke_once();
    }

    std::vector<double> samples(static_cast<std::size_t>(args.reps));
    for (int r = 0; r < args.reps; ++r) {
        if (is_dist) MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = MPI_Wtime();
        invoke_once();
        const double local_dt = MPI_Wtime() - t0;
        if (is_dist) {
            double max_dt = 0.0;
            MPI_Reduce(&local_dt, &max_dt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            samples[static_cast<std::size_t>(r)] = max_dt;
        } else {
            samples[static_cast<std::size_t>(r)] = local_dt;
        }
    }

    double max_wait_s = last_wait_s;
    if (is_dist) {
        MPI_Reduce(&last_wait_s, &max_wait_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }

    const long local_peak = std::max(ring_attn::read_vm_hwm_kb(), ring_attn::read_vm_rss_kb());
    const long local_delta = (rss_baseline_kb >= 0 && local_peak >= 0)
        ? std::max(0L, local_peak - rss_baseline_kb)
        : -1L;
    long peak_rss_kb = local_peak;
    long delta_rss_kb = local_delta;
    if (is_dist) {
        MPI_Reduce(&local_peak, &peak_rss_kb, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_delta, &delta_rss_kb, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    }

    ring_attn::ErrorMetrics err{};
    int verified = (args.version == "v0_fp64") ? 1 : -1;
    if (args.verify && args.version != "v0_fp64") {
        if (is_dist) {
            const int chunk = static_cast<int>(local_elems);
            MPI_Gather(O_loc.data(), chunk, MPI_FLOAT,
                       rank == 0 ? O_full.data() : nullptr, chunk, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
        if (rank == 0) {
            Q64.resize(full_elems); K64.resize(full_elems); V64.resize(full_elems); O64.resize(full_elems);
            for (std::size_t i = 0; i < full_elems; ++i) {
                Q64[i] = static_cast<double>(Q_full[i]);
                K64[i] = static_cast<double>(K_full[i]);
                V64[i] = static_cast<double>(V_full[i]);
            }
            ring_attn::attention_v0_fp64(Q64.data(), K64.data(), V64.data(), O64.data(), args.s, args.d, args.causal);
            err = ring_attn::compute_error_metrics(O_full.data(), O64.data(), full_elems);
            verified = (err.all_finite && err.rel_l2_err < 1e-5) ? 1 : 0;
        }
        if (is_dist) {
            MPI_Bcast(&verified, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        const TimingStats st = summarize_times(samples);
        const double alloc_kb = compute_theoretical_alloc_kb(args, eff_threads);
        std::cout << "[attn] version=" << args.version
                  << " comm=" << comm_mode_str
                  << " s=" << args.s << " d=" << args.d << " c=" << args.c
                  << " tile=" << args.bq << "x" << args.bk
                  << " P=" << world_size << " thr=" << eff_threads
                  << " | time=" << std::fixed << std::setprecision(5) << st.median_s << "s"
                  << " wait=" << max_wait_s << "s"
                  << " peak_rss=" << peak_rss_kb << "KB"
                  << " delta_rss=" << delta_rss_kb << "KB";
        if (args.verify) {
            std::cout << " | max_abs=" << std::scientific << std::setprecision(3) << err.max_abs_err
                      << " rel_l2=" << err.rel_l2_err
                      << " verify=" << (verified == 1 ? "PASS" : "FAIL");
        }
        std::cout << "\n";
        append_csv(args, comm_mode_str, world_size, eff_threads, st, max_wait_s,
                   peak_rss_kb, delta_rss_kb, alloc_kb, err, verified);
    }

    return (args.verify && verified == 0) ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    const long rss_baseline_kb = ring_attn::read_vm_rss_kb();

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int exit_code = 0;
    try {
        CliArgs args = parse_cli(argc, argv);
        resolve_and_validate_args(args, world_size);

        if (!args.weights_dir.empty() || !args.input_path.empty()) {
            throw std::runtime_error("DNA triage inference mode is added in Milestone 2 (src/inference)");
        }
        exit_code = run_synthetic_benchmark(args, rank, world_size, rss_baseline_kb);
    } catch (const std::bad_alloc& ex) {
        if (rank == 0) {
            std::cerr << "[OOM] rank " << rank << " out of memory: " << ex.what() << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    } catch (const std::exception& ex) {
        if (rank == 0) {
            std::cerr << "[attn error] " << ex.what() << "\n";
        }
        exit_code = 1;
    }

    MPI_Finalize();
    return exit_code;
}
