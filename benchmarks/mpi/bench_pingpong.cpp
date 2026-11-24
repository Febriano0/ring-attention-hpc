#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    int reps = 50;
    int warmup = 10;
    std::string csv_path;
};

struct TimingStats {
    double median_s = 0.0;
    double stddev_s = 0.0;
};

Args parse_args(int argc, char** argv) {
    Args args;
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
            throw std::invalid_argument("Unknown flag: " + flag);
        }
    }
    if (args.reps <= 0 || args.warmup < 0) {
        throw std::invalid_argument("reps must be > 0 and warmup >= 0");
    }
    return args;
}

TimingStats summarize(std::vector<double> samples) {
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
            const double d = v - mean;
            sq += d * d;
        }
        st.stddev_s = std::sqrt(sq / static_cast<double>(n - 1));
    }
    return st;
}

void write_csv_row(
    const std::string& path,
    std::size_t bytes,
    int reps,
    const TimingStats& st,
    double bw_bytes_per_s) {
    if (path.empty()) return;
    std::ifstream check_in(path);
    const bool needs_header = !check_in.good() || check_in.peek() == std::ifstream::traits_type::eof();
    check_in.close();
    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open CSV path: " + path);
    }
    if (needs_header) {
        out << "bytes,reps,oneway_median_s,oneway_stddev_s,bandwidth_bytes_per_s,bandwidth_gb_s\n";
    }
    out << bytes << "," << reps << ","
        << std::scientific << std::setprecision(6) << st.median_s << "," << st.stddev_s << ","
        << bw_bytes_per_s << "," << std::fixed << std::setprecision(4)
        << (bw_bytes_per_s / 1e9) << "\n";
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 2) {
        if (rank == 0) {
            std::cerr << "bench_pingpong requires exactly 2 MPI ranks (mpirun -np 2)\n";
        }
        MPI_Finalize();
        return 1;
    }

    try {
        const Args args = parse_args(argc, argv);
        // Covers latency (64 B) and [K|V] payload sizes 2*c*d*4 (d=64) for c=128..4096 + 4 MiB
        const std::size_t sizes[] = {
            64,
            65536,    // c=128
            131072,   // c=256
            262144,   // c=512
            524288,   // c=1024
            1048576,  // c=2048
            2097152,  // c=4096
            4194304   // 4 MiB
        };

        if (rank == 0) {
            std::cout << "=== bench_pingpong (reps=" << args.reps << ", warmup=" << args.warmup << ") ===\n";
        }

        for (std::size_t nbytes : sizes) {
            const int count = static_cast<int>(nbytes);
            std::vector<char> buf(nbytes, 1);

            for (int w = 0; w < args.warmup; ++w) {
                if (rank == 0) {
                    MPI_Send(buf.data(), count, MPI_BYTE, 1, 77, MPI_COMM_WORLD);
                    MPI_Recv(buf.data(), count, MPI_BYTE, 1, 78, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                    MPI_Recv(buf.data(), count, MPI_BYTE, 0, 77, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(buf.data(), count, MPI_BYTE, 0, 78, MPI_COMM_WORLD);
                }
            }

            std::vector<double> oneway_samples(static_cast<std::size_t>(args.reps));
            MPI_Barrier(MPI_COMM_WORLD);
            for (int r = 0; r < args.reps; ++r) {
                const double t0 = MPI_Wtime();
                if (rank == 0) {
                    MPI_Send(buf.data(), count, MPI_BYTE, 1, 77, MPI_COMM_WORLD);
                    MPI_Recv(buf.data(), count, MPI_BYTE, 1, 78, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                    MPI_Recv(buf.data(), count, MPI_BYTE, 0, 77, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(buf.data(), count, MPI_BYTE, 0, 78, MPI_COMM_WORLD);
                }
                const double rtt = MPI_Wtime() - t0;
                oneway_samples[static_cast<std::size_t>(r)] = 0.5 * rtt;
            }

            if (rank == 0) {
                const TimingStats st = summarize(oneway_samples);
                const double bw = static_cast<double>(nbytes) / std::max(st.median_s, 1e-12);
                std::cout << "bytes=" << std::setw(8) << nbytes
                          << " | oneway=" << std::fixed << std::setprecision(3) << std::setw(9)
                          << (st.median_s * 1e6) << " us"
                          << " | B=" << std::setprecision(3) << std::setw(6) << (bw / 1e9) << " GB/s\n";
                write_csv_row(args.csv_path, nbytes, args.reps, st, bw);
            }
        }
    } catch (const std::exception& ex) {
        if (rank == 0) {
            std::cerr << "bench_pingpong error: " << ex.what() << "\n";
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
