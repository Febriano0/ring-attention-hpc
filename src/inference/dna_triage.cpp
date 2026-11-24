#include "inference/dna_triage.hpp"
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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ring_attn {
namespace {

constexpr int kVocabSize = 4098;
constexpr int kPadTokenId = 4097;
constexpr double kTokensPerRead = 25.0; // 150 bp / 6-mer

std::vector<float> read_exact_floats(const std::string& path, std::size_t expected_count) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open weight file: " + path);
    }
    const std::streamsize size_bytes = in.tellg();
    const std::size_t expected_bytes = expected_count * sizeof(float);
    if (size_bytes < 0 || static_cast<std::size_t>(size_bytes) != expected_bytes) {
        throw std::runtime_error("Unexpected byte size in weight file: " + path);
    }
    in.seekg(0, std::ios::beg);
    std::vector<float> data(expected_count);
    if (!in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(expected_bytes))) {
        throw std::runtime_error("Failed reading weight file: " + path);
    }
    return data;
}

void embed_tokens(
    const std::vector<int32_t>& tokens,
    const std::vector<float>& embed_table,
    int d,
    float* out_x) {
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        int32_t tid = tokens[i];
        if (tid < 0 || tid >= kVocabSize) {
            tid = 4096; // <UNK>
        }
        const float* src = embed_table.data() + static_cast<std::size_t>(tid) * d;
        std::copy_n(src, d, out_x + i * static_cast<std::size_t>(d));
    }
}

void write_triage_outputs(const TriageConfig& cfg, const TriageResult& res, int world_size) {
    if (!cfg.json_out.empty()) {
        std::ofstream jout(cfg.json_out);
        if (!jout.is_open()) {
            throw std::runtime_error("Cannot open --json-out: " + cfg.json_out);
        }
        jout << "{\n"
             << "  \"version\": \"" << cfg.version << "\",\n"
             << "  \"s\": " << cfg.s << ",\n"
             << "  \"d\": " << cfg.d << ",\n"
             << "  \"c\": " << cfg.c << ",\n"
             << "  \"P\": " << world_size << ",\n"
             << "  \"predicted_class\": " << res.predicted_class << ",\n"
             << "  \"predicted_label\": \"" << res.predicted_label << "\",\n"
             << std::setprecision(8)
             << "  \"prob_normal\": " << res.prob_normal << ",\n"
             << "  \"prob_progression\": " << res.prob_progression << ",\n"
             << "  \"logit0\": " << res.logit0 << ",\n"
             << "  \"logit1\": " << res.logit1 << ",\n"
             << "  \"time_s\": " << res.time_s << ",\n"
             << "  \"throughput_reads_per_s\": " << res.throughput_reads_per_s << ",\n"
             << "  \"peak_rss_kb\": " << res.peak_rss_kb << ",\n"
             << "  \"delta_rss_kb\": " << res.delta_rss_kb << ",\n"
             << "  \"alloc_kb\": " << res.alloc_kb << "\n"
             << "}\n";
    }
    if (!cfg.csv_path.empty()) {
        std::ifstream check_in(cfg.csv_path);
        const bool needs_header = !check_in.good() || check_in.peek() == std::ifstream::traits_type::eof();
        check_in.close();
        std::ofstream cout_f(cfg.csv_path, std::ios::app);
        if (!cout_f.is_open()) {
            throw std::runtime_error("Cannot open --csv: " + cfg.csv_path);
        }
        if (needs_header) {
            cout_f << "version,input,s,d,c,P,predicted_class,predicted_label,prob_normal,prob_progression,"
                      "time_s,throughput_reads_per_s,peak_rss_kb,delta_rss_kb,alloc_kb\n";
        }
        cout_f << cfg.version << "," << cfg.input_path << "," << cfg.s << "," << cfg.d << ","
               << cfg.c << "," << world_size << "," << res.predicted_class << ",\""
               << res.predicted_label << "\"," << std::fixed << std::setprecision(6)
               << res.prob_normal << "," << res.prob_progression << ","
               << std::scientific << res.time_s << ","
               << std::fixed << std::setprecision(2) << res.throughput_reads_per_s << ","
               << res.peak_rss_kb << "," << res.delta_rss_kb << "," << res.alloc_kb << "\n";
    }
}

} // namespace

TriageWeights load_triage_weights(const std::string& weights_dir, int expected_d) {
    if (expected_d <= 0) {
        throw std::invalid_argument("expected_d must be positive");
    }
    TriageWeights w{};
    w.vocab_size = kVocabSize;
    w.d = expected_d;
    w.num_classes = 2;

    const std::size_t dd = static_cast<std::size_t>(expected_d) * static_cast<std::size_t>(expected_d);
    w.embed = read_exact_floats(weights_dir + "/embed.bin", static_cast<std::size_t>(kVocabSize) * expected_d);
    w.wq = read_exact_floats(weights_dir + "/wq.bin", dd);
    w.wk = read_exact_floats(weights_dir + "/wk.bin", dd);
    w.wv = read_exact_floats(weights_dir + "/wv.bin", dd);
    w.wo = read_exact_floats(weights_dir + "/wo.bin", dd);
    w.fc_w = read_exact_floats(weights_dir + "/fc_w.bin", static_cast<std::size_t>(2) * expected_d);
    w.fc_b = read_exact_floats(weights_dir + "/fc_b.bin", 2);
    return w;
}

std::vector<int32_t> load_token_slice(
    const std::string& input_path,
    std::size_t offset,
    std::size_t count) {
    std::ifstream in(input_path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open token binary file: " + input_path);
    }
    const std::streamsize file_bytes = in.tellg();
    const std::size_t total_tokens = (file_bytes > 0)
        ? static_cast<std::size_t>(file_bytes) / sizeof(int32_t)
        : 0ULL;

    std::vector<int32_t> tokens(count, kPadTokenId);
    if (offset < total_tokens) {
        const std::size_t avail = std::min(count, total_tokens - offset);
        in.seekg(static_cast<std::streamoff>(offset * sizeof(int32_t)), std::ios::beg);
        in.read(reinterpret_cast<char*>(tokens.data()), static_cast<std::streamsize>(avail * sizeof(int32_t)));
    }
    return tokens;
}

TriageResult run_dna_triage(
    const TriageConfig& cfg,
    long rss_baseline_kb,
    MPI_Comm comm) {
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    const bool is_dist = (cfg.version == "v3" || cfg.version == "v4");
    const int local_rows = is_dist ? cfg.c : cfg.s;
    const std::size_t token_offset = is_dist
        ? static_cast<std::size_t>(rank) * static_cast<std::size_t>(cfg.c)
        : 0ULL;

    const TriageWeights w = load_triage_weights(cfg.weights_dir, cfg.d);
    const std::vector<int32_t> tokens = load_token_slice(
        cfg.input_path, token_offset, static_cast<std::size_t>(local_rows));

    const std::size_t mat_elems = static_cast<std::size_t>(local_rows) * static_cast<std::size_t>(cfg.d);
    std::vector<float> X(mat_elems), Q(mat_elems), K(mat_elems), V(mat_elems), H(mat_elems, 0.0f), Z(mat_elems);

    if (is_dist) MPI_Barrier(comm);
    const double t0 = MPI_Wtime();

    // 1. Embedding lookup
    embed_tokens(tokens, w.embed, cfg.d, X.data());

    // 2. Q, K, V projections: Y = X * W^T
    gemm_nt_fp32(local_rows, cfg.d, cfg.d, 1.0f, X.data(), cfg.d, w.wq.data(), cfg.d, 0.0f, Q.data(), cfg.d);
    gemm_nt_fp32(local_rows, cfg.d, cfg.d, 1.0f, X.data(), cfg.d, w.wk.data(), cfg.d, 0.0f, K.data(), cfg.d);
    gemm_nt_fp32(local_rows, cfg.d, cfg.d, 1.0f, X.data(), cfg.d, w.wv.data(), cfg.d, 0.0f, V.data(), cfg.d);

    // 3. Attention forward pass
    if (cfg.version == "v0") {
        attention_v0_fp32(Q.data(), K.data(), V.data(), H.data(), cfg.s, cfg.d, false);
    } else if (cfg.version == "v1") {
        AttentionConfig acfg{cfg.s, cfg.d, cfg.c, cfg.bq, cfg.bk, 1, false};
        attention_v1_blockwise(Q.data(), K.data(), V.data(), H.data(), acfg);
    } else if (cfg.version == "v2") {
        AttentionConfig acfg{cfg.s, cfg.d, cfg.c, cfg.bq, cfg.bk, cfg.threads, false};
        attention_v2_openmp(Q.data(), K.data(), V.data(), H.data(), acfg);
    } else if (cfg.version == "v3") {
        RingConfig rcfg{cfg.c, cfg.d, cfg.bq, cfg.bk, false, cfg.comm_mode};
        attention_v3_ring_mpi(Q.data(), K.data(), V.data(), H.data(), rcfg, comm);
    } else if (cfg.version == "v4") {
        HybridRingConfig hcfg{cfg.c, cfg.d, cfg.sub_block_q, cfg.bq, cfg.bk, cfg.threads, false};
        attention_v4_hybrid(Q.data(), K.data(), V.data(), H.data(), hcfg, comm);
    } else {
        throw std::invalid_argument("Unsupported triage version: " + cfg.version);
    }

    // 4. Output projection: Z = H * W_o^T
    gemm_nt_fp32(local_rows, cfg.d, cfg.d, 1.0f, H.data(), cfg.d, w.wo.data(), cfg.d, 0.0f, Z.data(), cfg.d);

    // 5. Exact global mean pooling across sequence length s
    std::vector<double> local_sum(static_cast<std::size_t>(cfg.d), 0.0);
    for (int r = 0; r < local_rows; ++r) {
        const float* z_row = Z.data() + static_cast<std::size_t>(r) * cfg.d;
        for (int col = 0; col < cfg.d; ++col) {
            local_sum[static_cast<std::size_t>(col)] += static_cast<double>(z_row[col]);
        }
    }
    std::vector<double> global_sum(static_cast<std::size_t>(cfg.d), 0.0);
    if (is_dist) {
        MPI_Allreduce(local_sum.data(), global_sum.data(), cfg.d, MPI_DOUBLE, MPI_SUM, comm);
    } else {
        global_sum = local_sum;
    }

    const double inv_s = 1.0 / static_cast<double>(cfg.s);
    double logits[2] = {static_cast<double>(w.fc_b[0]), static_cast<double>(w.fc_b[1])};
    for (int col = 0; col < cfg.d; ++col) {
        const double p = global_sum[static_cast<std::size_t>(col)] * inv_s;
        logits[0] += p * static_cast<double>(w.fc_w[col]);
        logits[1] += p * static_cast<double>(w.fc_w[static_cast<std::size_t>(cfg.d) + col]);
    }

    const double max_l = std::max(logits[0], logits[1]);
    const double e0 = std::exp(logits[0] - max_l);
    const double e1 = std::exp(logits[1] - max_l);
    const double local_dt = MPI_Wtime() - t0;

    double elapsed_s = local_dt;
    if (is_dist) {
        MPI_Reduce(&local_dt, &elapsed_s, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    }

    const long local_peak = std::max(read_vm_hwm_kb(), read_vm_rss_kb());
    const long local_delta = (rss_baseline_kb >= 0 && local_peak >= 0)
        ? std::max(0L, local_peak - rss_baseline_kb)
        : -1L;
    long peak_rss_kb = local_peak;
    long delta_rss_kb = local_delta;
    if (is_dist) {
        MPI_Reduce(&local_peak, &peak_rss_kb, 1, MPI_LONG, MPI_MAX, 0, comm);
        MPI_Reduce(&local_delta, &delta_rss_kb, 1, MPI_LONG, MPI_MAX, 0, comm);
    }

    TriageResult res{};
    res.logit0 = logits[0];
    res.logit1 = logits[1];
    res.prob_normal = e0 / (e0 + e1);
    res.prob_progression = e1 / (e0 + e1);
    res.predicted_class = (res.prob_progression >= res.prob_normal) ? 1 : 0;
    res.predicted_label = (res.predicted_class == 0) ? "Normal-like" : "Progresi Karsinogenesis";
    res.time_s = elapsed_s;
    res.throughput_reads_per_s = (static_cast<double>(cfg.s) / kTokensPerRead) / std::max(elapsed_s, 1e-12);
    res.peak_rss_kb = peak_rss_kb;
    res.delta_rss_kb = delta_rss_kb;

    const double s_d = static_cast<double>(cfg.s);
    const double c_d = static_cast<double>(cfg.c);
    const double d_d = static_cast<double>(cfg.d);
    if (cfg.version == "v0") {
        res.alloc_kb = ((6.0 * s_d * d_d + s_d * s_d) * 4.0) / 1024.0;
    } else if (is_dist) {
        res.alloc_kb = ((10.0 * c_d * d_d + 2.0 * c_d + static_cast<double>(cfg.bq) * cfg.bk) * 4.0) / 1024.0;
    } else {
        res.alloc_kb = ((6.0 * s_d * d_d + 2.0 * c_d + static_cast<double>(cfg.bq) * cfg.bk) * 4.0) / 1024.0;
    }

    if (rank == 0) {
        std::cout << "[triage] version=" << cfg.version << " s=" << cfg.s << " d=" << cfg.d
                  << " P=" << world_size << " | pred=" << res.predicted_label
                  << " (p_norm=" << std::fixed << std::setprecision(4) << res.prob_normal
                  << ", p_prog=" << res.prob_progression << ")"
                  << " | time=" << std::setprecision(5) << res.time_s << "s"
                  << " throughput=" << std::setprecision(1) << res.throughput_reads_per_s << " reads/s"
                  << " peak_rss=" << res.peak_rss_kb << "KB delta_rss=" << res.delta_rss_kb << "KB\n";
        write_triage_outputs(cfg, res, world_size);
    }
    return res;
}

} // namespace ring_attn
