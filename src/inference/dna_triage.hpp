#ifndef RING_ATTN_INFERENCE_DNA_TRIAGE_HPP
#define RING_ATTN_INFERENCE_DNA_TRIAGE_HPP

#include "v3_ring_mpi/v3_ring_mpi.hpp"

#include <cstdint>
#include <mpi.h>
#include <string>
#include <vector>

namespace ring_attn {

struct TriageWeights {
    int vocab_size = 4098;
    int d = 64;
    int num_classes = 2;
    std::vector<float> embed; // [vocab_size x d]
    std::vector<float> wq;    // [d x d] (PyTorch Linear weight: out x in)
    std::vector<float> wk;    // [d x d]
    std::vector<float> wv;    // [d x d]
    std::vector<float> wo;    // [d x d]
    std::vector<float> fc_w;  // [2 x d]
    std::vector<float> fc_b;  // [2]
};

struct TriageConfig {
    std::string version = "v3";
    std::string weights_dir;
    std::string input_path;
    std::string json_out;
    std::string csv_path;
    int s = 1024;
    int d = 64;
    int c = 256;
    int bq = 64;
    int bk = 64;
    int sub_block_q = 64;
    int threads = 1;
    RingCommMode comm_mode = RingCommMode::NonBlocking;
};

struct TriageResult {
    int predicted_class = 0;
    std::string predicted_label;
    double prob_normal = 0.0;
    double prob_progression = 0.0;
    double logit0 = 0.0;
    double logit1 = 0.0;
    double time_s = 0.0;
    double throughput_reads_per_s = 0.0;
    long peak_rss_kb = -1;
    long delta_rss_kb = -1;
    double alloc_kb = 0.0;
};

TriageWeights load_triage_weights(const std::string& weights_dir, int expected_d);

std::vector<int32_t> load_token_slice(
    const std::string& input_path,
    std::size_t offset,
    std::size_t count);

TriageResult run_dna_triage(
    const TriageConfig& cfg,
    long rss_baseline_kb,
    MPI_Comm comm = MPI_COMM_WORLD);

} // namespace ring_attn

#endif // RING_ATTN_INFERENCE_DNA_TRIAGE_HPP
