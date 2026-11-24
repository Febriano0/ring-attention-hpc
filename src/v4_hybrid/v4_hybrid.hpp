#ifndef RING_ATTN_V4_HYBRID_HPP
#define RING_ATTN_V4_HYBRID_HPP

#include <mpi.h>

namespace ring_attn {

struct HybridRingConfig {
    int c = 256;            // Local Q/K/V rows per MPI rank (total s = P * c)
    int d = 64;             // Head dimension
    int sub_block_q = 64;   // Intra-rank OpenMP query sub-block rows
    int bq = 64;            // Internal query tile rows
    int bk = 64;            // Internal key tile rows
    int num_threads = 0;    // OpenMP threads per rank (0 = runtime default)
    bool causal = false;    // Optional causal mask across global sequence
};

/// Computes single-head self-attention across an MPI ring of ranks, with intra-rank
/// OpenMP parallelism over local query sub-blocks.
/// Q_local, K_local, V_local, and O_local have row-major shape [cfg.c x cfg.d].
/// Safe under MPI_THREAD_FUNNELED: all MPI calls execute strictly on the master thread
/// outside OpenMP parallel regions.
void attention_v4_hybrid(
    const float* Q_local,
    const float* K_local,
    const float* V_local,
    float* O_local,
    const HybridRingConfig& cfg,
    MPI_Comm comm = MPI_COMM_WORLD);

} // namespace ring_attn

#endif // RING_ATTN_V4_HYBRID_HPP
