#ifndef RING_ATTN_V3_RING_MPI_HPP
#define RING_ATTN_V3_RING_MPI_HPP

#include <cstddef>
#include <mpi.h>

namespace ring_attn {

enum class RingCommMode {
    NonBlocking, // Default: MPI_Irecv / MPI_Isend overlapped with compute + MPI_Waitall (T_nb)
    Blocking,    // Synchronous MPI_Sendrecv after compute (T_blk)
    NoComm       // Compute-only baseline without KV ring transfers (T_comp)
};

struct RingConfig {
    int c = 256;                               // Local Q/K/V rows per MPI rank (total s = P * c)
    int d = 64;                                // Head dimension
    int bq = 64;                               // Internal query tile rows
    int bk = 64;                               // Internal key tile rows
    bool causal = false;                       // Optional causal mask across global sequence
    RingCommMode comm_mode = RingCommMode::NonBlocking;
};

struct RingStats {
    double total_s = 0.0;
    double compute_s = 0.0;
    double wait_s = 0.0;
    std::size_t activation_bytes = 0;
};

/// Computes single-head self-attention across an MPI ring of P ranks.
/// Each rank holds local Q_local, K_local, V_local, O_local of shape [cfg.c x cfg.d]
/// and rotates combined contiguous [K | V] double buffers of size 2 * cfg.c * cfg.d.
RingStats attention_v3_ring_mpi(
    const float* Q_local,
    const float* K_local,
    const float* V_local,
    float* O_local,
    const RingConfig& cfg,
    MPI_Comm comm = MPI_COMM_WORLD);

} // namespace ring_attn

#endif // RING_ATTN_V3_RING_MPI_HPP
