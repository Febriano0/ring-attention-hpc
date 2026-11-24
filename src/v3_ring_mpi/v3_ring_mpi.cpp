#include "v3_ring_mpi/v3_ring_mpi.hpp"
#include "common/block_attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ring_attn {
namespace {

void validate_ring_config(
    const RingConfig& cfg,
    const float* Q_local,
    const float* K_local,
    const float* V_local,
    const float* O_local) {
    if (cfg.c <= 0 || cfg.d <= 0 || cfg.bq <= 0 || cfg.bk <= 0) {
        throw std::invalid_argument("RingConfig dimensions c, d, bq, bk must be positive");
    }
    if (Q_local == nullptr || K_local == nullptr || V_local == nullptr || O_local == nullptr) {
        throw std::invalid_argument("attention_v3_ring_mpi received null local tensor pointer");
    }
    const std::size_t block_elems = static_cast<std::size_t>(cfg.c) * static_cast<std::size_t>(cfg.d);
    if (block_elems > static_cast<std::size_t>(std::numeric_limits<int>::max() / 2)) {
        throw std::overflow_error("Local [K|V] buffer element count exceeds MPI int limit");
    }
}

} // namespace

RingStats attention_v3_ring_mpi(
    const float* Q_local,
    const float* K_local,
    const float* V_local,
    float* O_local,
    const RingConfig& cfg,
    MPI_Comm comm) {
    validate_ring_config(cfg, Q_local, K_local, V_local, O_local);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    const int c = cfg.c;
    const int d = cfg.d;
    const int eff_bq = std::min(cfg.bq, c);
    const int eff_bk = std::min(cfg.bk, c);
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    const int prev_rank = (rank - 1 + world_size) % world_size;
    const int next_rank = (rank + 1) % world_size;
    const std::size_t block_elems = static_cast<std::size_t>(c) * static_cast<std::size_t>(d);
    const int kv_count = static_cast<int>(2 * block_elems);

    // Six activation blocks per rank: Q_local, O_local, kv_cur (K|V), kv_next (K|V).
    std::vector<float> kv_cur(2 * block_elems);
    std::vector<float> kv_next(2 * block_elems);
    std::copy_n(K_local, block_elems, kv_cur.data());
    std::copy_n(V_local, block_elems, kv_cur.data() + block_elems);

    std::vector<float> m_state(static_cast<std::size_t>(c));
    std::vector<float> l_state(static_cast<std::size_t>(c));
    std::vector<float> tile_buf(static_cast<std::size_t>(eff_bq) * static_cast<std::size_t>(eff_bk));

    RingStats stats{};
    stats.activation_bytes =
        (6ULL * block_elems + 2ULL * static_cast<std::size_t>(c) +
         static_cast<std::size_t>(eff_bq) * static_cast<std::size_t>(eff_bk)) * sizeof(float);

    const double t_total_start = MPI_Wtime();
    init_block_state(O_local, m_state.data(), l_state.data(), c, d);

    const int q_global_base = rank * c;

    for (int step = 0; step < world_size; ++step) {
        MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        const bool has_comm = (step < world_size - 1) && (cfg.comm_mode != RingCommMode::NoComm);

        if (has_comm && cfg.comm_mode == RingCommMode::NonBlocking) {
            const int tag = 200 + step;
            MPI_Irecv(kv_next.data(), kv_count, MPI_FLOAT, prev_rank, tag, comm, &reqs[0]);
            MPI_Isend(kv_cur.data(), kv_count, MPI_FLOAT, next_rank, tag, comm, &reqs[1]);
        }

        const int kv_owner = (rank - step + world_size) % world_size;
        const int k_global_base = kv_owner * c;
        const bool skip_block = cfg.causal && (k_global_base > q_global_base + c - 1);

        if (!skip_block) {
            const float* K_cur = kv_cur.data();
            const float* V_cur = kv_cur.data() + block_elems;
            const double t_comp_start = MPI_Wtime();
            attend_block_online_softmax(
                Q_local, c, q_global_base,
                K_cur, V_cur, c, k_global_base,
                d, cfg.bq, cfg.bk, scale, cfg.causal,
                O_local, m_state.data(), l_state.data(), tile_buf.data());
            stats.compute_s += (MPI_Wtime() - t_comp_start);
        }

        if (has_comm) {
            const double t_wait_start = MPI_Wtime();
            if (cfg.comm_mode == RingCommMode::NonBlocking) {
                MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
            } else {
                const int tag = 200 + step;
                MPI_Sendrecv(kv_cur.data(), kv_count, MPI_FLOAT, next_rank, tag,
                             kv_next.data(), kv_count, MPI_FLOAT, prev_rank, tag,
                             comm, MPI_STATUS_IGNORE);
            }
            stats.wait_s += (MPI_Wtime() - t_wait_start);
            kv_cur.swap(kv_next);
        }
    }

    finalize_block_output(O_local, l_state.data(), c, d);
    stats.total_s = MPI_Wtime() - t_total_start;
    return stats;
}

} // namespace ring_attn
