#include "v4_hybrid/v4_hybrid.hpp"
#include "common/block_attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <vector>

namespace ring_attn {
namespace {

void validate_hybrid_config(
    const HybridRingConfig& cfg,
    const float* Q_local,
    const float* K_local,
    const float* V_local,
    const float* O_local) {
    if (cfg.c <= 0 || cfg.d <= 0 || cfg.sub_block_q <= 0 || cfg.bq <= 0 || cfg.bk <= 0) {
        throw std::invalid_argument("HybridRingConfig dimensions c, d, sub_block_q, bq, bk must be positive");
    }
    if (cfg.num_threads < 0) {
        throw std::invalid_argument("HybridRingConfig num_threads must be non-negative");
    }
    if (Q_local == nullptr || K_local == nullptr || V_local == nullptr || O_local == nullptr) {
        throw std::invalid_argument("attention_v4_hybrid received null local tensor pointer");
    }
    const std::size_t block_elems = static_cast<std::size_t>(cfg.c) * static_cast<std::size_t>(cfg.d);
    if (block_elems > static_cast<std::size_t>(std::numeric_limits<int>::max() / 2)) {
        throw std::overflow_error("Local [K|V] buffer element count exceeds MPI int limit");
    }
}

} // namespace

void attention_v4_hybrid(
    const float* Q_local,
    const float* K_local,
    const float* V_local,
    float* O_local,
    const HybridRingConfig& cfg,
    MPI_Comm comm) {
    validate_hybrid_config(cfg, Q_local, K_local, V_local, O_local);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    const int c = cfg.c;
    const int d = cfg.d;
    const int eff_sub_q = std::min(cfg.sub_block_q, c);
    const int eff_bq = std::min(cfg.bq, eff_sub_q);
    const int eff_bk = std::min(cfg.bk, c);
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    const int threads = (cfg.num_threads > 0) ? cfg.num_threads : omp_get_max_threads();

    const int prev_rank = (rank - 1 + world_size) % world_size;
    const int next_rank = (rank + 1) % world_size;
    const std::size_t block_elems = static_cast<std::size_t>(c) * static_cast<std::size_t>(d);
    const int kv_count = static_cast<int>(2 * block_elems);

    // Double-buffered combined [K | V] block to overlap ring transfer with computation.
    std::vector<float> kv_cur(2 * block_elems);
    std::vector<float> kv_next(2 * block_elems);
    std::copy_n(K_local, block_elems, kv_cur.data());
    std::copy_n(V_local, block_elems, kv_cur.data() + block_elems);

    std::vector<float> m_state(static_cast<std::size_t>(c));
    std::vector<float> l_state(static_cast<std::size_t>(c));
    init_block_state(O_local, m_state.data(), l_state.data(), c, d);

    // 64-byte padded per-thread tile scratchpad allocated once across all ring steps.
    const std::size_t tile_elems = static_cast<std::size_t>(eff_bq) * static_cast<std::size_t>(eff_bk);
    const std::size_t tile_stride = ((tile_elems + 15) / 16) * 16;
    std::vector<float> all_tile_bufs(static_cast<std::size_t>(threads) * tile_stride);

    const int q_global_base = rank * c;
    const int num_sub_blocks = (c + eff_sub_q - 1) / eff_sub_q;

    for (int step = 0; step < world_size; ++step) {
        MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        if (step < world_size - 1) {
            const int tag = 100 + step;
            MPI_Irecv(kv_next.data(), kv_count, MPI_FLOAT, prev_rank, tag, comm, &reqs[0]);
            MPI_Isend(kv_cur.data(), kv_count, MPI_FLOAT, next_rank, tag, comm, &reqs[1]);
        }

        const int kv_owner = (rank - step + world_size) % world_size;
        const int k_global_base = kv_owner * c;
        const bool skip_block = cfg.causal && (k_global_base > q_global_base + c - 1);

        if (!skip_block) {
            const float* K_cur = kv_cur.data();
            const float* V_cur = kv_cur.data() + block_elems;

#pragma omp parallel num_threads(threads)
            {
                float* my_tile = all_tile_bufs.data() +
                                 static_cast<std::size_t>(omp_get_thread_num()) * tile_stride;
                auto run_sub_block = [&](int sb) {
                    const int q_sub = sb * eff_sub_q;
                    const int q_rows = std::min(eff_sub_q, c - q_sub);
                    attend_block_online_softmax(
                        Q_local + static_cast<std::size_t>(q_sub) * d, q_rows, q_global_base + q_sub,
                        K_cur, V_cur, c, k_global_base,
                        d, cfg.bq, cfg.bk, scale, cfg.causal,
                        O_local + static_cast<std::size_t>(q_sub) * d,
                        m_state.data() + q_sub,
                        l_state.data() + q_sub,
                        my_tile);
                };

                if (cfg.causal) {
#pragma omp for schedule(dynamic, 1)
                    for (int sb = 0; sb < num_sub_blocks; ++sb) {
                        run_sub_block(sb);
                    }
                } else {
#pragma omp for schedule(static)
                    for (int sb = 0; sb < num_sub_blocks; ++sb) {
                        run_sub_block(sb);
                    }
                }
            }
        }

        if (step < world_size - 1) {
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
            kv_cur.swap(kv_next);
        }
    }

    finalize_block_output(O_local, l_state.data(), c, d);
}

} // namespace ring_attn
