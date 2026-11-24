#ifndef RING_ATTN_COMMON_MEMORY_UTILS_HPP
#define RING_ATTN_COMMON_MEMORY_UTILS_HPP

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sys/resource.h>

namespace ring_attn {

inline long read_proc_status_kb(const char* field_prefix) noexcept {
    std::FILE* fp = std::fopen("/proc/self/status", "r");
    if (!fp) {
        return -1;
    }
    char line[256];
    const std::size_t prefix_len = std::strlen(field_prefix);
    long value_kb = -1;
    while (std::fgets(line, sizeof(line), fp) != nullptr) {
        if (std::strncmp(line, field_prefix, prefix_len) == 0) {
            long parsed = -1;
            if (std::sscanf(line + prefix_len, "%ld", &parsed) == 1) {
                value_kb = parsed;
            }
            break;
        }
    }
    std::fclose(fp);
    return value_kb;
}

inline long read_vm_rss_kb() noexcept {
    return read_proc_status_kb("VmRSS:");
}

inline long read_vm_size_kb() noexcept {
    return read_proc_status_kb("VmSize:");
}

inline long read_vm_hwm_kb() noexcept {
    const long hwm = read_proc_status_kb("VmHWM:");
    if (hwm >= 0) {
        return hwm;
    }
    struct rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        return ru.ru_maxrss;
    }
    return -1;
}

} // namespace ring_attn

#endif // RING_ATTN_COMMON_MEMORY_UTILS_HPP
