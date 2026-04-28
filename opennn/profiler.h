#pragma once

// Lightweight profiler — wall-clock + cudaDeviceSynchronize so GPU work is
// actually fenced before reading the timer. Only intended for ad-hoc benchmark
// runs; remove or guard with #ifdef before production.

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#ifdef OPENNN_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace opennn::profiler {

struct Stats
{
    std::map<std::string, double> times_ms;
    std::map<std::string, long>   counts;

    void add(const std::string& key, double ms)
    {
        times_ms[key] += ms;
        counts[key]   += 1;
    }

    void clear()
    {
        times_ms.clear();
        counts.clear();
    }

    void print(std::ostream& os, const std::string& title, double total_ms = 0.0) const
    {
        std::vector<std::pair<std::string, double>> sorted(times_ms.begin(), times_ms.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        os << "\n[PROFILE] " << title << "\n";
        os << "  " << std::left << std::setw(48) << "section"
           << std::right << std::setw(12) << "total_ms"
           << std::setw(10)  << "calls"
           << std::setw(12) << "ms/call";
        if (total_ms > 0.0) os << std::setw(8) << "%";
        os << "\n";

        for (const auto& [k, v] : sorted)
        {
            const long n = counts.at(k);
            os << "  " << std::left << std::setw(48) << k
               << std::right << std::setw(12) << std::fixed << std::setprecision(2) << v
               << std::setw(10) << n
               << std::setw(12) << std::fixed << std::setprecision(3) << (v / double(n));
            if (total_ms > 0.0)
                os << std::setw(7) << std::fixed << std::setprecision(1) << (v / total_ms * 100.0) << "%";
            os << "\n";
        }
        os << "\n";
    }
};

inline Stats& global_stats()
{
    static Stats s;
    return s;
}

inline bool& enabled()
{
    static bool e = false;
    return e;
}

class ScopedTimer
{
    std::string key_;
    std::chrono::steady_clock::time_point t0_;
    bool sync_gpu_;
public:
    ScopedTimer(std::string key, bool sync_gpu = true)
        : key_(std::move(key)), sync_gpu_(sync_gpu)
    {
        if (!enabled()) return;
#ifdef OPENNN_WITH_CUDA
        if (sync_gpu_) cudaDeviceSynchronize();
#endif
        t0_ = std::chrono::steady_clock::now();
    }

    ~ScopedTimer()
    {
        if (!enabled()) return;
#ifdef OPENNN_WITH_CUDA
        if (sync_gpu_) cudaDeviceSynchronize();
#endif
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0_).count();
        global_stats().add(key_, ms);
    }
};

}  // namespace opennn::profiler

// Concatenation tricks so PROFILE_SCOPE can appear multiple times in one scope.
#define OPENNN_PROFILE_CAT_INNER(a, b) a##b
#define OPENNN_PROFILE_CAT(a, b)       OPENNN_PROFILE_CAT_INNER(a, b)

// PROFILE_SCOPE — fences GPU before/after, accurate but adds overhead.
// PROFILE_SCOPE_HOST — host-only timing, no cudaDeviceSynchronize.
#define PROFILE_SCOPE(name) \
    ::opennn::profiler::ScopedTimer OPENNN_PROFILE_CAT(_profile_, __LINE__)(name, true)
#define PROFILE_SCOPE_HOST(name) \
    ::opennn::profiler::ScopedTimer OPENNN_PROFILE_CAT(_profile_, __LINE__)(name, false)
