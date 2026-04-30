#pragma once

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

        for (const auto& [key, total] : sorted)
        {
            const long call_count = counts.at(key);
            os << "  " << std::left << std::setw(48) << key
               << std::right << std::setw(12) << std::fixed << std::setprecision(2) << total
               << std::setw(10) << call_count
               << std::setw(12) << std::fixed << std::setprecision(3) << (total / double(call_count));
            if (total_ms > 0.0)
                os << std::setw(7) << std::fixed << std::setprecision(1) << (total / total_ms * 100.0) << "%";
            os << "\n";
        }
        os << "\n";
    }
};

inline Stats& global_stats()
{
    static Stats stats;
    return stats;
}

inline bool& enabled()
{
    static bool is_enabled = false;
    return is_enabled;
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
        const auto end_time = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - t0_).count();
        global_stats().add(key_, elapsed_ms);
    }
};

}  // namespace opennn::profiler

#define OPENNN_PROFILE_CAT_INNER(a, b) a##b
#define OPENNN_PROFILE_CAT(a, b)       OPENNN_PROFILE_CAT_INNER(a, b)

#define PROFILE_SCOPE(name) \
    ::opennn::profiler::ScopedTimer OPENNN_PROFILE_CAT(_profile_, __LINE__)(name, true)
#define PROFILE_SCOPE_HOST(name) \
    ::opennn::profiler::ScopedTimer OPENNN_PROFILE_CAT(_profile_, __LINE__)(name, false)
