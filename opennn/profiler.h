#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#ifdef OPENNN_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace opennn {

struct Stats
{
    struct Entry { double total_ms = 0.0; long calls = 0; };

    map<string, Entry> entries;

    void add(const string& key, double ms)
    {
        auto& e = entries[key];
        e.total_ms += ms;
        e.calls    += 1;
    }

    void clear()
    {
        entries.clear();
    }

    void print(ostream& os, const string& title, double total_ms = 0.0) const
    {
        vector<pair<string, Entry>> sorted(entries.begin(), entries.end());
        sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.second.total_ms > b.second.total_ms; });

        os << "\n[PROFILE] " << title << "\n";
        os << "  " << left << setw(48) << "section"
           << right << setw(12) << "total_ms"
           << setw(10)  << "calls"
           << setw(12) << "ms/call";
        if (total_ms > 0.0) os << setw(8) << "%";
        os << "\n";

        for (const auto& [key, entry] : sorted)
        {
            os << "  " << left << setw(48) << key
               << right << setw(12) << fixed << setprecision(2) << entry.total_ms
               << setw(10) << entry.calls
               << setw(12) << fixed << setprecision(3) << (entry.total_ms / double(entry.calls));
            if (total_ms > 0.0)
                os << setw(7) << fixed << setprecision(1) << (entry.total_ms / total_ms * 100.0) << "%";
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
    string key_;
    chrono::steady_clock::time_point t0_;
    bool sync_gpu_;
public:
    ScopedTimer(string key, bool sync_gpu = true)
        : key_(move(key)), sync_gpu_(sync_gpu)
    {
        if (!enabled()) return;
#ifdef OPENNN_HAS_CUDA
        if (sync_gpu_) cudaDeviceSynchronize();
#endif
        t0_ = chrono::steady_clock::now();
    }

    ~ScopedTimer()
    {
        if (!enabled()) return;
#ifdef OPENNN_HAS_CUDA
        if (sync_gpu_) cudaDeviceSynchronize();
#endif
        const auto end_time = chrono::steady_clock::now();
        const double elapsed_ms = chrono::duration<double, milli>(end_time - t0_).count();
        global_stats().add(key_, elapsed_ms);
    }
};

}  // namespace opennn

#define OPENNN_PROFILE_CAT_INNER(a, b) a##b
#define OPENNN_PROFILE_CAT(a, b)       OPENNN_PROFILE_CAT_INNER(a, b)

// The ternary short-circuits: when profiling is disabled, the `name` expression
// is not evaluated and no string is built. The empty-string fallback is cheap.
#define PROFILE_SCOPE_IMPL(name, sync) \
    ::opennn::ScopedTimer OPENNN_PROFILE_CAT(_profile_, __LINE__)( \
        ::opennn::enabled() ? string(name) : string{}, sync)

#define PROFILE_SCOPE(name)      PROFILE_SCOPE_IMPL(name, true)
#define PROFILE_SCOPE_HOST(name) PROFILE_SCOPE_IMPL(name, false)
