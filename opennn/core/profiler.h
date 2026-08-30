#pragma once

#include <atomic>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "opennn/core/device_backend.h"

namespace opennn::profiler
{

// Achievable bandwidth in GB/s, used only to express a scope as a percentage of
// peak. Set OPENNN_PEAK_GBPS from a STREAM-style triad on the machine that is
// measuring; a spec-sheet figure is never reached in practice and would make
// every kernel look worse than it is. Unset means the percentage is omitted.
inline double peak_gbps()
{
    static const double value = []
    {
        const char* const text = std::getenv("OPENNN_PEAK_GBPS");
        if (text == nullptr) return 0.0;

        const double parsed = std::atof(text);
        return parsed > 0.0 ? parsed : 0.0;
    }();

    return value;
}

class Stats
{
public:
    struct Entry { double total_ms = 0.0; long calls = 0; double total_bytes = 0.0; };

    void add(const string& key, double ms, double bytes = 0.0)
    {
        const std::lock_guard lock(entries_mutex);
        Entry& entry = entries[key];
        entry.total_ms += ms;
        entry.total_bytes += bytes;
        ++entry.calls;
    }

    void set(const string& key, double total_ms, long calls, double total_bytes = 0.0)
    {
        if (calls <= 0) return;

        const std::lock_guard lock(entries_mutex);
        entries[key] = {total_ms, calls, total_bytes};
    }

    void clear()
    {
        const std::lock_guard lock(entries_mutex);
        entries.clear();
    }

    double total_ms() const
    {
        const std::lock_guard lock(entries_mutex);

        double total = 0.0;
        for (const auto& entry : entries)
            total += entry.second.total_ms;

        return total;
    }

    long call_count(const string& key) const
    {
        const std::lock_guard lock(entries_mutex);
        const auto found = entries.find(key);
        return found == entries.end() ? 0 : found->second.calls;
    }

    void print(ostream& os,
               const string& title,
               double total_ms = 0.0,
               string_view category = "PROFILE") const
    {
        vector<pair<string, Entry>> sorted;
        {
            const std::lock_guard lock(entries_mutex);
            sorted.assign(entries.begin(), entries.end());
        }
        ranges::sort(sorted, greater<>{}, [](const auto& entry) { return entry.second.total_ms; });

        bool any_bytes = false;
        for (const auto& entry : sorted)
            if (entry.second.total_bytes > 0.0) { any_bytes = true; break; }

        const double peak = peak_gbps();

        os << "\n[" << category << "] " << title << "\n";
        os << "  " << left << setw(48) << "section"
           << right << setw(12) << "total_ms"
           << setw(10)  << "calls"
           << setw(12) << "ms/call";
        if (total_ms > 0.0) os << setw(8) << "%";
        if (any_bytes)
        {
            os << setw(14) << "GiB" << setw(10) << "GB/s";
            if (peak > 0.0) os << setw(8) << "%peak";
        }
        os << "\n";

        for (const auto& [key, entry] : sorted)
        {
            os << "  " << left << setw(48) << key
               << right << setw(12) << fixed << setprecision(2) << entry.total_ms
               << setw(10) << entry.calls
               << setw(12) << fixed << setprecision(3) << (entry.total_ms / double(entry.calls));
            if (total_ms > 0.0)
                os << setw(7) << fixed << setprecision(1) << (entry.total_ms / total_ms * 100.0) << "%";

            if (any_bytes)
            {
                const bool measured = entry.total_bytes > 0.0 && entry.total_ms > 0.0;

                // Sizes are binary GiB to match every other size OpenNN reports;
                // the rate is decimal GB/s, which is how bandwidth is quoted.
                const double gbps = measured ? entry.total_bytes / (entry.total_ms * 1.0e6) : 0.0;

                // Six decimals so a kernel moving a few hundred kilobytes still
                // reads as a number rather than 0.000, which is the case that
                // matters when chasing a small kernel that runs a great many
                // times.
                if (measured)
                    os << setw(14) << fixed << setprecision(6) << (entry.total_bytes / 1073741824.0)
                       << setw(10) << fixed << setprecision(1) << gbps;
                else
                    os << setw(14) << "-" << setw(10) << "-";

                if (peak > 0.0)
                {
                    // Nothing beats the memory system, so a rate above peak
                    // means the scope and its byte count disagree about what
                    // happened. Two ways to get here. The scope may be timing
                    // less work than it is charged for, which is what an
                    // operator fused into its predecessor looks like: the call
                    // returns almost immediately while the model still charges
                    // it a full pass. Or the byte model is simply wrong. Either
                    // way the number is not a bandwidth, so refuse to print one.
                    //
                    // A scope on a CUDA-graph path is a different problem and
                    // does not land here: it reports honestly, but only for the
                    // eager warmup calls, since capture disables the profiler
                    // and the replays never enter the scope at all. Set
                    // OPENNN_GRAPH_TIMING to keep the run eager when what you
                    // want is bandwidth rather than throughput.
                    if (measured && gbps <= peak)
                        os << setw(7) << fixed << setprecision(1) << (gbps / peak * 100.0) << "%";
                    else if (measured)
                        os << setw(8) << ">peak";
                    else
                        os << setw(8) << "-";
                }
            }
            os << "\n";
        }
        os << "\n";
    }

private:
    map<string, Entry> entries;
    mutable std::mutex entries_mutex;
};

inline Stats& stats()
{
    static Stats instance;
    return instance;
}

namespace detail
{

inline std::atomic_bool& enabled_flag()
{
    static std::atomic_bool enabled{std::getenv("OPENNN_PROFILE") != nullptr};
    return enabled;
}

struct ExitDump
{
    ExitDump() { stats(); }
    ~ExitDump();
};

inline ExitDump& exit_dump()
{
    static ExitDump dump;
    return dump;
}

}

inline bool is_enabled() noexcept
{
    return detail::enabled_flag().load(std::memory_order_relaxed);
}

inline void set_enabled(bool enabled) noexcept
{
    detail::enabled_flag().store(enabled, std::memory_order_relaxed);
}

class ScopedTimer
{
public:
    ScopedTimer(string new_key, bool synchronize_gpu = true, double moved_bytes = 0.0)
        : key(std::move(new_key)),
          bytes(moved_bytes),
          sync_gpu(synchronize_gpu),
          active(is_enabled() && !key.empty())
    {
        if (!active) return;
        detail::exit_dump();
        if (sync_gpu) device::synchronize();
        start = chrono::steady_clock::now();
    }

    // For a scope that only learns its working set partway through, such as one
    // that picks between a fused and a materialized path after it has started.
    void set_bytes(double moved_bytes) noexcept { bytes = moved_bytes; }

    ~ScopedTimer()
    {
        if (!active) return;
        if (sync_gpu) device::synchronize();
        const auto end_time = chrono::steady_clock::now();
        const double elapsed_ms = chrono::duration<double, milli>(end_time - start).count();
        stats().add(key, elapsed_ms, bytes);
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    string key;
    double bytes;
    chrono::steady_clock::time_point start;
    bool sync_gpu;
    bool active;
};

namespace detail
{

inline ExitDump::~ExitDump()
{
    if (is_enabled() && stats().total_ms() > 0.0)
        stats().print(cerr, "profile (OPENNN_PROFILE)");
}

}

}

#define OPENNN_PROFILE_CAT_INNER(a, b) a##b
#define OPENNN_PROFILE_CAT(a, b)       OPENNN_PROFILE_CAT_INNER(a, b)

#define PROFILE_SCOPE_IMPL(name, sync) \
    ::opennn::profiler::ScopedTimer OPENNN_PROFILE_CAT(_profile_, __LINE__)( \
        ::opennn::profiler::is_enabled() ? string(name) : string{}, sync)

#define PROFILE_SCOPE(name)      PROFILE_SCOPE_IMPL(name, true)
#define PROFILE_SCOPE_HOST(name) PROFILE_SCOPE_IMPL(name, false)

// Like PROFILE_SCOPE, but also records how many bytes the scope moves so the
// report can put achieved bandwidth next to the time. Pass every byte the
// memory system sees -- reads plus writes, counting a tensor once per pass over
// it, not once in total.
#define PROFILE_SCOPE_BYTES(name, bytes) \
    ::opennn::profiler::ScopedTimer OPENNN_PROFILE_CAT(_profile_, __LINE__)( \
        ::opennn::profiler::is_enabled() ? string(name) : string{}, true, double(bytes))

// A named timer, for when the byte count is only known after the scope opens.
// Call timer.set_bytes(...) once the path is chosen.
#define PROFILE_SCOPE_NAMED(timer, name) \
    ::opennn::profiler::ScopedTimer timer( \
        ::opennn::profiler::is_enabled() ? string(name) : string{}, true)
