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

class Stats
{
public:
    struct Entry { double total_ms = 0.0; long calls = 0; };

    void add(const string& key, double ms)
    {
        const std::lock_guard lock(entries_mutex);
        Entry& entry = entries[key];
        entry.total_ms += ms;
        ++entry.calls;
    }

    void set(const string& key, double total_ms, long calls)
    {
        if (calls <= 0) return;

        const std::lock_guard lock(entries_mutex);
        entries[key] = {total_ms, calls};
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

        os << "\n[" << category << "] " << title << "\n";
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
    ScopedTimer(string new_key, bool synchronize_gpu = true)
        : key(std::move(new_key)),
          sync_gpu(synchronize_gpu),
          active(is_enabled() && !key.empty())
    {
        if (!active) return;
        detail::exit_dump();
        if (sync_gpu) device::synchronize();
        start = chrono::steady_clock::now();
    }

    ~ScopedTimer()
    {
        if (!active) return;
        if (sync_gpu) device::synchronize();
        const auto end_time = chrono::steady_clock::now();
        const double elapsed_ms = chrono::duration<double, milli>(end_time - start).count();
        stats().add(key, elapsed_ms);
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    string key;
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
