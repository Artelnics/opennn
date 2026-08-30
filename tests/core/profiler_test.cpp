#include "tests/pch.h"

#include <sstream>
#include <thread>

#include "opennn/core/profiler.h"

using namespace opennn;

TEST(ProfilerTest, ConcurrentAddsPreserveEverySample)
{
    profiler::Stats stats;
    constexpr int threads_number = 4;
    constexpr int samples_per_thread = 250;

    vector<thread> threads;
    threads.reserve(threads_number);
    for (int i = 0; i < threads_number; ++i)
    {
        threads.emplace_back([&stats]
        {
            for (int sample = 0; sample < samples_per_thread; ++sample)
                stats.add("parallel", 1.0);
        });
    }
    for (thread& worker : threads) worker.join();

    EXPECT_DOUBLE_EQ(stats.total_ms(), double(threads_number * samples_per_thread));
    EXPECT_EQ(stats.call_count("parallel"), threads_number * samples_per_thread);
    EXPECT_EQ(stats.call_count("missing"), 0);

    ostringstream output;
    stats.print(output,
                "test",
                stats.total_ms(),
                "GRAPH_TIMING");

    const string report = output.str();
    EXPECT_NE(report.find("[GRAPH_TIMING] test"), string::npos);
    const size_t entry_start = report.find("parallel");
    ASSERT_NE(entry_start, string::npos);
    const size_t entry_end = report.find('\n', entry_start);
    const string entry = report.substr(entry_start, entry_end - entry_start);

    double total_ms = 0.0;
    long calls = 0;
    istringstream values(entry.substr(48));
    values >> total_ms >> calls;

    EXPECT_DOUBLE_EQ(total_ms, double(threads_number * samples_per_thread));
    EXPECT_EQ(calls, threads_number * samples_per_thread);
}

TEST(ProfilerTest, TimerRetainsConstructionTimeState)
{
    profiler::stats().clear();
    profiler::set_enabled(true);

    {
        profiler::ScopedTimer timer("active", false);
        profiler::set_enabled(false);
    }
    {
        profiler::ScopedTimer timer("inactive", false);
    }

    ostringstream output;
    profiler::stats().print(output, "test");
    const string report = output.str();

    EXPECT_NE(report.find("active"), string::npos);
    EXPECT_EQ(report.find("inactive"), string::npos);

    profiler::stats().clear();
    profiler::set_enabled(false);
}
