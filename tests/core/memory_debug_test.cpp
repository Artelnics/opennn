#include "tests/pch.h"

#include <sstream>
#include <thread>

#include "opennn/core/memory_debug.h"

using namespace opennn;

TEST(MemoryDebugTest, ConcurrentRecordsAreAggregated)
{
    memory_debug::reset();

    constexpr int threads_number = 4;
    constexpr int records_per_thread = 100;
    vector<thread> threads;
    threads.reserve(threads_number);

    for (int i = 0; i < threads_number; ++i)
    {
        threads.emplace_back([]
        {
            for (int record = 0; record < records_per_thread; ++record)
                memory_debug::record("test", "parallel", 1024, "note");
        });
    }
    for (thread& worker : threads) worker.join();

    ostringstream output;
    memory_debug::print(output);

    if (!memory_debug::enabled())
    {
        EXPECT_TRUE(output.str().empty());
        return;
    }

    EXPECT_NE(output.str().find("[MEMORY_DEBUG] rows=1"), string::npos);
    EXPECT_NE(output.str().find("test,parallel,400,0.39,note"), string::npos);

    memory_debug::reset();
}
