#include "tests/pch.h"
#include "opennn/core/log.h"
#include "opennn/core/profiler.h"
#include "opennn/core/string_utilities.h"
#include "opennn/model_selection/selection_utilities.h"

#include <atomic>
#include <thread>

#ifdef OPENNN_HAS_CUDA
#include "opennn/core/cuda/flash_attention_shim/c10/cuda/CUDAException.h"
#endif

using namespace opennn;

namespace
{
bool log_at_exit = false;
struct ExitLoggingProbe
{
    ~ExitLoggingProbe()
    {
        if (log_at_exit) logging::warning() << "late shutdown diagnostic\n";
    }
} exit_logging_probe;
}

class LoggingTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        previous_level = logging::level();
        logging::set_level(logging::Level::Info);
    }

    void TearDown() override
    {
        logging::set_sink({});
        logging::set_level(previous_level);
    }

    logging::Level previous_level;
};

TEST_F(LoggingTest, FiltersLevelsAndPreservesMessageText)
{
    std::string text;
    logging::set_sink([&](logging::Level, std::string_view message) { text += message; });
    logging::debug() << "hidden";
    logging::info() << "epoch " << 3 << std::endl;
    EXPECT_EQ(text, "epoch 3\n");
    logging::set_level(logging::Level::Silent);
    logging::error() << "hidden too";
    EXPECT_EQ(text, "epoch 3\n");
}

TEST_F(LoggingTest, ContainsSinkExceptionsAndContinuesLogging)
{
    logging::set_sink([](logging::Level, std::string_view) { throw std::runtime_error("sink failed"); });
    EXPECT_NO_THROW(logging::info() << "failed delivery");
    EXPECT_NO_THROW(logging::write(logging::Level::Warning, "failed direct delivery"));

    int delivered = 0;
    logging::set_sink([&](logging::Level, std::string_view) { ++delivered; });
    logging::info() << "next delivery";
    EXPECT_EQ(delivered, 1);
}

TEST_F(LoggingTest, CanReplaceSinkFromCallback)
{
    int first = 0, second = 0;
    logging::set_sink([&](logging::Level, std::string_view)
    {
        ++first;
        logging::set_sink([&](logging::Level, std::string_view) { ++second; });
    });
    logging::info() << "first";
    logging::info() << "second";
    EXPECT_EQ(first, 1);
    EXPECT_EQ(second, 1);
}

TEST_F(LoggingTest, DiscardsRecursiveMessagesWithoutDisablingLaterWrites)
{
    int delivered = 0;
    logging::set_sink([&](logging::Level, std::string_view)
    {
        ++delivered;
        logging::warning() << "recursive";
    });
    logging::info() << "first";
    logging::info() << "second";
    EXPECT_EQ(delivered, 2);
}

TEST_F(LoggingTest, KeepsStateInTheRegisteredCallback)
{
    int delivered = 0;
    logging::set_sink([count = 0, &delivered](logging::Level, std::string_view) mutable
    {
        delivered = ++count;
    });
    logging::info() << "first";
    logging::info() << "second";
    EXPECT_EQ(delivered, 2);
}

TEST_F(LoggingTest, DestroysReplacedSinkOutsideTheConfigurationLock)
{
    bool destroyed = false;
    auto token = std::shared_ptr<int>(new int(0), [&](int* value)
    {
        delete value;
        destroyed = true;
        logging::set_sink({});
    });
    logging::set_sink([token](logging::Level, std::string_view) {});
    token.reset();
    logging::set_sink({});
    EXPECT_TRUE(destroyed);
}

TEST_F(LoggingTest, AllowsConcurrentWritesAndSinkReplacement)
{
    std::atomic<int> delivered{0};
    auto sink = [&](logging::Level, std::string_view) { delivered.fetch_add(1); };
    logging::set_sink(sink);
    std::vector<std::thread> writers;
    for (int i = 0; i < 4; ++i)
        writers.emplace_back([]
        {
            for (int j = 0; j < 200; ++j) logging::info() << j;
        });
    for (int i = 0; i < 200; ++i) logging::set_sink(sink);
    for (auto& writer : writers) writer.join();
    EXPECT_EQ(delivered.load(), 800);
}

TEST_F(LoggingTest, ProgressIsOneRedirectedMessageAndCanBeSilenced)
{
    std::vector<std::string> messages;
    logging::set_sink([&](logging::Level, std::string_view text) { messages.emplace_back(text); });
    display_progress_bar(1, 2);
    ASSERT_EQ(messages.size(), 1);
    EXPECT_EQ(messages[0], "\r[" + std::string(25, '=') + ">" + std::string(24, ' ') + "] 50 %   ");
    logging::set_level(logging::Level::Silent);
    display_progress_bar(2, 2);
    EXPECT_EQ(messages.size(), 1);
}

TEST_F(LoggingTest, SelectionMessagesRespectDisplayAndLogLevel)
{
    std::string text;
    logging::set_sink([&](logging::Level, std::string_view message) { text += message; });
    EXPECT_EQ(first_stopping_condition<int>(false, {{true, 1, "stopped\n"}}), 1);
    EXPECT_TRUE(text.empty());
    EXPECT_EQ(first_stopping_condition<int>(true, {{true, 1, "stopped\n"}}), 1);
    EXPECT_EQ(text, "stopped\n");
    logging::set_level(logging::Level::Silent);
    EXPECT_EQ(first_stopping_condition<int>(true, {{true, 1, "hidden"}}), 1);
    EXPECT_EQ(text, "stopped\n");
}

TEST_F(LoggingTest, ProfilerPreservesItsStreamFormatAndRespectsLevels)
{
    profiler::Stats stats;
    std::ostringstream expected;
    stats.print(expected, "test report");
    std::string actual;
    logging::set_sink([&](logging::Level, std::string_view text) { actual += text; });
    stats.log("test report");
    EXPECT_EQ(actual, expected.str());
    logging::set_level(logging::Level::Warning);
    stats.log("hidden report");
    EXPECT_EQ(actual, expected.str());
}

TEST_F(LoggingTest, ExitDiagnosticsRemainSafeAfterSinkCleanup)
{
    // Test normal process shutdown, not inherited state after a multithreaded fork.
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT({
        log_at_exit = true;
        logging::set_sink([](logging::Level, std::string_view) {});
        std::exit(EXIT_SUCCESS);
    }, ::testing::ExitedWithCode(EXIT_SUCCESS), "late shutdown diagnostic");
}

TEST_F(LoggingTest, SilentAlsoSuppressesExitDiagnostics)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT({
        log_at_exit = true;
        logging::set_level(logging::Level::Silent);
        std::exit(EXIT_SUCCESS);
    }, ::testing::ExitedWithCode(EXIT_SUCCESS), "^$");
}

#ifdef OPENNN_HAS_CUDA
TEST_F(LoggingTest, FlashAttentionLaunchCheckStillAbortsOnError)
{
    EXPECT_NO_THROW(C10_CUDA_CHECK(cudaSuccess));
    EXPECT_DEATH(C10_CUDA_CHECK(cudaErrorInvalidValue), "CUDA error:");
}
#endif
