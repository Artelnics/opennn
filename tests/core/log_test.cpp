#include "tests/pch.h"
#include "opennn/core/log.h"

#include <atomic>
#include <thread>

using namespace opennn;

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
