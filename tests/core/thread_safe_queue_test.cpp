#include "tests/pch.h"

#include <future>

#include "opennn/core/thread_safe_queue.h"

using namespace opennn;

TEST(ThreadSafeQueueTest, PreservesFifoOrderAndDefaultValues)
{
    ThreadSafeQueue<int> queue;
    queue.push(0);
    queue.push(1);

    int value = -1;
    ASSERT_TRUE(queue.wait_pop(value));
    EXPECT_EQ(value, 0);
    ASSERT_TRUE(queue.wait_pop(value));
    EXPECT_EQ(value, 1);
}

TEST(ThreadSafeQueueTest, CloseUnblocksWaitingConsumer)
{
    ThreadSafeQueue<int> queue;
    std::future<bool> result = std::async(std::launch::async, [&queue]
    {
        int value = 0;
        return queue.wait_pop(value);
    });

    EXPECT_EQ(result.wait_for(std::chrono::milliseconds(10)), std::future_status::timeout);
    queue.close();
    EXPECT_FALSE(result.get());
}

TEST(ThreadSafeQueueTest, ReopenAcceptsMoreItems)
{
    ThreadSafeQueue<int> queue;
    queue.close();

    int value = 0;
    EXPECT_FALSE(queue.wait_pop(value));

    queue.reopen();
    queue.push(7);
    ASSERT_TRUE(queue.wait_pop(value));
    EXPECT_EQ(value, 7);
}
