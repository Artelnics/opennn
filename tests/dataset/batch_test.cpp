#include "tests/pch.h"

#include "opennn/dataset/batch.h"
#include "opennn/core/configuration.h"
#include "opennn/dataset/tabular_dataset.h"

using namespace opennn;

TEST(BatchTest, FreshCpuBatchUsesConfiguredDevice)
{
    TabularDataset dataset(2, {3}, {1});
    const EffectiveConfig config{Device::CPU, Type::FP32, 0};

    Batch batch(2, &dataset, config);

    EXPECT_FALSE(batch.uses_cuda());
    EXPECT_EQ(batch.input.buffer.get_device(), Device::CPU);
    EXPECT_EQ(batch.input.buffer.byte_size(), 2 * 3 * Index(sizeof(float)));
}

TEST(BatchTest, ValidationQueueIsDerivedFromPoolOwnership)
{
    BatchPools pools;

    EXPECT_EQ(&pools.validation_queue(), &pools.training_empty_queue);

    pools.validation_pool.emplace_back();
    EXPECT_EQ(&pools.validation_queue(), &pools.validation_empty_queue);
}

TEST(BatchTest, PrefetchSessionPublishesBatchesWithoutPointerSentinels)
{
    TabularDataset dataset(2, {3}, {1});
    const EffectiveConfig config{Device::CPU, Type::FP32, 0};
    Batch batch(2, &dataset, config);
    ThreadSafeQueue<Batch*> queue;
    BatchPrefetchSession session(queue, 1);

    EXPECT_TRUE(session.publish(0, &batch));

    EXPECT_EQ(session.wait(0), &batch);
}

TEST(BatchTest, PrefetchSessionWakesWaitersWithTheWorkerError)
{
    ThreadSafeQueue<Batch*> queue;
    BatchPrefetchSession session(queue, 1);

    try
    {
        throw runtime_error("prefetch failed");
    }
    catch (...)
    {
        session.capture_current_exception();
    }

    EXPECT_FALSE(session.publish(0, nullptr));
    EXPECT_THROW(session.wait(0), runtime_error);
}

#ifdef OPENNN_HAS_CUDA

TEST(BatchTest, FreshCudaBatchUsesConfiguredDevice)
{
    TabularDataset dataset(2, {3}, {1});
    const EffectiveConfig config{Device::CUDA, Type::FP32, 0};

    Batch batch(2, &dataset, config);

    EXPECT_TRUE(batch.uses_cuda());
    EXPECT_EQ(batch.input.buffer.get_device(), Device::CUDA);
    EXPECT_EQ(batch.input.buffer.byte_size(), 2 * 3 * Index(sizeof(float)));
}

TEST(BatchTest, CudaPrefetchBatchKeepsDeviceIdentityWithoutDeviceStorage)
{
    TabularDataset dataset(2, {3}, {1});
    const EffectiveConfig config{Device::CUDA, Type::FP32, 0};

    Batch batch(2, &dataset, config, true);

    EXPECT_TRUE(batch.uses_cuda());
    EXPECT_EQ(batch.input.buffer.get_device(), Device::CUDA);
    EXPECT_EQ(batch.input.buffer.byte_size(), 0);
}

#endif
