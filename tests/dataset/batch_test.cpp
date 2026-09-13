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

namespace
{

std::array<float, 4> read_cuda_dataset_batch(TabularDataset& dataset)
{
    const EffectiveConfig config{Device::CUDA, Type::FP32, 0};
    Batch staged(2, &dataset, config, true);
    Batch destination(2, &dataset, config);
    staged.fill({0, 1}, dataset.get_feature_selection(), FillMode::Inference);
    EXPECT_TRUE(staged.device_gather.has_value());
    const DeviceStream stream = device::get_compute_stream();
    staged.upload_to_device_batch_async(destination, stream);
    std::array<float, 4> values{};
    device::copy_async(values.data(), destination.input.buffer.data(), 2 * sizeof(float),
                       device::CopyKind::DeviceToHost, stream);
    device::copy_async(values.data() + 2, destination.target.buffer.data(), 2 * sizeof(float),
                       device::CopyKind::DeviceToHost, stream);
    device::synchronize(stream);
    return values;
}

}

TEST(BatchTest, DatasetMutationsRefreshCudaBatches)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "CUDA device unavailable.";
    TabularDataset dataset(2, {1}, {1});
    MatrixR first(2, 2);
    first << 1.0f, 10.0f, 2.0f, 20.0f;
    dataset.set_data(first);
    dataset.enable_device_residency();
    ASSERT_TRUE(dataset.is_device_resident());

    const MatrixR replacement = 100.0f * first;
    dataset.set_data(replacement);
    EXPECT_FALSE(dataset.is_device_resident());
    EXPECT_EQ(dataset.get_device_data_columns(), 0);
    dataset.enable_device_residency();
    EXPECT_EQ(read_cuda_dataset_batch(dataset),
              (std::array<float, 4>{100.0f, 200.0f, 1000.0f, 2000.0f}));

    dataset.set_data(MatrixR(first));
    EXPECT_FALSE(dataset.is_device_resident());
    dataset.enable_device_residency();
    EXPECT_EQ(read_cuda_dataset_batch(dataset),
              (std::array<float, 4>{1.0f, 2.0f, 10.0f, 20.0f}));

    dataset.set_data_constant(7.0f);
    EXPECT_FALSE(dataset.is_device_resident());
    dataset.enable_device_residency();
    EXPECT_EQ(read_cuda_dataset_batch(dataset),
              (std::array<float, 4>{7.0f, 7.0f, 7.0f, 7.0f}));
    dataset.set_storage_mode(Dataset::StorageMode::BinaryFile);
    EXPECT_FALSE(dataset.is_device_resident());
    EXPECT_EQ(dataset.get_device_data_columns(), 0);
}

TEST(BatchTest, TrainingScalingRefreshesCudaBatchesWithoutLosingInputTransform)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "CUDA device unavailable.";
    TabularDataset dataset(2, {1}, {1});
    MatrixR raw(2, 2);
    raw << 1.0f, 10.0f, 2.0f, 20.0f;
    dataset.set_data(raw);
    dataset.set_sample_roles(SampleRole::Training);
    dataset.set_variable_scalers("MinimumMaximum");
    dataset.enable_device_residency();
    dataset.prepare_training_scaling(VariableRole::Input, FeatureScaling{}, 1);
    EXPECT_FALSE(dataset.is_device_resident());
    dataset.prepare_training_scaling(VariableRole::Target, FeatureScaling{}, 1);
    dataset.enable_device_residency();
    EXPECT_EQ(read_cuda_dataset_batch(dataset),
              (std::array<float, 4>{-1.0f, 1.0f, -1.0f, 1.0f}));

    FeatureScaling target_scaling;
    target_scaling.min_range = 0.0f;
    dataset.prepare_training_scaling(VariableRole::Target, target_scaling, 1);
    EXPECT_FALSE(dataset.is_device_resident());
    dataset.enable_device_residency();
    EXPECT_EQ(read_cuda_dataset_batch(dataset),
              (std::array<float, 4>{-1.0f, 1.0f, 0.0f, 1.0f}));

    dataset.scale_features("Input");
    EXPECT_FALSE(dataset.is_device_resident());
    dataset.enable_device_residency();
    EXPECT_EQ(read_cuda_dataset_batch(dataset),
              (std::array<float, 4>{-1.0f, 1.0f, 10.0f, 20.0f}));
}

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
