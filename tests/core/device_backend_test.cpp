#include "tests/pch.h"

#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/tensor_types.h"

#include <filesystem>
#include <format>
#include <chrono>
#include <thread>

using namespace opennn;

#ifdef OPENNN_HAS_CUDA
namespace
{
// Constructed before main initializes the backend, so destruction exercises
// GPU storage that outlives the backend and block cache.
Buffer static_cuda_buffer;

void CUDART_CB delay_compute(void*)
{
    // Keep work pending on the nonblocking stream so a default-stream copy
    // cannot accidentally pass merely because the GPU finished first.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}
}

TEST(DeviceBackendDeathTest, StaticCudaBufferExitsCleanly)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT({
        static_cuda_buffer.resize_bytes(256, Device::CUDA);
        std::exit(EXIT_SUCCESS);
    }, ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

TEST(DeviceBackendTest, BufferMigrationWaitsForPendingCompute)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";
    Buffer values(Device::CUDA);
    values.resize_bytes(256, Device::CUDA);
    values.setZero();
    device::synchronize();
    const auto stream = device::get_compute_stream();
    ASSERT_EQ(cudaLaunchHostFunc(stream, delay_compute, nullptr), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(values.data(), 0x5a, 256, stream), cudaSuccess);
    values.migrate_to(Device::CPU);
    device::synchronize(stream);
    EXPECT_TRUE(std::all_of(values.as<unsigned char>(), values.as<unsigned char>() + 256,
                            [](unsigned char value) { return value == 0x5a; }));
}

TEST(DeviceBackendTest, DefaultClearFollowsPendingCompute)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";
    Buffer values(Device::CUDA);
    values.resize_bytes(256, Device::CUDA);
    const auto stream = device::get_compute_stream();
    ASSERT_EQ(cudaLaunchHostFunc(stream, delay_compute, nullptr), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(values.data(), 0x5a, 256, stream), cudaSuccess);
    values.setZero();
    values.migrate_to(Device::CPU, stream);
    EXPECT_TRUE(std::all_of(values.as<unsigned char>(), values.as<unsigned char>() + 256,
                            [](unsigned char value) { return value == 0; }));
}

TEST(DeviceBackendTest, DefaultUploadFollowsPendingCompute)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";
    Buffer values(Device::CUDA);
    values.resize_bytes(256, Device::CUDA);
    const auto stream = device::get_compute_stream();
    ASSERT_EQ(cudaLaunchHostFunc(stream, delay_compute, nullptr), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(values.data(), 0, 256, stream), cudaSuccess);
    const std::vector<unsigned char> source(256, 0x5a);
    device::copy_async(values.data(), source.data(), 256, device::CopyKind::HostToDevice);
    values.migrate_to(Device::CPU, stream);
    EXPECT_TRUE(std::all_of(values.as<unsigned char>(), values.as<unsigned char>() + 256,
                            [](unsigned char value) { return value == 0x5a; }));
}
#endif

TEST(DeviceBackendTest, IsCudaBuildMatchesBuild)
{
#ifdef OPENNN_HAS_CUDA
    EXPECT_TRUE(device::is_cuda_build());
#else
    EXPECT_FALSE(device::is_cuda_build());
#endif
}

TEST(DeviceBackendTest, HasCudaDeviceMatchesBuild)
{
#ifdef OPENNN_HAS_CUDA
    EXPECT_EQ(device::has_cuda_device(), device::cuda_compute_capability() > 0);
#else
    EXPECT_FALSE(device::has_cuda_device());
#endif
}

TEST(DeviceBackendTest, ComputeCapabilityMatchesBuild)
{
    if (device::has_cuda_device())
        EXPECT_GT(device::cuda_compute_capability(), 0);
    else
        EXPECT_EQ(device::cuda_compute_capability(), -1);
}

TEST(DeviceBackendTest, AvailableMemoryMatchesBuild)
{
    if (device::has_cuda_device())
        EXPECT_GT(device::available_memory(), 0u);
    else
        EXPECT_THROW(device::available_memory(), runtime_error);
}

TEST(DeviceBackendTest, AllocationGrowthFlagRoundTrips)
{
    const bool previous = device::cuda_allocation_growth_forbidden();

    device::set_cuda_allocation_growth_forbidden(true);
    EXPECT_TRUE(device::cuda_allocation_growth_forbidden());

    device::set_cuda_allocation_growth_forbidden(false);
    EXPECT_FALSE(device::cuda_allocation_growth_forbidden());

    device::set_cuda_allocation_growth_forbidden(previous);
}

TEST(DeviceBackendTest, GrowthGuardMatchesBuild)
{
    const bool previous = device::cuda_allocation_growth_forbidden();

    device::set_cuda_allocation_growth_forbidden(false);

    {
        device::CudaAllocationGrowthGuard guard(true);
        EXPECT_EQ(device::cuda_allocation_growth_forbidden(), device::is_cuda_build());
    }

    EXPECT_FALSE(device::cuda_allocation_growth_forbidden());

    device::set_cuda_allocation_growth_forbidden(previous);
}

TEST(DeviceBackendTest, AllocateZeroBytesReturnsNull)
{
    EXPECT_EQ(device::allocate(Device::CPU, 0), nullptr);
}

TEST(DeviceBackendTest, AllocateNegativeBytesThrows)
{
    EXPECT_THROW(device::allocate(Device::CPU, -1), runtime_error);
}

TEST(DeviceBackendTest, BoundedCacheRetainsHitsAndLimitsEntries)
{
    unordered_map<int, string> entries;

    detail::bounded_cache_entry(entries, 1, 2) = "one";
    detail::bounded_cache_entry(entries, 2, 2) = "two";

    EXPECT_EQ(detail::bounded_cache_entry(entries, 1, 2), "one");
    EXPECT_EQ(entries.size(), 2);

    detail::bounded_cache_entry(entries, 3, 2) = "three";

    EXPECT_EQ(entries.size(), 2);
    EXPECT_TRUE(entries.contains(3));
}

TEST(DeviceBackendTest, AllocateAutoDeviceThrows)
{
    EXPECT_THROW(device::allocate(Device::Auto, 16), runtime_error);
}

TEST(DeviceBackendTest, AllocateAndDeallocateHostMemory)
{
    const Index byte_count = 64;

    void* pointer = device::allocate(Device::CPU, byte_count);
    ASSERT_NE(pointer, nullptr);

    device::set_zero(pointer, byte_count, Device::CPU);

    const unsigned char* bytes = static_cast<const unsigned char*>(pointer);
    for (Index i = 0; i < byte_count; i++)
        EXPECT_EQ(bytes[i], 0);

    device::deallocate(Device::CPU, pointer, byte_count);
}

TEST(DeviceBackendTest, DeallocateNullIsSafe)
{
    EXPECT_NO_THROW(device::deallocate(Device::CPU, nullptr, 0));
}

TEST(DeviceBackendTest, SetZeroNegativeThrows)
{
    int value = 7;
    EXPECT_THROW(device::set_zero(&value, -1, Device::CPU), runtime_error);
}

TEST(DeviceBackendTest, SetZeroAutoDeviceThrows)
{
    int value = 7;
    EXPECT_THROW(device::set_zero(&value, sizeof(value), Device::Auto), runtime_error);
}

TEST(DeviceBackendTest, SetZeroAsyncClearsHostBuffer)
{
    const Index byte_count = 32;

#ifdef OPENNN_HAS_CUDA
    if (!device::has_cuda_device())
        GTEST_SKIP() << "no CUDA device available";

    void* device_buffer = device::allocate(Device::CUDA, byte_count);
    ASSERT_NE(device_buffer, nullptr);

    vector<unsigned char> host(static_cast<size_t>(byte_count), 0xFF);
    device::copy_async(device_buffer, host.data(), byte_count,
                       device::CopyKind::HostToDevice, nullptr);
    device::set_zero_async(device_buffer, byte_count, nullptr);
    device::copy_async(host.data(), device_buffer, byte_count,
                       device::CopyKind::DeviceToHost, nullptr);
    device::synchronize(nullptr);

    for (Index i = 0; i < byte_count; i++)
        EXPECT_EQ(host[static_cast<size_t>(i)], 0);

    device::deallocate(Device::CUDA, device_buffer, byte_count);
#else
    void* pointer = device::allocate(Device::CPU, byte_count);
    ASSERT_NE(pointer, nullptr);

    memset(pointer, 0xFF, static_cast<size_t>(byte_count));
    device::set_zero_async(pointer, byte_count, nullptr);

    const unsigned char* bytes = static_cast<const unsigned char*>(pointer);
    for (Index i = 0; i < byte_count; i++)
        EXPECT_EQ(bytes[i], 0);

    device::deallocate(Device::CPU, pointer, byte_count);
#endif
}

TEST(DeviceBackendTest, CopyHostToHostCopiesBytes)
{
    const Index count = 5;
    const Index byte_count = count * static_cast<Index>(sizeof(float));

    vector<float> source = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
    vector<float> destination(static_cast<size_t>(count), 0.0f);

    device::copy_async(destination.data(), source.data(), byte_count,
                       device::CopyKind::HostToHost, nullptr);

    for (Index i = 0; i < count; i++)
        EXPECT_FLOAT_EQ(destination[static_cast<size_t>(i)], source[static_cast<size_t>(i)]);
}

TEST(DeviceBackendTest, CopyCpuToCpuCopiesBytes)
{
    const Index count = 4;
    const Index byte_count = count * static_cast<Index>(sizeof(int));

    vector<int> source = { 10, 20, 30, 40 };
    vector<int> destination(static_cast<size_t>(count), 0);

    device::copy_async(destination.data(), source.data(), byte_count,
                       Device::CPU, Device::CPU, nullptr);

    for (Index i = 0; i < count; i++)
        EXPECT_EQ(destination[static_cast<size_t>(i)], source[static_cast<size_t>(i)]);
}

TEST(DeviceBackendTest, CopyNegativeBytesThrows)
{
    int source = 1;
    int destination = 0;
    EXPECT_THROW(device::copy_async(&destination, &source, -1,
                                    device::CopyKind::HostToHost, nullptr),
                 runtime_error);
}

TEST(DeviceBackendTest, CopyDeviceKindMatchesBuild)
{
#ifdef OPENNN_HAS_CUDA
    if (!device::has_cuda_device())
        GTEST_SKIP() << "no CUDA device available";

    const Index count = 6;
    const Index byte_count = count * static_cast<Index>(sizeof(float));

    const vector<float> source = { 1.5f, -2.0f, 3.25f, 0.0f, 42.0f, -7.5f };
    vector<float> destination(static_cast<size_t>(count), 0.0f);

    void* device_buffer = device::allocate(Device::CUDA, byte_count);
    ASSERT_NE(device_buffer, nullptr);

    device::copy_async(device_buffer, source.data(), byte_count,
                       device::CopyKind::HostToDevice, nullptr);
    device::copy_async(destination.data(), device_buffer, byte_count,
                       device::CopyKind::DeviceToHost, nullptr);
    device::synchronize(nullptr);

    for (Index i = 0; i < count; i++)
        EXPECT_FLOAT_EQ(destination[static_cast<size_t>(i)], source[static_cast<size_t>(i)]);

    device::deallocate(Device::CUDA, device_buffer, byte_count);
#else
    int source = 1;
    int destination = 0;
    EXPECT_THROW(device::copy_async(&destination, &source, sizeof(int),
                                    device::CopyKind::HostToDevice, nullptr),
                 runtime_error);
#endif
}

TEST(DeviceBackendTest, CopyZeroBytesIsNoOp)
{
    int source = 99;
    int destination = 7;
    EXPECT_NO_THROW(device::copy_async(&destination, &source, 0,
                                       device::CopyKind::HostToHost, nullptr));
    EXPECT_EQ(destination, 7);
}

TEST(DeviceBackendTest, SynchronizeAndCheckLastErrorAreNoOps)
{
    EXPECT_NO_THROW(device::synchronize(nullptr));
    EXPECT_NO_THROW(device::check_last_error());
}

TEST(DeviceBackendTest, CudaEventOwnsAndTransfersHandle)
{
    device::CudaEvent event;
    event.create();
    const cudaEvent_t handle = event.get();

    EXPECT_EQ(static_cast<bool>(event), device::has_cuda_device());

    device::CudaEvent moved(std::move(event));

    EXPECT_FALSE(event);
    EXPECT_EQ(moved.get(), handle);

    device::CudaEvent assigned;
    assigned = std::move(moved);

    EXPECT_FALSE(moved);
    EXPECT_EQ(assigned.get(), handle);
}

TEST(DeviceBackendTest, EventOperationsTolerateNull)
{
    EXPECT_NO_THROW(device::synchronize_event(nullptr));
    EXPECT_NO_THROW(device::stream_wait_event(nullptr, nullptr));
}

TEST(DeviceBackendTest, PinnedBufferOwnsAndTransfersAllocation)
{
    const Index byte_count = 128;
    device::PinnedBuffer buffer(byte_count);
    ASSERT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.byte_size(), byte_count);

    memset(buffer.data(), 0, static_cast<size_t>(byte_count));
    void* const pointer = buffer.data();

    device::PinnedBuffer moved(std::move(buffer));
    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(moved.data(), pointer);

    device::PinnedBuffer assigned;
    assigned = std::move(moved);
    EXPECT_TRUE(moved.empty());
    EXPECT_EQ(assigned.data(), pointer);
}

TEST(DeviceBackendTest, PinnedBufferGrowthDoesNotShrink)
{
    device::PinnedBuffer buffer;
    EXPECT_TRUE(buffer.empty());

    buffer.grow_to(128);
    void* const pointer = buffer.data();
    buffer.grow_to(64);

    EXPECT_EQ(buffer.data(), pointer);
    EXPECT_EQ(buffer.byte_size(), 128);

    buffer.resize_bytes(0);
    EXPECT_TRUE(buffer.empty());
}

TEST(DeviceBackendTest, PinnedBufferRejectsNegativeSize)
{
    device::PinnedBuffer buffer;
    EXPECT_THROW(buffer.resize_bytes(-8), runtime_error);
    EXPECT_THROW(buffer.grow_to(-8), runtime_error);
}

TEST(DeviceBackendTest, ComputeStreamMatchesBuild)
{
    if (device::has_cuda_device())
    {
        EXPECT_NE(device::get_compute_stream(), nullptr);
        EXPECT_NE(device::get_transfer_stream(), nullptr);
        EXPECT_NE(device::get_compute_stream(), device::get_transfer_stream());
    }
    else
    {
        EXPECT_EQ(device::get_compute_stream(), nullptr);
        EXPECT_EQ(device::get_transfer_stream(), nullptr);
    }
}

TEST(DeviceBackendTest, LibraryHandlesMatchBuild)
{
    if (device::has_cuda_device())
    {
        EXPECT_NE(device::get_cublas_handle(), nullptr);
        EXPECT_NE(device::get_cublas_lt_handle(), nullptr);
        EXPECT_NE(device::get_cudnn_handle(), nullptr);
        EXPECT_NE(device::get_op_tensor_add_descriptor(), nullptr);
    }
    else
    {
        EXPECT_EQ(device::get_cublas_handle(), nullptr);
        EXPECT_EQ(device::get_cublas_lt_handle(), nullptr);
        EXPECT_EQ(device::get_cudnn_handle(), nullptr);
        EXPECT_EQ(device::get_op_tensor_add_descriptor(), nullptr);
    }
}

#ifdef OPENNN_HAS_CUDA
TEST(DeviceBackendTest, CublasPointerModeGuardRestoresMode)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "CUDA device unavailable.";

    const cublasHandle_t handle = device::get_cublas_handle();
    cublasPointerMode_t original_mode = CUBLAS_POINTER_MODE_HOST;
    ASSERT_EQ(cublasGetPointerMode(handle, &original_mode), CUBLAS_STATUS_SUCCESS);

    const cublasPointerMode_t temporary_mode =
        original_mode == CUBLAS_POINTER_MODE_HOST
            ? CUBLAS_POINTER_MODE_DEVICE
            : CUBLAS_POINTER_MODE_HOST;

    {
        const device::CublasPointerModeGuard guard(handle, temporary_mode);
        cublasPointerMode_t current_mode = original_mode;
        ASSERT_EQ(cublasGetPointerMode(handle, &current_mode), CUBLAS_STATUS_SUCCESS);
        EXPECT_EQ(current_mode, temporary_mode);
    }

    cublasPointerMode_t restored_mode = temporary_mode;
    ASSERT_EQ(cublasGetPointerMode(handle, &restored_mode), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(restored_mode, original_mode);
}

TEST(DeviceBackendTest, CublasMathModeGuardRestoresMode)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "CUDA device unavailable.";

    const cublasHandle_t handle = device::get_cublas_handle();
    cublasMath_t original_mode = CUBLAS_DEFAULT_MATH;
    ASSERT_EQ(cublasGetMathMode(handle, &original_mode), CUBLAS_STATUS_SUCCESS);

    const cublasMath_t temporary_mode =
        original_mode == CUBLAS_DEFAULT_MATH
            ? CUBLAS_TF32_TENSOR_OP_MATH
            : CUBLAS_DEFAULT_MATH;

    {
        const device::CublasMathModeGuard guard(handle, temporary_mode);
        cublasMath_t current_mode = original_mode;
        ASSERT_EQ(cublasGetMathMode(handle, &current_mode), CUBLAS_STATUS_SUCCESS);
        EXPECT_EQ(current_mode, temporary_mode);
    }

    cublasMath_t restored_mode = temporary_mode;
    ASSERT_EQ(cublasGetMathMode(handle, &restored_mode), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(restored_mode, original_mode);
}
#endif

TEST(DeviceBackendTest, GetDeviceProvidesThreadPoolDevice)
{
    ThreadPoolDevice& thread_pool_device = get_device();
    EXPECT_GT(thread_pool_device.numThreads(), 0);
}

TEST(DeviceBackendTest, GetDeviceReturnsStableThreadPoolDevice)
{
    ThreadPoolDevice& first = get_device();
    ThreadPoolDevice& second = get_device();

    EXPECT_EQ(&first, &second);
}

#ifdef OPENNN_HAS_CUDA
// The cuBLASLt tuner times its candidates and the timings overlap, so two
// processes could pick different kernels -- and different bf16 rounding -- for
// the same GEMM. The plan cache is what makes the second process take the
// first one's verdict. Whether it does is a cross-process property, checked by
// the Qwen benchmark's within_engine_deterministic gate; what a unit test can
// hold is that a tuned shape leaves a plan on disk under a directory that
// names the card and the library, so a driver or cuBLASLt update cannot serve
// a stale kernel.
TEST(DeviceBackendTest, LtPlanCacheNamesCardAndLibraryAndStoresTunedShape)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "CUDA device unavailable.";

    const string directory = device::lt_plan_cache_directory();
    if (directory.empty()) GTEST_SKIP() << "cuBLASLt plan cache disabled or without a directory.";

    const string name = std::filesystem::path(directory).filename().string();
    EXPECT_NE(name.find(format("-sm{}-", device::cuda_compute_capability())), string::npos) << name;
    EXPECT_NE(name.find("-cublaslt"), string::npos) << name;

    // Qwen3-4B's gate projection at decode: eight heuristic candidates on every
    // card measured, so the tuner has to choose and its choice is persisted.
    const int m = 9728, n = 1, k = 2560;
    const Index bf16 = Index(sizeof(uint16_t));
    void* a = device::allocate(Device::CUDA, Index(m) * k * bf16);
    void* b = device::allocate(Device::CUDA, Index(k) * n * bf16);
    void* d = device::allocate(Device::CUDA, Index(m) * n * bf16);
    device::set_zero(a, Index(m) * k * bf16, Device::CUDA);
    device::set_zero(b, Index(k) * n * bf16, Device::CUDA);

    run_lt_matmul_cached(m, n, k, CUBLAS_OP_N, CUBLAS_OP_N, CUBLASLT_EPILOGUE_DEFAULT,
                         a, b, d, nullptr, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF);
    device::synchronize(device::get_compute_stream());

    device::deallocate(Device::CUDA, d, Index(m) * n * bf16);
    device::deallocate(Device::CUDA, b, Index(k) * n * bf16);
    device::deallocate(Device::CUDA, a, Index(m) * k * bf16);

    size_t plans = 0;
    std::error_code failed;
    for (const auto& entry : std::filesystem::directory_iterator(directory, failed))
        if (entry.path().extension() == ".ltplan") ++plans;

    EXPECT_FALSE(failed) << directory;
    EXPECT_GE(plans, 1u) << directory;
}
#endif
