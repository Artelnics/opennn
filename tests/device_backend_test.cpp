#include "pch.h"

#include "opennn/configuration.h"
#include "opennn/device_backend.h"

using namespace opennn;


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

    std::vector<unsigned char> host(static_cast<size_t>(byte_count), 0xFF);
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

    std::vector<float> source = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
    std::vector<float> destination(static_cast<size_t>(count), 0.0f);

    device::copy_async(destination.data(), source.data(), byte_count,
                       device::CopyKind::HostToHost, nullptr);

    for (Index i = 0; i < count; i++)
        EXPECT_FLOAT_EQ(destination[static_cast<size_t>(i)], source[static_cast<size_t>(i)]);
}


TEST(DeviceBackendTest, CopyCpuToCpuCopiesBytes)
{
    const Index count = 4;
    const Index byte_count = count * static_cast<Index>(sizeof(int));

    std::vector<int> source = { 10, 20, 30, 40 };
    std::vector<int> destination(static_cast<size_t>(count), 0);

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

    const std::vector<float> source = { 1.5f, -2.0f, 3.25f, 0.0f, 42.0f, -7.5f };
    std::vector<float> destination(static_cast<size_t>(count), 0.0f);

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


TEST(DeviceBackendTest, CreateStreamMatchesBuild)
{
    cudaStream_t stream = device::create_stream(0);

    if (device::has_cuda_device())
        EXPECT_NE(stream, nullptr);
    else
        EXPECT_EQ(stream, nullptr);

    EXPECT_NO_THROW(device::destroy_stream(stream));
}


TEST(DeviceBackendTest, CreateEventMatchesBuild)
{
    cudaEvent_t event = device::create_event();

    if (device::has_cuda_device())
        EXPECT_NE(event, nullptr);
    else
        EXPECT_EQ(event, nullptr);

    EXPECT_NO_THROW(device::destroy_event(event));
}


TEST(DeviceBackendTest, EventOperationsTolerateNull)
{
    EXPECT_NO_THROW(device::synchronize_event(nullptr));
    EXPECT_NO_THROW(device::stream_wait_event(nullptr, nullptr));
}


TEST(DeviceBackendTest, PinnedHostAllocationRoundTrips)
{
    const Index byte_count = 128;

    void* pointer = device::allocate_pinned_host(byte_count);
    ASSERT_NE(pointer, nullptr);

    memset(pointer, 0, static_cast<size_t>(byte_count));

    EXPECT_NO_THROW(device::deallocate_pinned_host(pointer));
}


TEST(DeviceBackendTest, PinnedHostZeroBytesReturnsNull)
{
    EXPECT_EQ(device::allocate_pinned_host(0), nullptr);
}


TEST(DeviceBackendTest, PinnedHostNegativeBytesThrows)
{
    EXPECT_THROW(device::allocate_pinned_host(-8), runtime_error);
}


TEST(DeviceBackendTest, ComputeStreamMatchesBuild)
{
    EXPECT_EQ(device::get_compute_stream(), Backend::get_compute_stream());

    if (device::has_cuda_device())
        EXPECT_NE(Backend::get_compute_stream(), nullptr);
    else
        EXPECT_EQ(Backend::get_compute_stream(), nullptr);
}


TEST(DeviceBackendTest, BackendProvidesThreadPoolDevice)
{
    ThreadPoolDevice* thread_pool_device = Backend::instance().get_thread_pool_device();
    EXPECT_NE(thread_pool_device, nullptr);
    EXPECT_GT(thread_pool_device->numThreads(), 0);
}


TEST(DeviceBackendTest, GetDeviceReturnsBackendThreadPoolDevice)
{
    ThreadPoolDevice& reference = get_device();
    EXPECT_EQ(&reference, Backend::instance().get_thread_pool_device());
}
