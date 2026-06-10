//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E V I C E   B A C K E N D
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "device_backend.h"
#include "string_utilities.h"

#include <atomic>

namespace opennn::device
{

namespace
{

std::atomic_bool cuda_allocation_growth_forbidden_runtime{false};

void throw_if_auto(Device device_type)
{
    throw_if(device_type == Device::Auto,
             "device backend expects a resolved device.");
}

CopyKind copy_kind(Device source, Device target)
{
    throw_if_auto(source);
    throw_if_auto(target);

    if (source == Device::CUDA && target == Device::CUDA) return CopyKind::DeviceToDevice;
    if (source == Device::CUDA) return CopyKind::DeviceToHost;
    if (target == Device::CUDA) return CopyKind::HostToDevice;

    return CopyKind::HostToHost;
}

#ifdef OPENNN_HAS_CUDA

cudaMemcpyKind to_cuda_copy_kind(CopyKind kind)
{
    switch (kind)
    {
        case CopyKind::HostToHost:     return cudaMemcpyHostToHost;
        case CopyKind::HostToDevice:   return cudaMemcpyHostToDevice;
        case CopyKind::DeviceToHost:   return cudaMemcpyDeviceToHost;
        case CopyKind::DeviceToDevice: return cudaMemcpyDeviceToDevice;
    }

    throw runtime_error("unsupported CUDA copy kind.");
}

void* allocate_cuda(Index byte_count)
{
    void* device_pointer = nullptr;
    CHECK_CUDA(cudaMalloc(&device_pointer, static_cast<size_t>(byte_count)));
    return device_pointer;
}

void deallocate_cuda(void* pointer)
{
    cudaFree(pointer);
}

void set_zero_cuda(void* data, Index byte_count)
{
    CHECK_CUDA(cudaMemset(data, 0, static_cast<size_t>(byte_count)));
}

void set_zero_async_impl(void* data, Index byte_count, cudaStream_t stream)
{
    CHECK_CUDA(stream ? cudaMemsetAsync(data, 0, static_cast<size_t>(byte_count), stream)
                      : cudaMemset(data, 0, static_cast<size_t>(byte_count)));
}

void copy_async_impl(void* destination,
                     const void* source,
                     Index byte_count,
                     CopyKind kind,
                     cudaStream_t stream)
{
    const cudaMemcpyKind cuda_kind = to_cuda_copy_kind(kind);

    if (stream)
        CHECK_CUDA(cudaMemcpyAsync(destination, source,
                                   static_cast<size_t>(byte_count),
                                   cuda_kind,
                                   stream));
    else
        CHECK_CUDA(cudaMemcpy(destination, source,
                              static_cast<size_t>(byte_count),
                              cuda_kind));
}

void synchronize_impl(cudaStream_t stream)
{
    if (stream)
        CHECK_CUDA(cudaStreamSynchronize(stream));
    else
        CHECK_CUDA(cudaDeviceSynchronize());
}

void check_last_error_impl()
{
    CHECK_CUDA(cudaPeekAtLastError());
}

#else

void* allocate_cuda(Index)
{
    throw runtime_error("CUDA support is not compiled in.");
}

void deallocate_cuda(void*)
{
}

void set_zero_cuda(void*, Index)
{
    throw runtime_error("CUDA support is not compiled in.");
}

void set_zero_async_impl(void* data, Index byte_count, cudaStream_t)
{
    memset(data, 0, static_cast<size_t>(byte_count));
}

void copy_async_impl(void* destination,
                     const void* source,
                     Index byte_count,
                     CopyKind kind,
                     cudaStream_t)
{
    throw_if(kind != CopyKind::HostToHost,
             "CUDA support is not compiled in.");

    memcpy(destination, source, static_cast<size_t>(byte_count));
}

void synchronize_impl(cudaStream_t)
{
}

void check_last_error_impl()
{
}

#endif

}

bool is_cuda_build() noexcept
{
#ifdef OPENNN_HAS_CUDA
    return true;
#else
    return false;
#endif
}

bool has_cuda_device() noexcept
{
#ifdef OPENNN_HAS_CUDA
    int count = 0;
    const cudaError_t error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess)
    {
        cudaGetLastError();
        return false;
    }

    return count > 0;
#else
    return false;
#endif
}

int cuda_compute_capability() noexcept
{
#ifdef OPENNN_HAS_CUDA
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess)
    {
        cudaGetLastError();
        return -1;
    }

    return properties.major * 10 + properties.minor;
#else
    return -1;
#endif
}

size_t available_memory()
{
#ifdef OPENNN_HAS_CUDA
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
#else
    throw runtime_error("CUDA support is not compiled in.");
#endif
}

bool cuda_allocation_growth_forbidden() noexcept
{
    return cuda_allocation_growth_forbidden_runtime.load(std::memory_order_relaxed)
        || env_flag_enabled("OPENNN_CUDA_NO_ALLOC_GROWTH");
}

void set_cuda_allocation_growth_forbidden(bool forbidden) noexcept
{
    cuda_allocation_growth_forbidden_runtime.store(forbidden, std::memory_order_relaxed);
}

CudaAllocationGrowthGuard::CudaAllocationGrowthGuard(bool enabled)
    : active(enabled && is_cuda_build())
{
    if (active)
    {
        previous = cuda_allocation_growth_forbidden();
        set_cuda_allocation_growth_forbidden(true);
    }
}

CudaAllocationGrowthGuard::~CudaAllocationGrowthGuard() noexcept
{
    if (active)
        set_cuda_allocation_growth_forbidden(previous);
}

void* allocate(Device device_type, Index byte_count)
{
    throw_if_auto(device_type);
    throw_if(byte_count < 0, "device allocation size cannot be negative.");

    if (byte_count == 0) return nullptr;

    if (device_type == Device::CUDA)
    {
        throw_if(cuda_allocation_growth_forbidden(),
                 format("CUDA allocation of {} bytes while CUDA allocation growth is forbidden "
                        "(warmup incomplete before CUDA graph capture).",
                        byte_count));
        return allocate_cuda(byte_count);
    }

    return Eigen::aligned_allocator<uint8_t>{}.allocate(static_cast<size_t>(byte_count));
}

void deallocate(Device device_type, void* pointer, Index byte_count)
{
    if (!pointer) return;

    throw_if_auto(device_type);

    if (device_type == Device::CUDA)
    {
        deallocate_cuda(pointer);
        return;
    }

    Eigen::aligned_allocator<uint8_t>{}.deallocate(static_cast<uint8_t*>(pointer),
                                                   static_cast<size_t>(byte_count));
}

void set_zero(void* data, Index byte_count, Device device_type)
{
    throw_if_auto(device_type);
    throw_if(byte_count < 0, "device memset size cannot be negative.");

    if (!data || byte_count == 0) return;

    if (device_type == Device::CUDA)
    {
        set_zero_cuda(data, byte_count);
        return;
    }

    memset(data, 0, static_cast<size_t>(byte_count));
}

void set_zero_async(void* data, Index byte_count, cudaStream_t stream)
{
    throw_if(byte_count < 0, "device async memset size cannot be negative.");

    if (!data || byte_count == 0) return;

    set_zero_async_impl(data, byte_count, stream);
}

void copy_async(void* destination,
                const void* source,
                Index byte_count,
                CopyKind kind,
                cudaStream_t stream)
{
    throw_if(byte_count < 0, "device copy size cannot be negative.");

    if (byte_count == 0 || !destination || !source) return;

    copy_async_impl(destination, source, byte_count, kind, stream);
}

void copy_async(void* destination,
                const void* source,
                Index byte_count,
                Device source_device,
                Device target_device,
                cudaStream_t stream)
{
    copy_async(destination, source, byte_count, copy_kind(source_device, target_device), stream);
}

void synchronize(cudaStream_t stream)
{
    synchronize_impl(stream);
}

void check_last_error()
{
    check_last_error_impl();
}

cudaStream_t create_stream(unsigned flags)
{
#ifdef OPENNN_HAS_CUDA
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, flags));
    return stream;
#else
    (void)flags;
    return nullptr;
#endif
}

void destroy_stream(cudaStream_t stream)
{
    if (!stream) return;

#ifdef OPENNN_HAS_CUDA
    cudaStreamDestroy(stream);
#endif
}

void* allocate_pinned_host(Index byte_count)
{
    throw_if(byte_count < 0, "pinned host allocation size cannot be negative.");

    if (byte_count == 0) return nullptr;

#ifdef OPENNN_HAS_CUDA
    void* host_pointer = nullptr;
    CHECK_CUDA(cudaMallocHost(&host_pointer, static_cast<size_t>(byte_count)));
    return host_pointer;
#else
    void* host_pointer = malloc(static_cast<size_t>(byte_count));
    if (!host_pointer) throw bad_alloc();
    return host_pointer;
#endif
}

void deallocate_pinned_host(void* pointer)
{
    if (!pointer) return;

#ifdef OPENNN_HAS_CUDA
    cudaFreeHost(pointer);
#else
    free(pointer);
#endif
}

cudaEvent_t create_event(unsigned flags)
{
#ifdef OPENNN_HAS_CUDA
    cudaEvent_t event = nullptr;
    CHECK_CUDA(cudaEventCreateWithFlags(&event, flags));
    return event;
#else
    (void)flags;
    return nullptr;
#endif
}

cudaEvent_t create_event()
{
#ifdef OPENNN_HAS_CUDA
    return create_event(cudaEventDisableTiming);
#else
    return nullptr;
#endif
}

void destroy_event(cudaEvent_t event)
{
    if (!event) return;

#ifdef OPENNN_HAS_CUDA
    cudaEventDestroy(event);
#endif
}

void record_event(cudaEvent_t event, cudaStream_t stream)
{
#ifdef OPENNN_HAS_CUDA
    throw_if(!event, "cannot record a null CUDA event.");
    CHECK_CUDA(cudaEventRecord(event, stream));
#else
    (void)event;
    (void)stream;
#endif
}

void synchronize_event(cudaEvent_t event)
{
    if (!event) return;

#ifdef OPENNN_HAS_CUDA
    CHECK_CUDA(cudaEventSynchronize(event));
#endif
}

void stream_wait_event(cudaStream_t stream, cudaEvent_t event)
{
    if (!event) return;

#ifdef OPENNN_HAS_CUDA
    CHECK_CUDA(cudaStreamWaitEvent(stream, event, 0));
#else
    (void)stream;
#endif
}

#ifdef OPENNN_HAS_CUDA

void begin_graph_capture(cudaStream_t stream)
{
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
}

void* end_graph_capture(cudaStream_t stream)
{
    cudaGraph_t graph = nullptr;
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

    cudaGraphExec_t exec = nullptr;
    CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CHECK_CUDA(cudaGraphDestroy(graph));
    return static_cast<void*>(exec);
}

void launch_graph(void* graph_exec, cudaStream_t stream)
{
    CHECK_CUDA(cudaGraphLaunch(static_cast<cudaGraphExec_t>(graph_exec), stream));
}

void destroy_graph(void* graph_exec)
{
    if (graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec));
}

#else

void  begin_graph_capture(cudaStream_t) { throw runtime_error("CUDA support is not compiled in."); }
void* end_graph_capture(cudaStream_t)   { throw runtime_error("CUDA support is not compiled in."); }
void  launch_graph(void*, cudaStream_t) { throw runtime_error("CUDA support is not compiled in."); }
void  destroy_graph(void*) {}

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
