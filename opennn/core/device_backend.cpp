//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E V I C E   B A C K E N D
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/device_backend.h"
#include "opennn/core/profiler.h"
#include "opennn/core/tensor_types.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/memory_debug.h"

#ifdef EIGEN_USE_MKL_ALL
#include <mkl_service.h>
#endif

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <utility>

#ifdef __linux__
#include <sched.h>
#endif
#include "opennn/core/cuda/kernel_cast.cuh"

namespace opennn
{

class Backend
{
public:

    static Backend& instance();
    ThreadPoolDevice* get_thread_pool_device();
    void set_threads_number(int);

    static cublasHandle_t get_cublas_handle()      { return instance().cublas(device::active_lane()); }
    static cublasLtHandle_t get_cublas_lt_handle() { return instance().cublas_lt_handle; }
    static cudnnHandle_t get_cudnn_handle()        { return instance().cudnn(device::active_lane()); }
    static cudnnOpTensorDescriptor_t get_op_tensor_add_descriptor()
    {
        Backend& backend = instance();
        backend.cudnn(0);
        return backend.op_tensor_add_descriptor;
    }

private:

    Backend();
    ~Backend();

    cublasHandle_t cublas(int lane);
    cudnnHandle_t cudnn(int lane);
    cudaStream_t stream(int lane);

    unique_ptr<ThreadPool> thread_pool;
    unique_ptr<ThreadPoolDevice> thread_pool_device;

    cublasLtHandle_t cublas_lt_handle = nullptr;
    cudnnOpTensorDescriptor_t op_tensor_add_descriptor = nullptr;

    std::mutex lane_mutex;
    std::array<cudaStream_t, device::MAX_LANES>   lane_streams{};
    std::array<cublasHandle_t, device::MAX_LANES> cublas_handles{};
    std::array<cudnnHandle_t, device::MAX_LANES>  cudnn_handles{};

    cudaStream_t transfer_stream = nullptr;

    friend cudaStream_t device::get_compute_stream();
    friend cudaStream_t device::get_transfer_stream();
    friend cudaStream_t device::lane_stream(int);
};

}

namespace opennn::device
{

namespace
{

static int device_poison_mode()
{
    static const int mode = int(env_int_or("OPENNN_DEVICE_POISON", 0));
    return mode;
}

static int device_poison_byte()
{
    return device_poison_mode() == 2 ? 0x00 : 0xFF;
}

#ifdef OPENNN_HAS_CUDA

static void fill_device_memory(void* pointer, int value, Index byte_count)
{
    if (cudaMemset(pointer, value, size_t(byte_count)) != cudaSuccess)
        cudaGetLastError();

    if (cudaDeviceSynchronize() != cudaSuccess)
        cudaGetLastError();
}

static void poison_device_memory(void* pointer, Index byte_count)
{
    fill_device_memory(pointer, device_poison_byte(), byte_count);
}

#endif

atomic_bool cuda_allocation_growth_forbidden_runtime{false};
atomic_bool cuda_matmul_plan_creation_forbidden_runtime{false};

cudaEvent_t create_event_handle(unsigned);
cudaEvent_t create_event_handle();
void destroy_event_handle(cudaEvent_t) noexcept;
cudaStream_t create_stream_handle(unsigned);
void destroy_stream_handle(cudaStream_t) noexcept;

#ifdef OPENNN_HAS_CUDA
bool cuda_matmul_plan_creation_forbidden() noexcept
{
    return cuda_matmul_plan_creation_forbidden_runtime.load(memory_order_relaxed);
}
#endif

constexpr int64_t conv_workspace_auto_ceiling = int64_t(256) * 1024 * 1024;
atomic<int64_t> conv_workspace_cap_mode{-1};
atomic<int64_t> conv_workspace_auto_bytes{conv_workspace_auto_ceiling};
atomic_bool conv_autotune_enabled_flag{false};
template<typename Rung> atomic<Rung>& rung_setting() noexcept
{
    static atomic<Rung> setting{Rung::Auto};
    return setting;
}

thread_local GraphWorkspaceRequirements* active_graph_workspace_requirements = nullptr;
thread_local const GraphWorkspaceViews* active_graph_workspace_views = nullptr;

void throw_if_auto(Device device_type)
{
    throw_if(device_type == Device::Auto,
             "device backend expects a resolved device.");
}

#ifndef OPENNN_HAS_CUDA
[[noreturn]] void throw_cuda_unavailable()
{
    throw runtime_error("CUDA support is not compiled in.");
}
#endif

void* allocate_cuda(Index byte_count)
{
#ifdef OPENNN_HAS_CUDA
    void* device_pointer = nullptr;
    const cudaError_t cuda_err = cudaMalloc(&device_pointer, static_cast<size_t>(byte_count));
    if (cuda_err != cudaSuccess)
        throw runtime_error(
            string("CUDA Error: ") + to_string(static_cast<int>(cuda_err)) +
            " in " + string(__FILE__) + ":" + to_string(__LINE__) +
            " — cudaMalloc(" + to_string(byte_count) + " bytes = " +
            to_string(byte_count / Index(1024*1024)) + " MiB)");
    return device_pointer;
#else
    (void)byte_count;
    throw_cuda_unavailable();
#endif
}

}

CudaGraphWorkspaceScope::CudaGraphWorkspaceScope(
    GraphWorkspaceRequirements& requirements,
    const GraphWorkspaceViews* views)
    : previous_requirements(active_graph_workspace_requirements),
      previous_views(active_graph_workspace_views)
{
    active_graph_workspace_requirements = &requirements;
    if (views)
    {
        owned_views = *views;
        active_graph_workspace_views = &owned_views;
    }
    else
        active_graph_workspace_views = nullptr;
}

CudaGraphWorkspaceScope::~CudaGraphWorkspaceScope() noexcept
{
    active_graph_workspace_requirements = previous_requirements;
    active_graph_workspace_views = previous_views;
}

namespace
{

#ifdef OPENNN_HAS_CUDA
optional<void*> graph_workspace_override(GraphWorkspaceKind kind,
                                         Index minimum_bytes)
{
    if (active_graph_workspace_requirements)
    {
        Index& high_water =
            (*active_graph_workspace_requirements)[size_t(kind)];
        high_water = max(high_water, minimum_bytes);
    }

    if (!active_graph_workspace_views) return nullopt;

    const auto view = (*active_graph_workspace_views)[size_t(kind)];
    throw_if(minimum_bytes > view.bytes,
             "CUDA graph workspace needs {} bytes, but the stable "
                    "capture buffer has {} bytes.",
                    minimum_bytes, view.bytes);

    return view.data;
}
#endif

}

bool has_cuda_device() noexcept
{
#ifdef OPENNN_HAS_CUDA

    static const bool available = []() noexcept
    {
        int count = 0;
        const cudaError_t error = cudaGetDeviceCount(&count);
        if (error != cudaSuccess)
        {
            cudaGetLastError();
            return false;
        }

        return count > 0;
    }();

    return available;
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
    throw_cuda_unavailable();
#endif
}

string gpu_info_string() noexcept
{
#ifdef OPENNN_HAS_CUDA
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return "GPU info unavailable";
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    int ver = 0;
    cudaRuntimeGetVersion(&ver);
    return format("{:<32s}  {:.0f} MB total / {:.0f} MB free  CC {}.{}  CUDA {:d}.{:d}",
                       p.name,
                       total_b / 1048576.0,
                       free_b  / 1048576.0,
                       p.major, p.minor,
                       ver / 1000, (ver % 1000) / 10);
#else
    return "CPU only";
#endif
}

bool cuda_allocation_growth_forbidden() noexcept
{
    return cuda_allocation_growth_forbidden_runtime.load(memory_order_relaxed);
}

void set_cuda_allocation_growth_forbidden(bool forbidden) noexcept
{
    cuda_allocation_growth_forbidden_runtime.store(forbidden, memory_order_relaxed);
}

int64_t conv_workspace_limit_bytes() noexcept
{
    const int64_t mode = conv_workspace_cap_mode.load(memory_order_relaxed);
    return mode >= 0 ? mode : conv_workspace_auto_bytes.load(memory_order_relaxed);
}

void set_conv_workspace_cap(int64_t mode) noexcept
{
    conv_workspace_cap_mode.store(mode, memory_order_relaxed);
}

void set_conv_workspace_auto_limit_bytes(int64_t bytes) noexcept
{
    if (bytes > 0)
        conv_workspace_auto_bytes.store(min(bytes, conv_workspace_auto_ceiling), memory_order_relaxed);
}

bool conv_autotune_enabled() noexcept
{
    return conv_autotune_enabled_flag.load(memory_order_relaxed);
}

void set_conv_autotune(bool enabled) noexcept
{
    conv_autotune_enabled_flag.store(enabled, memory_order_relaxed);
}

template<typename Rung> Rung rung() noexcept
{
    return rung_setting<Rung>().load(memory_order_relaxed);
}

template<typename Rung> void set_rung(Rung value) noexcept
{
    rung_setting<Rung>().store(value, memory_order_relaxed);
}

#define OPENNN_RUNG(R) \
    template R rung<R>() noexcept; \
    template void set_rung<R>(R) noexcept;
OPENNN_RUNG(BatchNormBackwardRung)
OPENNN_RUNG(BatchNormForwardRung)
OPENNN_RUNG(MaxPoolingRung)
OPENNN_RUNG(AttentionRung)
#undef OPENNN_RUNG

CudaAllocationGrowthGuard::CudaAllocationGrowthGuard(
    bool enabled, bool forbid_matmul_plan_creation)
    : active(enabled && is_cuda_build()),
      guard_matmul_plans(active && forbid_matmul_plan_creation)
{
    if (active)
    {
        previous = cuda_allocation_growth_forbidden();
        set_cuda_allocation_growth_forbidden(true);
        if (guard_matmul_plans)
        {
            previous_matmul_plan_guard =
                cuda_matmul_plan_creation_forbidden_runtime.exchange(
                    true, memory_order_relaxed);
        }
    }
}

CudaAllocationGrowthGuard::~CudaAllocationGrowthGuard() noexcept
{
    if (active)
    {
        set_cuda_allocation_growth_forbidden(previous);
        if (guard_matmul_plans)
            cuda_matmul_plan_creation_forbidden_runtime.store(
                previous_matmul_plan_guard, memory_order_relaxed);
    }
}

namespace
{

#ifdef OPENNN_HAS_CUDA

class CudaBlockCache
{
public:

    static CudaBlockCache& instance()
    {
        static CudaBlockCache cache;
        return cache;
    }

    void* take(Index byte_count)
    {
        if (!is_enabled || byte_count <= 0) return nullptr;

        const lock_guard<mutex> guard(blocks_mutex);

        const auto entry = blocks.find(byte_count);
        if (entry == blocks.end())
        {
            note("blockcache:miss");
            return nullptr;
        }

        vector<CachedBlock>& candidates = entry->second;

        const auto ready = ranges::find_if(candidates, is_ready);

        if (ready == candidates.end())
        {
            note(candidates.empty() ? "blockcache:miss" : "blockcache:miss_pending");
            return nullptr;
        }

        note("blockcache:hit");

        void* pointer = ready->pointer;
        recycle_events(*ready);

        *ready = std::move(candidates.back());
        candidates.pop_back();
        cached_bytes -= byte_count;

        if (device_poison_mode() == 4)
            fill_device_memory(pointer, 0x00, byte_count);
        else if (poison_on_reuse)
            poison_device_memory(pointer, byte_count);

        return pointer;
    }

    bool give(void* pointer, Index byte_count) noexcept
    {
        if (!is_enabled || byte_count <= 0) return false;

        const lock_guard<mutex> guard(blocks_mutex);

        if (cached_bytes + byte_count > byte_cap)
        {
            note("blockcache:give_over_cap");
            return false;
        }

        note("blockcache:give");

        if (device_poison_mode() == 4)
        {
            if (cudaDeviceSynchronize() != cudaSuccess) cudaGetLastError();
            fill_device_memory(pointer, 0xFF, byte_count);
        }

        CachedBlock block;
        block.pointer = pointer;

        bool recorded = true;

        for (int lane = 0; lane < lanes_available(); ++lane)
            recorded = record_pending(block, lane_stream(lane)) && recorded;

        recorded = record_pending(block, get_transfer_stream()) && recorded;

        if (!recorded)
        {
            recycle_events(block);
            return false;
        }

        blocks[byte_count].push_back(std::move(block));
        cached_bytes += byte_count;

        return true;
    }

    bool flush()
    {
        const lock_guard<mutex> guard(blocks_mutex);

        bool released = false;

        for (auto& [size_in_bytes, cached] : blocks)
        {
            for (CachedBlock& block : cached)
            {
                for (cudaEvent_t event : block.pending_events)
                    cudaEventSynchronize(event);

                recycle_events(block);
                cudaFree(block.pointer);
                released = true;
            }
        }

        blocks.clear();
        cached_bytes = 0;

        return released;
    }

private:

    static void note(const char* key)
    {
        if (profiler::is_enabled()) profiler::stats().add(key, 0.0);
    }

    struct CachedBlock
    {
        void* pointer = nullptr;
        vector<cudaEvent_t> pending_events;
    };

    CudaBlockCache()
        : is_enabled(env_flag_enabled("OPENNN_DEVICE_CACHE", true)),
          poison_on_reuse(device_poison_mode() != 0),
          byte_cap(read_cap_bytes())
    {
    }

    static Index read_cap_bytes()
    {
        const Index megabytes = Index(env_int_or("OPENNN_DEVICE_CACHE_MB", 512));
        return (megabytes > 0 ? megabytes : Index(512)) * 1024 * 1024;
    }

    bool record_pending(CachedBlock& block, cudaStream_t stream) noexcept
    {
        if (!stream) return true;

        cudaEvent_t event = nullptr;

        if (!event_pool.empty())
        {
            event = event_pool.back();
            event_pool.pop_back();
        }
        else if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess)
        {
            cudaGetLastError();
            return false;
        }

        if (!event) return false;

        if (cudaEventRecord(event, stream) != cudaSuccess)
        {
            cudaGetLastError();
            event_pool.push_back(event);
            return false;
        }

        block.pending_events.push_back(event);
        return true;
    }

    static bool is_ready(const CachedBlock& block)
    {
        return ranges::all_of(block.pending_events,
                              [](cudaEvent_t event)
                              {
                                  const cudaError_t status = cudaEventQuery(event);

                                  cudaGetLastError();

                                  return status == cudaSuccess;
                              });
    }

    void recycle_events(CachedBlock& block) noexcept
    {
        event_pool.insert(event_pool.end(),
                          block.pending_events.begin(), block.pending_events.end());

        block.pending_events.clear();
    }

    const bool is_enabled;
    const bool poison_on_reuse;
    const Index byte_cap;
    Index cached_bytes = 0;
    unordered_map<Index, vector<CachedBlock>> blocks;
    vector<cudaEvent_t> event_pool;
    mutex blocks_mutex;
};

#endif

}

void* allocate(Device device_type, Index byte_count)
{
    PROFILE_SCOPE_HOST("device:allocate");
    throw_if_auto(device_type);
    throw_if(byte_count < 0, "device allocation size cannot be negative.");

    if (byte_count == 0) return nullptr;

    if (device_type == Device::CUDA)
    {
#ifdef OPENNN_HAS_CUDA
        if (void* recycled = CudaBlockCache::instance().take(byte_count))
            return recycled;

        throw_if(cuda_allocation_growth_forbidden(),
                 "CUDA alloc of {} bytes forbidden (warmup incomplete).", byte_count);

        try
        {
            void* const fresh = allocate_cuda(byte_count);

            if (device_poison_mode() == 3) poison_device_memory(fresh, byte_count);

            return fresh;
        }
        catch (const runtime_error&)
        {
            if (!CudaBlockCache::instance().flush()) throw;
        }
#endif
        return allocate_cuda(byte_count);
    }

    return Eigen::aligned_allocator<uint8_t>{}.allocate(static_cast<size_t>(byte_count));
}

void deallocate(Device device_type, void* pointer, Index byte_count) noexcept
{
    if (!pointer) return;

    PROFILE_SCOPE_HOST("device:deallocate");

    if (device_type == Device::CUDA)
    {
#ifdef OPENNN_HAS_CUDA
        if (!CudaBlockCache::instance().give(pointer, byte_count))
            cudaFree(pointer);
#endif
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
#ifdef OPENNN_HAS_CUDA
        CHECK_CUDA(cudaMemset(data, 0, static_cast<size_t>(byte_count)));
#else
        throw_cuda_unavailable();
#endif
        return;
    }

    memset(data, 0, static_cast<size_t>(byte_count));
}

void set_zero_async(void* data, Index byte_count, cudaStream_t stream)
{
    throw_if(byte_count < 0, "device async memset size cannot be negative.");

    if (!data || byte_count == 0) return;

#ifdef OPENNN_HAS_CUDA
    CHECK_CUDA(stream ? cudaMemsetAsync(data, 0, static_cast<size_t>(byte_count), stream)
                      : cudaMemset(data, 0, static_cast<size_t>(byte_count)));
#else
    (void)stream;
    memset(data, 0, static_cast<size_t>(byte_count));
#endif
}

void copy_async(void* destination,
                const void* source,
                Index byte_count,
                CopyKind kind,
                cudaStream_t stream)
{
    throw_if(byte_count < 0, "device copy size cannot be negative.");

    if (byte_count == 0 || !destination || !source) return;

#ifdef OPENNN_HAS_CUDA
    cudaMemcpyKind cuda_kind = cudaMemcpyHostToHost;
    switch (kind)
    {
        case CopyKind::HostToHost:     cuda_kind = cudaMemcpyHostToHost;     break;
        case CopyKind::HostToDevice:   cuda_kind = cudaMemcpyHostToDevice;   break;
        case CopyKind::DeviceToHost:   cuda_kind = cudaMemcpyDeviceToHost;   break;
        case CopyKind::DeviceToDevice: cuda_kind = cudaMemcpyDeviceToDevice; break;
        default: throw runtime_error("Invalid device copy kind.");
    }

    CHECK_CUDA(stream
        ? cudaMemcpyAsync(destination, source, size_t(byte_count), cuda_kind, stream)
        : cudaMemcpy(destination, source, size_t(byte_count), cuda_kind));

#else
    (void)stream;
    if (kind != CopyKind::HostToHost) throw_cuda_unavailable();
    memcpy(destination, source, static_cast<size_t>(byte_count));
#endif
}

void copy_async(void* destination,
                const void* source,
                Index byte_count,
                Device source_device,
                Device target_device,
                cudaStream_t stream)
{
    throw_if_auto(source_device);
    throw_if_auto(target_device);

    CopyKind kind = CopyKind::HostToHost;
    if (source_device == Device::CUDA && target_device == Device::CUDA) kind = CopyKind::DeviceToDevice;
    else if (source_device == Device::CUDA)                             kind = CopyKind::DeviceToHost;
    else if (target_device == Device::CUDA)                             kind = CopyKind::HostToDevice;

    copy_async(destination, source, byte_count, kind, stream);
}

void synchronize(cudaStream_t stream)
{
#ifdef OPENNN_HAS_CUDA
    CHECK_CUDA(stream ? cudaStreamSynchronize(stream)
                      : cudaDeviceSynchronize());
#else
    (void)stream;
#endif
}

void check_last_error()
{
#ifdef OPENNN_HAS_CUDA
    CHECK_CUDA(cudaPeekAtLastError());
#endif
}

void reset_last_error() noexcept
{
#ifdef OPENNN_HAS_CUDA
    cudaGetLastError();
#endif
}

#ifdef OPENNN_HAS_CUDA
CublasPointerModeGuard::CublasPointerModeGuard(
    const cublasHandle_t new_handle,
    const cublasPointerMode_t mode)
    : handle(new_handle)
{
    CHECK_CUBLAS(cublasGetPointerMode(handle, &previous_mode));
    CHECK_CUBLAS(cublasSetPointerMode(handle, mode));
}

CublasPointerModeGuard::~CublasPointerModeGuard() noexcept
{
    if (handle) cublasSetPointerMode(handle, previous_mode);
}

CublasMathModeGuard::CublasMathModeGuard(
    const cublasHandle_t new_handle,
    const cublasMath_t mode)
    : handle(new_handle)
{
    CHECK_CUBLAS(cublasGetMathMode(handle, &previous_mode));
    CHECK_CUBLAS(cublasSetMathMode(handle, mode));
}

CublasMathModeGuard::~CublasMathModeGuard() noexcept
{
    if (handle) cublasSetMathMode(handle, previous_mode);
}
#endif

namespace
{

cudaStream_t create_stream_handle(unsigned flags)
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

void destroy_stream_handle(cudaStream_t stream) noexcept
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

void deallocate_pinned_host(void* pointer) noexcept
{
    if (!pointer) return;

#ifdef OPENNN_HAS_CUDA
    cudaFreeHost(pointer);
#else
    free(pointer);
#endif
}

cudaEvent_t create_event_handle(unsigned flags)
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

cudaEvent_t create_event_handle()
{
#ifdef OPENNN_HAS_CUDA
    return create_event_handle(cudaEventDisableTiming);
#else
    return nullptr;
#endif
}

void destroy_event_handle(cudaEvent_t event) noexcept
{
    if (!event) return;

#ifdef OPENNN_HAS_CUDA
    cudaEventDestroy(event);
#endif
}

}

PinnedBuffer::PinnedBuffer(const Index byte_count)
{
    resize_bytes(byte_count);
}

PinnedBuffer::PinnedBuffer(PinnedBuffer&& other) noexcept
    : pointer(std::exchange(other.pointer, nullptr)),
      allocated_bytes(std::exchange(other.allocated_bytes, 0))
{
}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept
{
    if (this == &other) return *this;

    reset();
    pointer = std::exchange(other.pointer, nullptr);
    allocated_bytes = std::exchange(other.allocated_bytes, 0);
    return *this;
}

PinnedBuffer::~PinnedBuffer() noexcept
{
    reset();
}

void PinnedBuffer::resize_bytes(const Index byte_count)
{
    throw_if(byte_count < 0, "pinned buffer size cannot be negative.");
    if (byte_count == allocated_bytes) return;

    PinnedBuffer replacement;
    replacement.pointer = allocate_pinned_host(byte_count);
    replacement.allocated_bytes = byte_count;
    swap(replacement);
}

void PinnedBuffer::grow_to(const Index minimum_bytes)
{
    throw_if(minimum_bytes < 0, "pinned buffer size cannot be negative.");
    if (minimum_bytes > allocated_bytes) resize_bytes(minimum_bytes);
}

void PinnedBuffer::reset() noexcept
{
    deallocate_pinned_host(pointer);
    pointer = nullptr;
    allocated_bytes = 0;
}

void PinnedBuffer::swap(PinnedBuffer& other) noexcept
{
    std::swap(pointer, other.pointer);
    std::swap(allocated_bytes, other.allocated_bytes);
}

CudaEvent::CudaEvent(const unsigned flags)
    : handle(create_event_handle(flags))
{
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept
    : handle(std::exchange(other.handle, nullptr))
{
}

CudaEvent& CudaEvent::operator=(CudaEvent&& other) noexcept
{
    if (this == &other) return *this;

    reset();
    handle = std::exchange(other.handle, nullptr);
    return *this;
}

CudaEvent::~CudaEvent() noexcept
{
    reset();
}

void CudaEvent::create()
{
    reset();
    handle = create_event_handle();
}

void CudaEvent::reset() noexcept
{
    destroy_event_handle(handle);
    handle = nullptr;
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

namespace
{

struct GraphDeleter { void operator()(cudaGraph_t graph) const noexcept { cudaGraphDestroy(graph); } };

using GraphHandle = unique_ptr<remove_pointer_t<cudaGraph_t>, GraphDeleter>;

void instantiate_or_update(GraphExecHandle& exec, cudaGraph_t graph)
{
    if (exec)
    {
        cudaGraphExecUpdateResultInfo update_info{};
        if (cudaGraphExecUpdate(exec.get(), graph, &update_info) == cudaSuccess)
            return;

        cudaGetLastError();
        exec.reset();
    }

    cudaGraphExec_t raw = nullptr;
    CHECK_CUDA(cudaGraphInstantiate(&raw, graph, nullptr, nullptr, 0));
    exec.reset(raw);
}

}

StreamCapture::StreamCapture(cudaStream_t new_stream)
    : stream(new_stream)
{
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
}

void StreamCapture::end(GraphExecHandle& exec)
{
    cudaGraph_t raw_graph = nullptr;
    CHECK_CUDA(cudaStreamEndCapture(stream, &raw_graph));
    finished = true;

    const GraphHandle graph(raw_graph);

    if (env_flag_enabled("OPENNN_PROFILE") || env_flag_enabled("OPENNN_GRAPH_NODES"))
    {
        size_t nodes = 0;
        if (cudaGraphGetNodes(graph.get(), nullptr, &nodes) == cudaSuccess)
            cerr << "CUDA graph captured: " << nodes << " nodes" << endl;
        cudaGetLastError();
    }

    instantiate_or_update(exec, graph.get());
}

StreamCapture::~StreamCapture() noexcept
{
    if (finished) return;

    cudaGraph_t orphan = nullptr;
    cudaStreamEndCapture(stream, &orphan);
    if (orphan) cudaGraphDestroy(orphan);
    cudaGetLastError();
}

void launch_graph(const GraphExecHandle& exec, cudaStream_t stream)
{
    CHECK_CUDA(cudaGraphLaunch(exec.get(), stream));
}

#else

StreamCapture::StreamCapture(cudaStream_t) { throw_cuda_unavailable(); }
StreamCapture::~StreamCapture() noexcept {}
void StreamCapture::end(GraphExecHandle&) { throw_cuda_unavailable(); }
void launch_graph(const GraphExecHandle&, cudaStream_t) { throw_cuda_unavailable(); }

#endif

namespace
{
thread_local int active_lane_index = 0;
}

int lanes_available() noexcept
{
    static const int lanes =
        int(clamp(env_int_or("OPENNN_LANES", 1), 1LL, static_cast<long long>(MAX_LANES)));

    return lanes;
}

int active_lane() noexcept
{
    return active_lane_index;
}

void set_active_lane(int lane)
{
    throw_if(lane < 0 || lane >= lanes_available(),
             "set_active_lane: lane {} outside the {} configured (OPENNN_LANES).", lane, lanes_available());
    active_lane_index = lane;
}

cudaStream_t lane_stream(int lane)
{
    return Backend::instance().stream(lane);
}

cudaStream_t get_compute_stream()
{
    return Backend::instance().stream(active_lane_index);
}

cudaStream_t get_transfer_stream()
{
    return Backend::instance().transfer_stream;
}

cublasHandle_t get_cublas_handle()
{
    return Backend::get_cublas_handle();
}

cublasLtHandle_t get_cublas_lt_handle()
{
    return Backend::get_cublas_lt_handle();
}

cudnnHandle_t get_cudnn_handle()
{
    return Backend::get_cudnn_handle();
}

cudnnOpTensorDescriptor_t get_op_tensor_add_descriptor()
{
    return Backend::get_op_tensor_add_descriptor();
}

}

namespace opennn
{

Backend::Backend()
{
    const char* const threads_env = getenv("OPENNN_THREADS");
    set_threads_number(threads_env ? atoi(threads_env) : 0);

#ifdef OPENNN_HAS_CUDA
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0)
    {
        cudaGetLastError();
        cerr << "OpenNN: no CUDA device available (" << cudaGetErrorString(status)
             << "); running on CPU.\n";
        return;
    }

    lane_streams[0] = device::create_stream_handle(cudaStreamDefault);
    transfer_stream = device::create_stream_handle(cudaStreamNonBlocking);

    CHECK_CUBLAS(cublasLtCreate(&cublas_lt_handle));
    CHECK_CUDNN(cudnnCreateOpTensorDescriptor(&op_tensor_add_descriptor));
    CHECK_CUDNN(cudnnSetOpTensorDescriptor(op_tensor_add_descriptor,
                                           CUDNN_OP_TENSOR_ADD,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_NOT_PROPAGATE_NAN));
#endif
}

cudaStream_t Backend::stream(int lane)
{
#ifdef OPENNN_HAS_CUDA
    if (!lane_streams[0]) return nullptr;
    if (lane == 0) return lane_streams[0];
    std::lock_guard<std::mutex> lock(lane_mutex);
    if (!lane_streams[lane])
        lane_streams[lane] = device::create_stream_handle(cudaStreamNonBlocking);
    return lane_streams[lane];
#else
    (void)lane;
    return nullptr;
#endif
}

cublasHandle_t Backend::cublas(int lane)
{
#ifdef OPENNN_HAS_CUDA
    cudaStream_t lane_stream = stream(lane);
    if (!lane_stream) return nullptr;
    std::lock_guard<std::mutex> lock(lane_mutex);
    if (!cublas_handles[lane])
    {
        CHECK_CUBLAS(cublasCreate(&cublas_handles[lane]));
        CHECK_CUBLAS(cublasSetMathMode(cublas_handles[lane], CUBLAS_TF32_TENSOR_OP_MATH));
        CHECK_CUBLAS(cublasSetStream(cublas_handles[lane], lane_stream));
    }
    return cublas_handles[lane];
#else
    (void)lane;
    return nullptr;
#endif
}

cudnnHandle_t Backend::cudnn(int lane)
{
#ifdef OPENNN_HAS_CUDA
    cudaStream_t lane_stream = stream(lane);
    if (!lane_stream) return nullptr;
    std::lock_guard<std::mutex> lock(lane_mutex);
    if (!cudnn_handles[lane])
    {
        CHECK_CUDNN(cudnnCreate(&cudnn_handles[lane]));
        CHECK_CUDNN(cudnnSetStream(cudnn_handles[lane], lane_stream));
    }
    return cudnn_handles[lane];
#else
    (void)lane;
    return nullptr;
#endif
}

Backend::~Backend()
{
#ifdef OPENNN_HAS_CUDA
    if (op_tensor_add_descriptor)
        cudnnDestroyOpTensorDescriptor(op_tensor_add_descriptor);

    if (cublas_lt_handle)
        cublasLtDestroy(cublas_lt_handle);

    for (int lane = 0; lane < device::MAX_LANES; ++lane)
    {
        if (cublas_handles[lane]) cublasDestroy(cublas_handles[lane]);
        if (cudnn_handles[lane])  cudnnDestroy(cudnn_handles[lane]);

        device::destroy_stream_handle(lane_streams[lane]);
    }

    device::destroy_stream_handle(transfer_stream);
#endif
}

void Backend::set_threads_number(int num_threads)
{
    if (num_threads <= 0)
    {
        // Affinity first, machine size second. `hardware_concurrency` counts
        // the cores the machine has, not the ones this process may run on, so
        // under `taskset` or a cgroup CPU limit it oversubscribes -- 28
        // threads onto 16 permitted CPUs in the benchmark harness, which also
        // makes every per-thread sizing decision downstream come out wrong.
#ifdef __linux__
        cpu_set_t permitted;

        if (sched_getaffinity(0, sizeof(permitted), &permitted) == 0)
            num_threads = CPU_COUNT(&permitted);
#endif
        if (num_threads <= 0) num_threads = thread::hardware_concurrency();
        if (num_threads <= 0) num_threads = omp_get_max_threads();
        if (num_threads <= 0) num_threads = 1;
    }

    thread_pool = make_unique<ThreadPool>(num_threads);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), num_threads);

    Eigen::setNbThreads(num_threads);
    omp_set_num_threads(num_threads);

    // Every parallel region must ask for the same team. libgomp keeps exactly
    // one pool of workers, sized to the last region: a region that wants fewer
    // threads makes the surplus exit, and the next full-size region creates
    // them again with `pthread_create` -- fresh stacks, cold caches, and a
    // barrier that waits for the kernel to schedule them. The LSTM forward
    // pass was paying six thread births and deaths per batch, 10% of its
    // throughput, from two sources that each looked harmless alone:
    //
    //  - `omp_set_dynamic(1)`, which lets libgomp size a team as the CPU
    //    count minus the fifteen-minute load average, so a desktop with a
    //    browser open gave OpenNN's own regions 10 of 16 threads while
    //    oneDNN, which pins dynamic off, asked for all 16;
    //  - MKL's own thread heuristic (`MKL_DYNAMIC`), which chose 10 threads
    //    for a 256x128 `sgemv` between two 16-thread oneDNN regions.
    //
    // Dynamic teams are opt-in (`OPENNN_OMP_DYNAMIC=1`) and MKL is told to use
    // this team, exactly as PyTorch's ATen does for the same reason.
    const char* const omp_dynamic = getenv("OPENNN_OMP_DYNAMIC");
    omp_set_dynamic(omp_dynamic ? atoi(omp_dynamic) : 0);
#if defined(_OPENMP) && _OPENMP >= 200805
    omp_set_max_active_levels(1);
#endif
#ifdef EIGEN_USE_MKL_ALL
    mkl_set_dynamic(0);
    mkl_set_num_threads(num_threads);
#endif
}

Backend& Backend::instance()
{
    static Backend backend;
    return backend;
}

ThreadPoolDevice* Backend::get_thread_pool_device()
{
    return thread_pool_device.get();
}

ThreadPoolDevice& get_device()
{
    return *Backend::instance().get_thread_pool_device();
}

void set_threads_number(const int threads_number)
{
    Backend::instance().set_threads_number(threads_number);
}

}

#ifdef OPENNN_HAS_CUDA

namespace opennn
{

namespace
{
    // Tiles whose shape is known, so that a candidate kernel's data movement
    // can be compared: a tile of rows x columns reads (1/rows + 1/columns)
    // bytes through L2 and shared memory per multiply-add, and board power
    // follows that ratio closely. Measured on an RTX 5070 Ti, one
    // 1024x8192x1024 bf16 GEMM with bias and ReLU:
    //
    //     64x64   0.0312 -> 265 W      128x240  0.0120 -> 179 W
    //     64x640  0.0172 -> 189 W      128x320  0.0109 -> 172 W
    //     80x512  0.0145 -> 182 W      256x160  0.0102 -> 169 W
    //
    // The arithmetic is identical in every one of them; the 96 W is traffic.
    // Listed lowest-traffic first, which is the order they are tried in --
    // from probed_tiles_begin on; see below.
    struct LtTile { int id; int rows; int columns; };

    constexpr LtTile lt_known_tiles[] = {
        {CUBLASLT_MATMUL_TILE_256x256, 256, 256}, {CUBLASLT_MATMUL_TILE_192x256, 192, 256},
        {CUBLASLT_MATMUL_TILE_256x192, 256, 192}, {CUBLASLT_MATMUL_TILE_256x160, 256, 160},
        {CUBLASLT_MATMUL_TILE_128x320, 128, 320}, {CUBLASLT_MATMUL_TILE_192x128, 192, 128},
        {CUBLASLT_MATMUL_TILE_128x256, 128, 256}, {CUBLASLT_MATMUL_TILE_256x128, 256, 128},
        {CUBLASLT_MATMUL_TILE_128x240, 128, 240}, {CUBLASLT_MATMUL_TILE_256x96,  256,  96},
        {CUBLASLT_MATMUL_TILE_128x192, 128, 192}, {CUBLASLT_MATMUL_TILE_128x160, 128, 160},
        {CUBLASLT_MATMUL_TILE_80x512,   80, 512}, {CUBLASLT_MATMUL_TILE_128x128, 128, 128},
        {CUBLASLT_MATMUL_TILE_64x640,   64, 640}, {CUBLASLT_MATMUL_TILE_128x64,  128,  64},
        {CUBLASLT_MATMUL_TILE_64x128,   64, 128}, {CUBLASLT_MATMUL_TILE_64x64,    64,  64},
    };

    // The first three rows -- 256x256, 192x256 and 256x192 -- are priced but
    // never probed: AlgoCheck refused every one of them in all 14,708
    // measured configurations, and the candidate scan below has no early
    // exit, so probing them is a wasted sweep over every algorithm and stage
    // on every new plan. They stay in the table because the driver's own
    // heuristic may still return one and tile_traffic has to price it.
    constexpr size_t probed_tiles_begin = 3;

    // Unknown tiles report infinite traffic: they are never preferred over a
    // tile whose cost is known, only kept when they are the fastest.
    float tile_traffic(int tile_id, int splitk_number = 1)
    {
        for (const LtTile& tile : lt_known_tiles)
            if (tile.id == tile_id)
                // A split-k kernel writes fp32 partials for every split and
                // reads them all back to reduce; that traffic is invisible to
                // 1/rows + 1/columns, so charge one extra pass per split
                // rather than let a split-k kernel look cheap. It can still be
                // selected on time, only never preferred on traffic.
                return float(max(splitk_number, 1))
                     * (1.0f / float(tile.rows) + 1.0f / float(tile.columns));
        return numeric_limits<float>::infinity();
    }

    // Board power is close to linear in traffic over the tiles above -- 169 W
    // at 0.0102, 265 W at 0.0312 -- so a two-point fit is enough to compare
    // candidates by energy instead of by time alone. It overstates the
    // interior points by at most 6% (0.0172 models at 201 W against 189 W
    // measured), always in the direction of the thirstier tile. An unknown
    // tile has infinite traffic and so is priced at the top of the range,
    // the same worst-case assumption tile_traffic already makes about it.
    float tile_power_watts(float traffic)
    {
        constexpr float lowest_traffic  = 0.0102f, lowest_watts  = 169.0f;
        constexpr float highest_traffic = 0.0312f, highest_watts = 265.0f;
        constexpr float watts_per_traffic =
            (highest_watts - lowest_watts) / (highest_traffic - lowest_traffic);

        return lowest_watts
             + watts_per_traffic * (clamp(traffic, lowest_traffic, highest_traffic) - lowest_traffic);
    }

    struct LtMatmulCandidate
    {
        cublasLtMatmulAlgo_t algorithm{};
        size_t               workspace_bytes = 0;
        float                traffic = numeric_limits<float>::infinity();
    };

    struct LtMatmulPlan
    {
        cublasLtMatmulDesc_t   matmul_descriptor = nullptr;
        cublasLtMatrixLayout_t a_matrix_layout = nullptr;
        cublasLtMatrixLayout_t b_matrix_layout = nullptr;
        cublasLtMatrixLayout_t output_matrix_layout = nullptr;
        cublasLtMatmulAlgo_t   algorithm{};
        bool                   has_algorithm = false;
        size_t                 workspace_bytes = 0;

        vector<LtMatmulCandidate> candidates;
        bool                   tuned = true;

        LtMatmulPlan() = default;
        LtMatmulPlan(const LtMatmulPlan&) = delete;
        LtMatmulPlan& operator=(const LtMatmulPlan&) = delete;
        LtMatmulPlan& operator=(LtMatmulPlan&&) = delete;
        LtMatmulPlan(LtMatmulPlan&& other) noexcept
        {
            swap(matmul_descriptor, other.matmul_descriptor);
            swap(a_matrix_layout, other.a_matrix_layout);
            swap(b_matrix_layout, other.b_matrix_layout);
            swap(output_matrix_layout, other.output_matrix_layout);
            swap(algorithm, other.algorithm);
            swap(has_algorithm, other.has_algorithm);
            swap(workspace_bytes, other.workspace_bytes);
            swap(candidates, other.candidates);
            swap(tuned, other.tuned);
        }

        ~LtMatmulPlan()
        {
            cublasLtMatrixLayoutDestroy(output_matrix_layout);
            cublasLtMatrixLayoutDestroy(b_matrix_layout);
            cublasLtMatrixLayoutDestroy(a_matrix_layout);
            cublasLtMatmulDescDestroy(matmul_descriptor);
        }
    };

    // Everything a plan's descriptor and layouts are built from, and nothing
    // else. alpha never enters -- it is a call-time pointer that changes no
    // kernel -- and beta enters only as beta_is_zero, because a nonzero beta
    // makes the kernel read C, which changes which algorithm is valid and
    // which is fastest, while its value does not.
    //
    // Every field added here canonicalises to what the call sites already
    // implied: dtype_b equals dtype_a unless the operands genuinely differ,
    // and the three leading dimensions are stored as 0 when derived from m, n
    // and k. That is not tidiness. get_lt_matmul_plan throws on a miss and
    // optimizer.cpp holds cuda_matmul_plan_creation_forbidden across the whole
    // epoch loop, so a field that split one of today's keys in two would take
    // out training in steady state, not merely cost a plan.
    struct LtMatmulPlanKey
    {
        int m;
        int n;
        int k;
        int transA;
        int transB;
        int epilogue;
        int dtype_a;
        int dtype_b;
        int out_dtype;
        int lda;
        int ldb;
        int ldd;
        int beta_is_zero;

        bool operator==(const LtMatmulPlanKey&) const noexcept = default;
    };

    struct LtMatmulPlanKeyHash
    {
        size_t operator()(const LtMatmulPlanKey& key) const noexcept
        {
            return hash_combine(key.m, key.n, key.k,
                                key.transA, key.transB, key.epilogue,
                                key.dtype_a, key.dtype_b, key.out_dtype,
                                key.lda, key.ldb, key.ldd,
                                key.beta_is_zero);
        }
    };

    struct CudaMatmulThreadState
    {
        using LaneWorkspaces = std::array<Buffer, static_cast<size_t>(device::GraphWorkspaceKind::Count)>;
        std::array<LaneWorkspaces, device::MAX_LANES> workspaces =
            make_lanes(make_index_sequence<device::MAX_LANES>{});

        unordered_map<LtMatmulPlanKey, LtMatmulPlan, LtMatmulPlanKeyHash> lt_matmul_plans;

        template<size_t... I>
        static LaneWorkspaces make_workspaces(index_sequence<I...>)
        {
            return {((void)I, Buffer{Device::CUDA})...};
        }
        template<size_t... L>
        static std::array<LaneWorkspaces, sizeof...(L)> make_lanes(index_sequence<L...>)
        {
            return {((void)L, make_workspaces(make_index_sequence<static_cast<size_t>(device::GraphWorkspaceKind::Count)>{}))...};
        }
    };

    CudaMatmulThreadState& thread_state()
    {
        thread_local CudaMatmulThreadState state;
        return state;
    }

    constexpr size_t cublas_lt_workspace_search_bytes = 32ull * 1024 * 1024;
    constexpr size_t cublas_lt_plan_cache_capacity = 1024;

    cublasComputeType_t matmul_compute_type(cudaDataType_t a_type, 
                                            cudaDataType_t b_type = CUDA_R_32F)
    {
        return a_type == CUDA_R_16BF || b_type == CUDA_R_16BF
            ? CUBLAS_COMPUTE_32F_FAST_16BF
            : CUBLAS_COMPUTE_DTYPE;
    }    

    // Only used to size the tuner's scratch destination, so an unrecognised
    // type is rounded up rather than guessed: too large wastes a few bytes of
    // a block that is already megabytes, too small is a write past its end.
    size_t matmul_dtype_bytes(cudaDataType_t type)
    {
        switch (type)
        {
        case CUDA_R_8I:   return 1;
        case CUDA_R_16F:
        case CUDA_R_16BF: return 2;
        default:          return 4;
        }
    }

    void* thread_workspace(device::GraphWorkspaceKind kind, Index minimum_bytes)
    {
        if (device::active_lane() == 0)
            if (const optional<void*> graph_workspace =
                    device::graph_workspace_override(kind, minimum_bytes))
                return *graph_workspace;

        Buffer& buffer = thread_state().workspaces[static_cast<size_t>(device::active_lane())][static_cast<size_t>(kind)];
        if (minimum_bytes > buffer.byte_size() && buffer.data())
        {
            throw_if(device::cuda_allocation_growth_forbidden(),
                     "workspace growth forbidden (warmup incomplete).");
            device::synchronize(device::get_compute_stream());
        }
        const Index before = buffer.byte_size();
        void* pointer = buffer.ensure<uint8_t>(minimum_bytes);
        if (buffer.byte_size() > before)
            memory_debug::record(string("workspace.") + device::graph_workspace_labels[static_cast<size_t>(kind)],
                                 device::graph_workspace_labels[static_cast<size_t>(kind)],
                                 buffer.byte_size() - before, "high_water");
        return pointer;
    }

    // cuBLASLt's heuristic returns a handful of candidates and ranks them by
    // expected speed, so the low-traffic kernels never appear: for the dense
    // benchmark's 1024x8192x1024 GEMM it offers eight, of which the fastest
    // (64x64) draws 96 W more than a 256x160 that is 4% slower and that the
    // heuristic does not offer at all. This adds the tiles of
    // `lt_known_tiles` as extra candidates -- lowest-traffic first, over every
    // stage and custom option each algorithm advertises -- so that
    // `autotune_lt_plan` has something to choose between. A check costs under
    // 2 us and only the survivors are ever timed.
    void add_wide_tile_candidates(LtMatmulPlan& plan,
                                  const vector<cublasLtMatmulHeuristicResult_t>& heuristics,
                                  int m, int n,
                                  cudaDataType_t dtype_a,
                                  cudaDataType_t dtype_b,
                                  cudaDataType_t out_dtype)
    {
        // Every candidate kept here costs four timed matmuls in
        // autotune_lt_plan, so the overall cap stays where it was and the
        // wider option sweep is paid for out of it: 96 / 16 is six tiles,
        // each given four options over four stages instead of one option
        // over sixteen. Budget is charged on candidates kept, not on checks
        // attempted, so a tile the driver refuses outright costs nothing and
        // the next one gets the slots. Six is enough because the six probed
        // rows -- 256x160 through 128x240 -- already contain every tile at
        // or under the default OPENNN_LT_TRAFFIC_BUDGET of 0.0120, and a
        // tile above the budget can only ever be picked for being fastest,
        // which is what the driver's own heuristic already searches for.
        constexpr size_t most_added_candidates = 96;
        constexpr size_t most_candidates_per_tile = 16;
        constexpr int most_options_per_stage = 4;
        constexpr int most_custom_options = 256;

        vector<int> algorithm_ids;
        for (const cublasLtMatmulHeuristicResult_t& heuristic : heuristics)
        {
            int id = 0;
            size_t written = 0;
            if (cublasLtMatmulAlgoConfigGetAttribute(&heuristic.algo, CUBLASLT_ALGO_CONFIG_ID,
                                                     &id, sizeof(id), &written) != CUBLAS_STATUS_SUCCESS)
                continue;
            if (find(algorithm_ids.begin(), algorithm_ids.end(), id) == algorithm_ids.end())
                algorithm_ids.push_back(id);
        }

        const size_t before = plan.candidates.size();
        size_t tile_before = before;

        const auto candidate_budget_spent = [&]
        {
            return plan.candidates.size() - before >= most_added_candidates
                || plan.candidates.size() - tile_before >= most_candidates_per_tile;
        };

        size_t tile_index = 0;
        for (const LtTile& tile : lt_known_tiles)
        {
            if (tile_index++ < probed_tiles_begin) continue;
            if (plan.candidates.size() - before >= most_added_candidates) break;

            // A tile larger than the product is priced for an output that is
            // not there -- 1/rows + 1/columns charged over rows x columns of
            // which the shape has fewer -- so the traffic the tie-break reads
            // for it is a fiction. Skip it before the sweep rather than after,
            // because the budgets above count candidates kept, not checks
            // attempted: a shape no wide tile fits otherwise pays the whole
            // enumeration (13,460 configurations in 0.86 s on this card) and
            // keeps nothing. An attention QK^T at n = 64 is exactly that.
            if (tile.rows > m || tile.columns > n) continue;

            tile_before = plan.candidates.size();

            for (const int id : algorithm_ids)
            {
                if (candidate_budget_spent()) break;

                cublasLtMatmulAlgo_t algorithm{};
                if (cublasLtMatmulAlgoInit(Backend::get_cublas_lt_handle(),
                                           matmul_compute_type(dtype_a, dtype_b), CUDA_R_32F,
                                           dtype_a, dtype_b, out_dtype, out_dtype,
                                           id, &algorithm) != CUBLAS_STATUS_SUCCESS)
                {
                    device::reset_last_error();
                    continue;
                }

                size_t written = 0;
                vector<int> stages;
                if (cublasLtMatmulAlgoCapGetAttribute(&algorithm, CUBLASLT_ALGO_CAP_STAGES_IDS,
                                                      nullptr, 0, &written) == CUBLAS_STATUS_SUCCESS
                    && written > 0)
                {
                    stages.resize(written / sizeof(int));
                    cublasLtMatmulAlgoCapGetAttribute(&algorithm, CUBLASLT_ALGO_CAP_STAGES_IDS,
                                                      stages.data(), written, &written);
                }
                if (stages.empty()) stages.push_back(CUBLASLT_MATMUL_STAGES_UNDEFINED);

                int custom_maximum = 0;
                cublasLtMatmulAlgoCapGetAttribute(&algorithm, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX,
                                                  &custom_maximum, sizeof(custom_maximum), &written);
                custom_maximum = min(custom_maximum, most_custom_options);

                const auto set_config = [&](cublasLtMatmulAlgoConfigAttributes_t attribute, int value)
                {
                    cublasLtMatmulAlgoConfigSetAttribute(&algorithm, attribute, &value, sizeof(value));
                };
                set_config(CUBLASLT_ALGO_CONFIG_TILE_ID, tile.id);
                set_config(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 1);
                set_config(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_REDUCTION_SCHEME_NONE);
                set_config(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, 0);

                // Sweep the custom options rather than keep the first that
                // checks out. They do not "differ by well under a percent" as
                // this once claimed: over the timed configurations the first
                // valid option is 55.0% slower than the best one of the same
                // tile on 128x160, 31.1% on 128x192, 30.0% on 128x240 and
                // 29.9% on 256x96. Which one wins has to be timed, so keep
                // them all -- up to most_options_per_stage -- and let
                // autotune_lt_plan decide.
                for (const int stage : stages)
                {
                    if (candidate_budget_spent()) break;

                    set_config(CUBLASLT_ALGO_CONFIG_STAGES_ID, stage);

                    int kept_options = 0;
                    for (int custom = 0;
                         custom <= custom_maximum
                         && kept_options < most_options_per_stage
                         && !candidate_budget_spent();
                         ++custom)
                    {
                        set_config(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, custom);

                        cublasLtMatmulHeuristicResult_t check{};
                        if (cublasLtMatmulAlgoCheck(Backend::get_cublas_lt_handle(),
                                                    plan.matmul_descriptor,
                                                    plan.a_matrix_layout,
                                                    plan.b_matrix_layout,
                                                    plan.output_matrix_layout,
                                                    plan.output_matrix_layout,
                                                    &algorithm, &check) != CUBLAS_STATUS_SUCCESS
                            || check.state != CUBLAS_STATUS_SUCCESS
                            || check.workspaceSize > cublas_lt_workspace_search_bytes)
                        {
                            device::reset_last_error();
                            continue;
                        }

                        plan.candidates.push_back(
                            {algorithm, check.workspaceSize,
                             1.0f / float(tile.rows) + 1.0f / float(tile.columns)});
                        ++kept_options;
                    }
                }
            }
        }
    }

    LtMatmulPlan& get_lt_matmul_plan(
        int m, int n, int k,
        cublasOperation_t transA,
        cublasOperation_t transB,
        cublasLtEpilogue_t epilogue,
        cudaDataType_t dtype_a,
        cudaDataType_t dtype_b,
        cudaDataType_t out_dtype,
        int lda, int ldb, int ldd,
        bool beta_is_zero)
    {
        const LtMatmulPlanKey key{m, n, k,
                                  int(transA), int(transB), int(epilogue),
                                  int(dtype_a), int(dtype_b), int(out_dtype),
                                  lda, ldb, ldd, int(beta_is_zero)};
        auto& plans = thread_state().lt_matmul_plans;
        auto it = plans.find(key);
        if (it != plans.end()) return it->second;

        throw_if(device::cuda_matmul_plan_creation_forbidden(),
                 "matmul plan forbidden (warmup incomplete).");

        detail::make_bounded_cache_room(plans, cublas_lt_plan_cache_capacity);

        LtMatmulPlan plan;

        CHECK_CUBLAS(cublasLtMatmulDescCreate(&plan.matmul_descriptor, matmul_compute_type(dtype_a, dtype_b), CUDA_R_32F));

        auto set_desc = [&](cublasLtMatmulDescAttributes_t attr, const auto& value)
        {
            CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.matmul_descriptor, attr, &value, sizeof(value)));
        };

        set_desc(CUBLASLT_MATMUL_DESC_TRANSA,   transA);
        set_desc(CUBLASLT_MATMUL_DESC_TRANSB,   transB);
        set_desc(CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue);
        set_desc(CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, out_dtype);

        if (epilogue == CUBLASLT_EPILOGUE_RELU_AUX_BIAS
            || epilogue == CUBLASLT_EPILOGUE_DRELU)
            throw_if(m % 128 != 0,
                     "cuBLASLt ReLU bitmask epilogue requires m % 128 == 0, got {}.", m);

        if (epilogue == CUBLASLT_EPILOGUE_GELU_AUX_BIAS
            || epilogue == CUBLASLT_EPILOGUE_RELU_AUX_BIAS
            || epilogue == CUBLASLT_EPILOGUE_DRELU)
        {
            const int64_t aux_ld = m;
            set_desc(CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, aux_ld);
        }

        const int a_rows = (transA == CUBLAS_OP_N) ? m : k;
        const int a_cols = (transA == CUBLAS_OP_N) ? k : m;
        const int b_rows = (transB == CUBLAS_OP_N) ? k : n;
        const int b_cols = (transB == CUBLAS_OP_N) ? n : k;

        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.a_matrix_layout,  dtype_a,  a_rows, a_cols, lda ? lda : a_rows));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.b_matrix_layout,  dtype_b,  b_rows, b_cols, ldb ? ldb : b_rows));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.output_matrix_layout, out_dtype, m, n, ldd ? ldd : m));

        cublasLtMatmulPreference_t pref = nullptr;
        CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
        CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &cublas_lt_workspace_search_bytes, sizeof(cublas_lt_workspace_search_bytes)));

        const int requested = int(clamp(env_int_or("OPENNN_LT_AUTOTUNE_CANDIDATES", 8), 1LL, 32LL));
        vector<cublasLtMatmulHeuristicResult_t> heuristics(static_cast<size_t>(requested), cublasLtMatmulHeuristicResult_t{});
        int returned_results = 0;
        CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(Backend::get_cublas_lt_handle(),
                                                    plan.matmul_descriptor,
                                                    plan.a_matrix_layout,
                                                    plan.b_matrix_layout,
                                                    plan.output_matrix_layout,
                                                    plan.output_matrix_layout,
                                                    pref, requested,
                                                    heuristics.data(), &returned_results));
        cublasLtMatmulPreferenceDestroy(pref);

        heuristics.resize(static_cast<size_t>(max(returned_results, 0)));
        erase_if(heuristics, [](const cublasLtMatmulHeuristicResult_t& h) { return h.state != CUBLAS_STATUS_SUCCESS; });

        for (const cublasLtMatmulHeuristicResult_t& heuristic : heuristics)
        {
            int tile_id = CUBLASLT_MATMUL_TILE_UNDEFINED;
            int splitk_number = 1;
            size_t written = 0;
            cublasLtMatmulAlgoConfigGetAttribute(&heuristic.algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                                 &tile_id, sizeof(tile_id), &written);
            cublasLtMatmulAlgoConfigGetAttribute(&heuristic.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                                 &splitk_number, sizeof(splitk_number), &written);
            plan.candidates.push_back({heuristic.algo, heuristic.workspaceSize,
                                       tile_traffic(tile_id, splitk_number)});
        }

        add_wide_tile_candidates(plan, heuristics, m, n, dtype_a, dtype_b, out_dtype);

        if (!plan.candidates.empty())
        {
            plan.algorithm = plan.candidates.front().algorithm;
            plan.has_algorithm = true;
            plan.workspace_bytes = plan.candidates.front().workspace_bytes;
            plan.tuned = plan.candidates.size() <= 1;
        }

        return plans.emplace(key, std::move(plan)).first->second;
    }

    void autotune_lt_plan(LtMatmulPlan& plan,
                          const void* a_data, const void* b_data,
                          const void* c_data, void* d_data,
                          float alpha, float beta, size_t destination_bytes,
                          cudaStream_t stream)
    {
        if (plan.candidates.size() <= 1)
        {
            plan.tuned = true;
            return;
        }

        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &capture_status) != cudaSuccess
            || capture_status != cudaStreamCaptureStatusNone)
            return device::reset_last_error();

        plan.tuned = true;

        if (device::lanes_available() > 1) device::synchronize();

        size_t largest_workspace = 0;
        for (const auto& candidate : plan.candidates)
            largest_workspace = max(largest_workspace, candidate.workspace_bytes);

        // The tuner runs on the caller's own pointers and used to force beta
        // to zero for one reason: it launches each candidate four times, and a
        // destination the kernel also reads would be accumulated into four
        // times over -- a recurrent layer sums one weight gradient per
        // timestep into the same tensor, which is precisely that. So when D is
        // an operand, time into scratch and throw the result away.
        //
        // The test is aliasing, not beta alone. With C separate from D the
        // real call overwrites every element of D whatever beta is, so timing
        // there costs nothing and destroys nothing; making it scratch anyway
        // would add the whole destination -- an input delta, tens of MB on the
        // transformer -- to a shared-scratch high-water that is never returned.
        //
        // That scratch is taken out of the same shared block the algorithm
        // workspace comes from, at an offset past it, rather than from a
        // GraphWorkspaceKind of its own. Reusing SharedScratch does not dodge
        // the high water -- ensure_shared_scratch goes through
        // graph_workspace_override like every other kind and records one --
        // but it raises a buffer that already exists instead of adding a
        // second permanent Buffer to every lane of every thread's workspace
        // array and a second entry to the captured graph's workspace set.
        // thread_workspace throws when that raise falls under the growth
        // guard; that is the steady-state case, and it is caught here rather
        // than propagated -- a plan first seen after warmup keeps the
        // heuristic's candidate untimed, which is the one thing that must not
        // become an exception inside a training step.
        const bool destination_is_operand = beta != 0.0f && c_data == static_cast<const void*>(d_data);

        constexpr size_t destination_alignment = 256;
        const size_t destination_offset =
            (largest_workspace + destination_alignment - 1)
            / destination_alignment * destination_alignment;

        void* workspace = nullptr;
        try
        {
            workspace = ensure_shared_scratch(destination_is_operand
                                              ? destination_offset + destination_bytes
                                              : largest_workspace);
        }
        catch (const exception&)
        {
            plan.candidates.clear();
            return;
        }

        void* const destination = destination_is_operand
            ? static_cast<char*>(workspace) + destination_offset
            : d_data;

        const device::CudaEvent start(cudaEventDefault), stop(cudaEventDefault);
        constexpr int timed_runs = 3;

        vector<float> times(plan.candidates.size(), numeric_limits<float>::infinity());
        float best_ms = numeric_limits<float>::infinity();
        size_t best = 0;
        for (size_t index = 0; index < plan.candidates.size(); ++index)
        {
            const auto& candidate = plan.candidates[index];
            const auto run = [&]
            {
                return cublasLtMatmul(Backend::get_cublas_lt_handle(), plan.matmul_descriptor,
                                      &alpha, a_data, plan.a_matrix_layout, b_data, plan.b_matrix_layout,
                                      &beta, c_data, plan.output_matrix_layout,
                                      destination, plan.output_matrix_layout,
                                      &candidate.algorithm, workspace, candidate.workspace_bytes, stream);
            };
            if (run() != CUBLAS_STATUS_SUCCESS) { device::reset_last_error(); continue; }
            device::record_event(start.get(), stream);
            bool ok = true;
            for (int i = 0; i < timed_runs && ok; ++i) ok = run() == CUBLAS_STATUS_SUCCESS;
            device::record_event(stop.get(), stream);
            device::synchronize_event(stop.get());
            if (!ok) { device::reset_last_error(); continue; }
            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start.get(), stop.get()));
            times[index] = ms;
            if (ms < best_ms) { best_ms = ms; best = index; }
        }

        // Two kernels that take the same time need not cost the same energy:
        // on the dense benchmark's inference GEMM a 256x160 tile costs 34%
        // less energy for 4% more time than the 64x64 the heuristic ranks
        // first, which is the difference between that cell costing 9% more
        // than PyTorch's and costing 25% less. So among the candidates take
        // the one whose tile moves the least data -- but bound what may be
        // taken by traffic, not by a window around whatever the fastest
        // candidate happens to be.
        //
        // That distinction is the whole point of the rewrite. The old rule
        // kept anything within best_ms * 1.05, and the 256x160 sits at
        // 201.4 us against a 192.0 fastest -- 4.9%, one tenth of a point
        // inside the window. Any newly timed candidate at 191.8 us or less
        // pushed it back out and handed the cell to a 265 W kernel at 0.919x
        // energy, silently: no cell reports which tile it ran. Three
        // conditions now, and only the last one mentions best_ms.
        //
        //   * Traffic at or below OPENNN_LT_TRAFFIC_BUDGET, in units of 1e-4,
        //     default 120 = 0.0120. Every tile at or below 0.0120 measured
        //     within 10 W of the 169 W floor; the 64x64 is 0.0312 and 265 W.
        //     Nothing above the budget is ever preferred over the fastest, so
        //     a faster high-traffic kernel appearing cannot change the pick.
        //   * Lower modelled energy (time x tile_power_watts) than the
        //     fastest candidate, so time is only ever traded for a tile that
        //     is really cheaper: 8% slower for 2% less power is now refused,
        //     where a flat window took it. This is also what keeps a shape
        //     whose only low-traffic candidate is far slower on the fastest
        //     kernel -- the model breaks even at 265/169 = 1.57x.
        //   * At most OPENNN_LT_TILE_TOLERANCE percent slower than the
        //     fastest candidate, which bounds what any of this can cost
        //     throughput. The default is 10, not the old 5: 5 is under the
        //     4.9% the measured choice already spends, which is exactly what
        //     made it fragile. It is not higher because tile_power_watts is
        //     fitted on one L2-resident compute-bound GEMM and this rule now
        //     steers every matmul in the library; 10 leaves the measured
        //     choice a full point of headroom while bounding what an
        //     extrapolated power model can cost a shape nobody has measured.
        //
        // OPENNN_LT_TILE_TOLERANCE=0 still switches the rule off entirely and
        // restores the pick by time alone, which is the A/B.
        const float tolerance =
            float(clamp(env_int_or("OPENNN_LT_TILE_TOLERANCE", 10), 0LL, 100LL)) / 100.0f;
        const float traffic_budget =
            float(clamp(env_int_or("OPENNN_LT_TRAFFIC_BUDGET", 120), 1LL, 10000LL)) / 10000.0f;

        if (tolerance > 0.0f && best_ms < numeric_limits<float>::infinity())
        {
            const float allowed_ms = best_ms * (1.0f + tolerance);
            const float allowed_energy = best_ms * tile_power_watts(plan.candidates[best].traffic);
            float least_traffic = plan.candidates[best].traffic;

            for (size_t index = 0; index < plan.candidates.size(); ++index)
            {
                const LtMatmulCandidate& candidate = plan.candidates[index];

                if (candidate.traffic > traffic_budget) continue;
                if (candidate.traffic > least_traffic) continue;
                // Same tile, so same traffic: take the faster of the two.
                // The custom-option sweep above emits several candidates per
                // tile precisely because the first valid option is up to 55%
                // slower than the best one, so selecting the lowest-traffic
                // tile is not enough -- without this the pick would be the
                // first option in scan order, which is the one the sweep was
                // added to stop using.
                if (candidate.traffic == least_traffic && times[index] >= times[best]) continue;
                if (times[index] > allowed_ms) continue;
                if (times[index] * tile_power_watts(candidate.traffic) >= allowed_energy) continue;

                least_traffic = candidate.traffic;
                best = index;
            }
        }

        plan.algorithm = plan.candidates[best].algorithm;
        plan.workspace_bytes = plan.candidates[best].workspace_bytes;
        plan.candidates.clear();
    }
}

void* ensure_workspace_bytes(device::GraphWorkspaceKind kind, Index bytes)
{
    return thread_workspace(kind, bytes);
}

void release_thread_workspaces()
{
    device::synchronize(device::get_compute_stream());
    for (auto& lane : thread_state().workspaces)
        for (Buffer& buffer : lane)
            buffer.resize_bytes(0, Device::CUDA);
}

const void* data_for_gemm_dtype(const TensorView& input, Type target_type)
{
    if (input.get_type() == target_type) return input.get_data();

    if (input.is_fp32() && target_type == Type::BF16)
    {
        bfloat16* dst = ensure_workspace<bfloat16>(device::GraphWorkspaceKind::Bf16Input, input.size());
        cast_fp32_to_bf16(input.size(), input.as<float>(), dst);
        return dst;
    }

    if (input.is_bf16() && target_type == Type::FP32)
    {
        float* dst = ensure_bf16_to_fp32_workspace(input.size());
        cast_bf16_to_fp32(input.size(), input.as<bfloat16>(), dst);
        return dst;
    }

    throw runtime_error("data_for_gemm_dtype: unsupported type pair");
}

const void* bias_for_gemm_bf16(const TensorView& bias)
{

    bfloat16* dst = ensure_bf16_gradient_workspace(bias.size());
    cast_fp32_to_bf16(bias.size(), bias.as<float>(), dst);
    return dst;
}

void run_lt_matmul_cached(
    int m, int n, int k,
    cublasOperation_t transA,
    cublasOperation_t transB,
    cublasLtEpilogue_t epilogue,
    const void* a_data, const void* b_data, void* d_data,
    const void* bias_pointer,
    cudaDataType_t dtype_a,
    cudaDataType_t dtype_b,
    cudaDataType_t out_dtype,
    const void* aux_pointer,
    const void* addend,
    float alpha,
    float beta,
    int lda, int ldb, int ldd)
{
    // beta used to be derived from this pointer, so the combination could not
    // be asked for; now that it is a parameter, an addend at beta 0 is an
    // operand the kernel reads and discards. Say so rather than compute the
    // wrong sum quietly.
    throw_if(addend && beta == 0.0f,
             "run_lt_matmul_cached: an addend with beta 0 is a discarded operand.");

    LtMatmulPlan& plan = get_lt_matmul_plan(m, n, k, transA, transB, epilogue,
                                            dtype_a, dtype_b, out_dtype,
                                            lda, ldb, ldd, beta == 0.0f);

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.matmul_descriptor,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer, sizeof(bias_pointer)));

    if (aux_pointer)
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.matmul_descriptor,
            CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &aux_pointer, sizeof(aux_pointer)));

    // C is the addend when there is one and the destination otherwise; at
    // beta 0 the kernel reads neither, which is why aliasing it to D was
    // always safe and still is.
    const void* const c_data = addend ? addend : d_data;

    if (!plan.tuned)
        autotune_lt_plan(plan, a_data, b_data, c_data, d_data, alpha, beta,
                         size_t(ldd ? ldd : m) * size_t(n) * matmul_dtype_bytes(out_dtype),
                         device::get_compute_stream());

    CHECK_CUBLAS(cublasLtMatmul(Backend::get_cublas_lt_handle(),
                                plan.matmul_descriptor,
                                &alpha,
                                a_data, plan.a_matrix_layout,
                                b_data, plan.b_matrix_layout,
                                &beta,
                                c_data, plan.output_matrix_layout,
                                d_data, plan.output_matrix_layout,
                                plan.has_algorithm ? &plan.algorithm : nullptr,
                                ensure_shared_scratch(plan.workspace_bytes), 
                                plan.workspace_bytes,
                                device::get_compute_stream()));
}

void gemm_strided_batched_cuda(cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const void* A, cudaDataType_t Atype, int lda, long long stride_a,
                               const void* B, cudaDataType_t Btype, int ldb, long long stride_b,
                               void* C, cudaDataType_t Ctype, int ldc, long long stride_c,
                               int batch_count,
                               float alpha, float beta)
{
    const cublasComputeType_t compute = matmul_compute_type(Atype, Btype);
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(Backend::get_cublas_handle(),
                                            transa, transb,
                                            m, n, k,
                                            &alpha,
                                            A, Atype, lda, stride_a,
                                            B, Btype, ldb, stride_b,
                                            &beta,
                                            C, Ctype, ldc, stride_c,
                                            batch_count,
                                            compute,
                                            CUBLAS_GEMM_DEFAULT));
}

}

#else

namespace opennn
{

void* ensure_workspace_bytes(device::GraphWorkspaceKind, Index) OPENNN_CUDA_STUB_BODY(ensure_workspace_bytes)

void release_thread_workspaces() OPENNN_CUDA_STUB_BODY(release_thread_workspaces)

const void* data_for_gemm_dtype(const TensorView&, Type) OPENNN_CUDA_STUB_BODY(data_for_gemm_dtype)

const void* bias_for_gemm_bf16(const TensorView&) OPENNN_CUDA_STUB_BODY(bias_for_gemm_bf16)

void run_lt_matmul_cached(int, int, int,
                          cublasOperation_t,
                          cublasOperation_t,
                          cublasLtEpilogue_t,
                          const void*, const void*, void*,
                          const void*,
                          cudaDataType_t,
                          cudaDataType_t,
                          cudaDataType_t,
                          const void*,
                          const void*,
                          float,
                          float,
                          int, int, int) OPENNN_CUDA_STUB_BODY(run_lt_matmul_cached)

void gemm_strided_batched_cuda(cublasOperation_t, cublasOperation_t,
                               int, int, int,
                               const void*, cudaDataType_t, int, long long,
                               const void*, cudaDataType_t, int, long long,
                               void*, cudaDataType_t, int, long long,
                               int,
                               float, float) OPENNN_CUDA_STUB_BODY(gemm_strided_batched_cuda)

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
