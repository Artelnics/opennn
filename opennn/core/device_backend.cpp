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
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <utility>

#ifdef __linux__
#include <sched.h>
#endif
#include "opennn/core/cuda/kernel_cast.cuh"
#include "opennn/core/cuda/cudnn_matmul.h"

namespace opennn
{

#ifdef OPENNN_HAS_CUDA
namespace
{
// Trivial process-lifetime state: static user buffers may be destroyed after
// the backend or cache. They must not touch their destroyed streams or tables.
atomic_bool cuda_resources_shutting_down{false};
}
#endif

class Backend
{
public:

    static Backend& instance();
    ThreadPoolDevice* get_thread_pool_device();
    void set_threads_number(int);

    static cublasHandle_t get_cublas_handle()      { return instance().cublas(device::active_lane()); }
    static void apply_math_mode()                  { instance().set_math_mode_on_handles(); }
    static cublasLtHandle_t get_cublas_lt_handle()
    {
        Backend& backend = instance();
        backend.ensure_cuda();
        return backend.cublas_lt_handle;
    }
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
    void set_math_mode_on_handles();

    // The CUDA side of the backend -- the compute and transfer streams, the
    // cuBLASLt handle, the shared cuDNN descriptor -- is created on first use,
    // not in the constructor. Creating it is what creates the CUDA context,
    // and the constructor also runs for CPU-only work: the Eigen thread pool
    // lives here too, so a CPU forward pass used to pay for a context it never
    // touched (226 MiB of device memory, plus the driver's host-side state in
    // the process' resident set).
    void ensure_cuda();
    void release_cuda();
    static void register_cuda_cleanup();
    std::once_flag cuda_once;

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
thread_local bool cuda_block_cache_bypassed = false;

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

atomic_bool allow_tf32_flag{true};
bool allow_tf32_flag_initialised = false;
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

bool allow_tf32() noexcept
{
    if (!allow_tf32_flag_initialised)
    {
        allow_tf32_flag = env_flag_enabled("OPENNN_ALLOW_TF32", true);
        allow_tf32_flag_initialised = true;
    }
    return allow_tf32_flag;
}

void set_allow_tf32(bool enabled) noexcept
{
    allow_tf32_flag = enabled;
    allow_tf32_flag_initialised = true;
    Backend::apply_math_mode();
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

    ~CudaBlockCache()
    {
        cuda_resources_shutting_down.store(true, memory_order_relaxed);
        flush();
        for (cudaEvent_t event : event_pool) cudaEventDestroy(event);
    }

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
        // This cache is reached only for GPU storage. Initialize the runtime
        // before its destructor is registered, so cached blocks and events
        // can still be released during process shutdown.
        CHECK_CUDA(cudaFree(nullptr));
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
        if (!cuda_block_cache_bypassed)
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

CudaBlockCacheBypass::CudaBlockCacheBypass() noexcept
    : previous(cuda_block_cache_bypassed)
{
    cuda_block_cache_bypassed = true;
}

CudaBlockCacheBypass::~CudaBlockCacheBypass() noexcept
{
    cuda_block_cache_bypassed = previous;
}

void deallocate(Device device_type, void* pointer, Index byte_count) noexcept
{
    if (!pointer) return;

#ifdef OPENNN_HAS_CUDA
    if (device_type == Device::CUDA
        && cuda_resources_shutting_down.load(memory_order_relaxed))
    {
        cudaFree(pointer);
        return;
    }
#endif

    PROFILE_SCOPE_HOST("device:deallocate");

    if (device_type == Device::CUDA)
    {
#ifdef OPENNN_HAS_CUDA
        if (cuda_block_cache_bypassed || !CudaBlockCache::instance().give(pointer, byte_count))
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
        CHECK_CUDA(cudaMemsetAsync(data, 0, static_cast<size_t>(byte_count), get_compute_stream()));
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
    CHECK_CUDA(cudaMemsetAsync(data, 0, static_cast<size_t>(byte_count),
                              stream ? stream : get_compute_stream()));
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

    if (kind == CopyKind::HostToHost)
    {
        memcpy(destination, source, static_cast<size_t>(byte_count));
        return;
    }

    // The compute lanes are nonblocking CUDA streams. A synchronous copy on
    // CUDA's default stream does not wait for their pending kernels and can
    // return partially written gradients. Preserve the blocking default-copy
    // contract, but order it on the active compute lane.
    const cudaStream_t copy_stream = stream ? stream : get_compute_stream();
    CHECK_CUDA(cudaMemcpyAsync(destination, source, size_t(byte_count), cuda_kind, copy_stream));
    if (!stream) CHECK_CUDA(cudaStreamSynchronize(copy_stream));

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


}


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
    Backend& backend = Backend::instance();
    backend.ensure_cuda();
    return backend.transfer_stream;
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
}

void Backend::ensure_cuda()
{
#ifdef OPENNN_HAS_CUDA
    std::call_once(cuda_once, [this]
    {
        int device_count = 0;
        const cudaError_t status = cudaGetDeviceCount(&device_count);
        if (status != cudaSuccess || device_count == 0)
        {
            cudaGetLastError();
            logging::warning() << "OpenNN: no CUDA device available (" << cudaGetErrorString(status)
                 << "); running on CPU.\n";
            return;
        }

        lane_streams[0] = device::create_stream_handle(cudaStreamNonBlocking);
        transfer_stream = device::create_stream_handle(cudaStreamNonBlocking);

        CHECK_CUBLAS(cublasLtCreate(&cublas_lt_handle));
        CHECK_CUDNN(cudnnCreateOpTensorDescriptor(&op_tensor_add_descriptor));
        CHECK_CUDNN(cudnnSetOpTensorDescriptor(op_tensor_add_descriptor,
                                               CUDNN_OP_TENSOR_ADD,
                                               CUDNN_DATA_FLOAT,
                                               CUDNN_NOT_PROPAGATE_NAN));
        register_cuda_cleanup();
    });
#endif
}

cudaStream_t Backend::stream(int lane)
{
#ifdef OPENNN_HAS_CUDA
    ensure_cuda();
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
        CHECK_CUBLAS(cublasSetMathMode(cublas_handles[lane],
                                       device::allow_tf32() ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
        CHECK_CUBLAS(cublasSetStream(cublas_handles[lane], lane_stream));
        register_cuda_cleanup();
    }
    return cublas_handles[lane];
#else
    (void)lane;
    return nullptr;
#endif
}

void Backend::set_math_mode_on_handles()
{
#ifdef OPENNN_HAS_CUDA
    std::lock_guard<std::mutex> lock(lane_mutex);
    for (cublasHandle_t handle : cublas_handles)
        if (handle)
            CHECK_CUBLAS(cublasSetMathMode(handle, device::allow_tf32() ? CUBLAS_TF32_TENSOR_OP_MATH
                                                                       : CUBLAS_DEFAULT_MATH));
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
        register_cuda_cleanup();
    }
    return cudnn_handles[lane];
#else
    (void)lane;
    return nullptr;
#endif
}

Backend::~Backend()
{
    release_cuda();
}

void Backend::register_cuda_cleanup()
{
#ifdef OPENNN_HAS_CUDA
    // The backend can predate CUDA: CPU work constructs it without a context.
    // Register after each lazy library initialization so our handles are freed
    // before that library's own process-exit callbacks tear down its runtime.
    std::atexit([] { Backend::instance().release_cuda(); });
#endif
}

void Backend::release_cuda()
{
#ifdef OPENNN_HAS_CUDA
    cuda_resources_shutting_down.store(true, memory_order_relaxed);

    if (op_tensor_add_descriptor)
        cudnnDestroyOpTensorDescriptor(std::exchange(op_tensor_add_descriptor, nullptr));

    if (cublas_lt_handle)
        cublasLtDestroy(std::exchange(cublas_lt_handle, nullptr));

    for (int lane = 0; lane < device::MAX_LANES; ++lane)
    {
        if (cublas_handles[lane]) cublasDestroy(std::exchange(cublas_handles[lane], nullptr));
        if (cudnn_handles[lane])  cudnnDestroy(std::exchange(cudnn_handles[lane], nullptr));

        device::destroy_stream_handle(std::exchange(lane_streams[lane], nullptr));
    }

    device::destroy_stream_handle(std::exchange(transfer_stream, nullptr));
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
#if CUBLAS_VER_MAJOR >= 13
        {CUBLASLT_MATMUL_TILE_256x256, 256, 256}, {CUBLASLT_MATMUL_TILE_192x256, 192, 256},
        {CUBLASLT_MATMUL_TILE_256x192, 256, 192}, {CUBLASLT_MATMUL_TILE_256x160, 256, 160},
        {CUBLASLT_MATMUL_TILE_128x320, 128, 320},
#endif
        {CUBLASLT_MATMUL_TILE_192x128, 192, 128},
        {CUBLASLT_MATMUL_TILE_128x256, 128, 256}, {CUBLASLT_MATMUL_TILE_256x128, 256, 128},
#if CUBLAS_VER_MAJOR >= 13
        {CUBLASLT_MATMUL_TILE_128x240, 128, 240}, {CUBLASLT_MATMUL_TILE_256x96,  256,  96},
#endif
        {CUBLASLT_MATMUL_TILE_128x192, 128, 192}, {CUBLASLT_MATMUL_TILE_128x160, 128, 160},
#if CUBLAS_VER_MAJOR >= 13
        {CUBLASLT_MATMUL_TILE_80x512,   80, 512},
#endif
        {CUBLASLT_MATMUL_TILE_128x128, 128, 128},
#if CUBLAS_VER_MAJOR >= 13
        {CUBLASLT_MATMUL_TILE_64x640,   64, 640},
#endif
        {CUBLASLT_MATMUL_TILE_128x64,  128,  64},
        {CUBLASLT_MATMUL_TILE_64x128,   64, 128}, {CUBLASLT_MATMUL_TILE_64x64,    64,  64},
    };

    // With cuBLAS 13 or newer, the first three rows -- 256x256, 192x256 and
    // 256x192 -- are priced but never probed: AlgoCheck refused every one of
    // them in all 14,708 measured configurations, and the candidate scan below
    // has no early exit. Older cuBLAS headers do not define the wider tiles;
    // every tile present in their reduced table remains eligible for probing.
    constexpr size_t probed_tiles_begin = CUBLAS_VER_MAJOR >= 13 ? 3 : 0;

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

    // Where a candidate kernel comes from. This matters to the selection rule
    // below and nowhere else: tile_traffic and tile_power_watts are a model of
    // cuBLASLt's nvjet tiles, fitted on cuBLASLt's nvjet tiles, and they have
    // nothing to say about an engine from another library.
    enum class MatmulSource { CublasLt, Cudnn };

    struct LtMatmulCandidate
    {
        cublasLtMatmulAlgo_t algorithm{};
        size_t               workspace_bytes = 0;
        float                traffic = numeric_limits<float>::infinity();
        MatmulSource         source = MatmulSource::CublasLt;
        int                  cudnn_candidate = -1;
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
        int tf32;   // the compute type of an fp32 plan follows allow_tf32() at build time

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
                                key.beta_is_zero, key.tf32);
        }
    };

    struct LtMatmulPlan
    {
        LtMatmulPlanKey        key{};
        cublasLtMatmulDesc_t   matmul_descriptor = nullptr;
        cublasLtMatrixLayout_t a_matrix_layout = nullptr;
        cublasLtMatrixLayout_t b_matrix_layout = nullptr;
        cublasLtMatrixLayout_t output_matrix_layout = nullptr;
        cublasLtMatmulAlgo_t   algorithm{};
        bool                   has_algorithm = false;
        size_t                 workspace_bytes = 0;

        // The cuDNN engine set for this same shape, when there is one, and the
        // configuration the tuner chose out of it. These are an OVERLAY on the
        // cuBLASLt fields above, never a replacement: algorithm and
        // workspace_bytes always hold a usable cuBLASLt kernel, so "fall back
        // to cuBLASLt" is a one-line branch at the call site and stays true
        // even if a cuDNN execution fails at run time, years from now, on a
        // driver nobody here has seen.
        cudnn_matmul::Plan*    cudnn_plan = nullptr;
        int                    cudnn_candidate = -1;
        size_t                 cudnn_workspace_bytes = 0;

        vector<LtMatmulCandidate> candidates;
        bool                   tuned = true;

        LtMatmulPlan() = default;
        LtMatmulPlan(const LtMatmulPlan&) = delete;
        LtMatmulPlan& operator=(const LtMatmulPlan&) = delete;
        LtMatmulPlan& operator=(LtMatmulPlan&&) = delete;
        LtMatmulPlan(LtMatmulPlan&& other) noexcept
        {
            swap(key, other.key);
            swap(matmul_descriptor, other.matmul_descriptor);
            swap(a_matrix_layout, other.a_matrix_layout);
            swap(b_matrix_layout, other.b_matrix_layout);
            swap(output_matrix_layout, other.output_matrix_layout);
            swap(algorithm, other.algorithm);
            swap(has_algorithm, other.has_algorithm);
            swap(workspace_bytes, other.workspace_bytes);
            swap(cudnn_plan, other.cudnn_plan);
            swap(cudnn_candidate, other.cudnn_candidate);
            swap(cudnn_workspace_bytes, other.cudnn_workspace_bytes);
            swap(candidates, other.candidates);
            swap(tuned, other.tuned);
        }

        ~LtMatmulPlan()
        {
            cudnn_matmul::destroy(cudnn_plan);
            cublasLtMatrixLayoutDestroy(output_matrix_layout);
            cublasLtMatrixLayoutDestroy(b_matrix_layout);
            cublasLtMatrixLayoutDestroy(a_matrix_layout);
            cublasLtMatmulDescDestroy(matmul_descriptor);
        }

        // Called when the tuner has decided against cuDNN, or could not tune
        // at all. The graph and its built plans are the only thing on this
        // path that holds device memory of its own, and the benchmark reports
        // peak memory, so an unused engine set is released rather than parked.
        void release_cudnn() noexcept
        {
            cudnn_matmul::destroy(cudnn_plan);
            cudnn_plan = nullptr;
            cudnn_candidate = -1;
            cudnn_workspace_bytes = 0;
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

    // autotune_lt_plan picks the fastest of up to eight heuristic candidates
    // by timing each of them, and the timings overlap: on Qwen3-4B's
    // gate projection at decode (9728x1x2560 bf16) the split-K=1 kernels swing
    // 2.3x between processes while the winner moves 2%, so 30% of processes
    // under GPU contention picked a different kernel, and three of the eight
    // candidates round bf16 differently. Same operands, three output bit
    // patterns, and a greedy decode that disagreed with itself across runs.
    //
    // OPENNN_LT_DETERMINISTIC=1 takes the heuristic's first candidate for
    // every shape and never times anything -- 40 of 40 contended processes
    // gave one hash. It costs 26% on that GEMM (0.109 ms against 0.086), so
    // it is for tests and CI gates, not for the numbers that get published.
    bool lt_deterministic_selection()
    {
        static const bool deterministic = env_flag_enabled("OPENNN_LT_DETERMINISTIC", false);
        return deterministic;
    }

    // The default keeps the tuner and makes its verdict outlive the process:
    // the first process to see a shape times the candidates and writes the
    // winner below the temp directory, every later one loads it and skips the
    // timing. Mirrors the cuDNN plan cache in cudnn_frontend_utilities.h --
    // OPENNN_LT_PLAN_CACHE=0 turns it off, OPENNN_LT_PLAN_CACHE_DIR moves it.
    // The directory names the card, its architecture and the cuBLASLt build,
    // because a serialised cublasLtMatmulAlgo_t is only promised to mean the
    // same thing under the library that produced it.
    bool lt_plan_cache_enabled()
    {
        static const bool enabled = env_flag_enabled("OPENNN_LT_PLAN_CACHE", true);
        return enabled;
    }

    const filesystem::path& lt_plan_cache_path()
    {
        static const filesystem::path directory = []
        {
            const char* override_path = getenv("OPENNN_LT_PLAN_CACHE_DIR");

            filesystem::path root;

            if (override_path && *override_path)
                root = filesystem::path(override_path);
            else
            {
                error_code error;
                const filesystem::path temporary = filesystem::temp_directory_path(error);
                if (error) return filesystem::path{};
                root = temporary / "opennn-lt-plans";
            }

            cudaDeviceProp properties{};
            if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess)
            {
                cudaGetLastError();
                return filesystem::path{};
            }

            string card(properties.name);
            for (char& character : card)
                if (!isalnum(static_cast<unsigned char>(character))) character = '-';

            // cuDNN is in the name too: a cached plan may name a cuDNN matmul
            // engine by its position in that library's heuristic list.
            return root / format("{}-sm{}{}-cublaslt{}-cudnn{}", card, properties.major, properties.minor,
                                 cublasLtGetVersion(), CUDNN_VERSION);
        }();

        return directory;
    }

    // One cached plan: the key it was tuned for, every knob that steered the
    // tuner, and the winner. The knobs are stored and compared, not merely
    // hashed into the file name, so a name collision cannot hand a plan tuned
    // under one OPENNN_LT_TILE_TOLERANCE to a process running another.
    struct LtPlanCacheRecord
    {
        uint32_t magic = 0x4c50544fu;   // "OTPL"
        uint32_t version = 3;
        LtMatmulPlanKey key{};

        // Every knob the tuner reads, in the order it reads them: the cuBLASLt
        // candidate set and tie-break, then the cross-source rule, then what
        // cudnn_matmul::create() admits into the candidate list at all.
        long long candidates = 0;
        long long tile_tolerance = 0;
        long long traffic_budget = 0;
        long long cross_source_gain = 0;
        long long anchor_on_fastest = 0;
        long long cudnn_enabled = 0;
        long long cudnn_workspace_mb = 0;
        long long cudnn_min_gflop = 0;
        long long cudnn_min_dim = 0;
        long long cudnn_candidates = 0;
        uint64_t workspace_search_bytes = 0;

        // The winner: always a cuBLASLt kernel, plus the cuDNN engine the
        // tuner preferred over it when it did, as the same overlay the plan
        // carries. The engine is named by its index in cuDNN's own
        // enumeration, which is a stable identity for one cuDNN build on one
        // card -- both are in the directory name -- and nothing else; the
        // position it held in the tuner's candidate list is kept for the
        // record only. Version 3 added the enumeration index: rebuilding the
        // whole set to find the winner again cost a Qwen3-4B session about
        // twelve seconds of warm-up.
        cublasLtMatmulAlgo_t algorithm{};
        uint64_t workspace_bytes = 0;
        int cudnn_candidate = -1;
        uint64_t cudnn_workspace_bytes = 0;
        int64_t cudnn_plan_index = -1;

        bool same_tuning(const LtPlanCacheRecord& other) const noexcept
        {
            return magic == other.magic && version == other.version && key == other.key
                && candidates == other.candidates && tile_tolerance == other.tile_tolerance
                && traffic_budget == other.traffic_budget
                && cross_source_gain == other.cross_source_gain
                && anchor_on_fastest == other.anchor_on_fastest
                && cudnn_enabled == other.cudnn_enabled
                && cudnn_workspace_mb == other.cudnn_workspace_mb
                && cudnn_min_gflop == other.cudnn_min_gflop
                && cudnn_min_dim == other.cudnn_min_dim
                && cudnn_candidates == other.cudnn_candidates
                && workspace_search_bytes == other.workspace_search_bytes;
        }
    };

    static_assert(is_trivially_copyable_v<LtPlanCacheRecord>,
                  "LtPlanCacheRecord is written to disk as raw bytes.");

    LtPlanCacheRecord lt_plan_cache_record(const LtMatmulPlanKey& key)
    {
        LtPlanCacheRecord record;
        record.key = key;
        record.candidates = clamp(env_int_or("OPENNN_LT_AUTOTUNE_CANDIDATES", 8), 1LL, 32LL);
        record.tile_tolerance = clamp(env_int_or("OPENNN_LT_TILE_TOLERANCE", 10), 0LL, 100LL);
        record.traffic_budget = clamp(env_int_or("OPENNN_LT_TRAFFIC_BUDGET", 120), 1LL, 10000LL);
        record.cross_source_gain = clamp(env_int_or("OPENNN_MATMUL_CROSS_SOURCE_GAIN", 2), 0LL, 1000LL);
        record.anchor_on_fastest = env_flag_enabled("OPENNN_MATMUL_CROSS_SOURCE_ANCHOR_FASTEST", false);
        // The same defaults and clamps as cudnn_matmul.cpp applies to them.
        record.cudnn_enabled = env_flag_enabled("OPENNN_CUDNN_MATMUL", true);
        record.cudnn_workspace_mb = clamp(env_int_or("OPENNN_CUDNN_MATMUL_WORKSPACE_MB", 32), 0LL, 4096LL);
        record.cudnn_min_gflop = clamp(env_int_or("OPENNN_CUDNN_MATMUL_MIN_GFLOP", 8), 0LL, 1000000LL);
        record.cudnn_min_dim = clamp(env_int_or("OPENNN_CUDNN_MATMUL_MIN_DIM", 128), 1LL, 1000000LL);
        record.cudnn_candidates = clamp(env_int_or("OPENNN_CUDNN_MATMUL_CANDIDATES", 0), 0LL, 4096LL);
        record.workspace_search_bytes = cublas_lt_workspace_search_bytes;
        return record;
    }

    filesystem::path lt_plan_cache_file(const LtPlanCacheRecord& record)
    {
        // The record version is part of the name, so builds that write
        // different versions keep separate files instead of each rejecting
        // and re-tuning the other's on every launch.
        size_t name = LtMatmulPlanKeyHash{}(record.key);
        for (const long long knob : {static_cast<long long>(record.version),
                                     record.candidates, record.tile_tolerance, record.traffic_budget,
                                     record.cross_source_gain, record.anchor_on_fastest,
                                     record.cudnn_enabled, record.cudnn_workspace_mb,
                                     record.cudnn_min_gflop, record.cudnn_min_dim, record.cudnn_candidates,
                                     static_cast<long long>(record.workspace_search_bytes)})
            name ^= std::hash<long long>{}(knob) + 0x9e3779b9u + (name << 6) + (name >> 2);

        return lt_plan_cache_path() / format("{:016x}.ltplan", name);
    }

    // The plan already has its descriptor and layouts, which is what the
    // library needs to say whether the stored algorithm still applies. A file
    // from another driver, a truncated write, or an algorithm the check
    // refuses all fall through to the tuner rather than into the matmul.
    bool load_cached_lt_plan(LtMatmulPlan& plan)
    {
        if (!lt_plan_cache_enabled() || lt_plan_cache_path().empty()) return false;

        PROFILE_SCOPE_HOST("lt:plan_cache_load");

        const LtPlanCacheRecord expected = lt_plan_cache_record(plan.key);
        const filesystem::path file = lt_plan_cache_file(expected);

        error_code failed;
        if (!filesystem::exists(file, failed) || failed) return false;

        ifstream stream(file, ios::binary);
        LtPlanCacheRecord record;
        if (!stream.read(reinterpret_cast<char*>(&record), streamsize(sizeof(record)))) return false;
        if (!record.same_tuning(expected)) return false;

        cublasLtMatmulHeuristicResult_t check{};
        if (cublasLtMatmulAlgoCheck(Backend::get_cublas_lt_handle(),
                                    plan.matmul_descriptor,
                                    plan.a_matrix_layout,
                                    plan.b_matrix_layout,
                                    plan.output_matrix_layout,
                                    plan.output_matrix_layout,
                                    &record.algorithm, &check) != CUBLAS_STATUS_SUCCESS
            || check.state != CUBLAS_STATUS_SUCCESS
            || check.workspaceSize > cublas_lt_workspace_search_bytes)
        {
            device::reset_last_error();
            return false;
        }

        plan.algorithm = record.algorithm;
        plan.has_algorithm = true;
        plan.workspace_bytes = check.workspaceSize;
        plan.tuned = true;

        // The tuner preferred a cuDNN engine: rebuild that one engine, which
        // becomes the plan's only cuDNN candidate. If cuDNN no longer offers
        // it -- which the directory name says cannot happen, but the check is
        // cheaper than the argument -- the cuBLASLt kernel just loaded serves
        // instead, exactly as run_lt_matmul_cached would fall back at run time.
        if (record.cudnn_candidate >= 0 && record.cudnn_plan_index >= 0)
        {
            cudnn_matmul::Problem problem;
            problem.m = plan.key.m;
            problem.n = plan.key.n;
            problem.k = plan.key.k;
            problem.transA = cublasOperation_t(plan.key.transA);
            problem.transB = cublasOperation_t(plan.key.transB);
            problem.epilogue = cublasLtEpilogue_t(plan.key.epilogue);
            problem.dtype_a = cudaDataType_t(plan.key.dtype_a);
            problem.dtype_b = cudaDataType_t(plan.key.dtype_b);
            problem.out_dtype = cudaDataType_t(plan.key.out_dtype);
            problem.lda = plan.key.lda;
            problem.ldb = plan.key.ldb;
            problem.ldd = plan.key.ldd;
            problem.beta_is_zero = plan.key.beta_is_zero != 0;

            {
                PROFILE_SCOPE_HOST("lt:plan_cache_cudnn_rebuild");
                plan.cudnn_plan = cudnn_matmul::create(problem, record.cudnn_plan_index);
            }
            if (plan.cudnn_plan && cudnn_matmul::candidate_count(plan.cudnn_plan) == 1)
            {
                plan.cudnn_candidate = 0;
                plan.cudnn_workspace_bytes =
                    cudnn_matmul::candidate_workspace_bytes(plan.cudnn_plan, 0);
            }
            else
            {
                plan.release_cudnn();
            }
        }

        return true;
    }

    void store_cached_lt_plan(const LtMatmulPlan& plan)
    {
        if (!lt_plan_cache_enabled() || lt_plan_cache_path().empty()) return;

        LtPlanCacheRecord record = lt_plan_cache_record(plan.key);
        record.algorithm = plan.algorithm;
        record.workspace_bytes = plan.workspace_bytes;
        record.cudnn_candidate = plan.cudnn_candidate;
        record.cudnn_workspace_bytes = plan.cudnn_workspace_bytes;
        record.cudnn_plan_index =
            cudnn_matmul::candidate_plan_index(plan.cudnn_plan, plan.cudnn_candidate);

        error_code failed;
        filesystem::create_directories(lt_plan_cache_path(), failed);
        if (failed) return;

        static atomic<uint64_t> sequence{0};

        const filesystem::path file = lt_plan_cache_file(record);
        const filesystem::path pending = file.string()
            + format(".{:x}-{}.tmp", std::hash<thread::id>{}(this_thread::get_id()), sequence++);

        {
            ofstream stream(pending, ios::binary | ios::trunc);
            if (!stream) return;
            stream.write(reinterpret_cast<const char*>(&record), streamsize(sizeof(record)));
            if (!stream) { filesystem::remove(pending, failed); return; }
        }

        filesystem::rename(pending, file, failed);
        if (failed) filesystem::remove(pending, failed);
    }

    cublasComputeType_t matmul_compute_type(cudaDataType_t a_type, 
                                            cudaDataType_t b_type = CUDA_R_32F)
    {
        if (a_type == CUDA_R_16BF || b_type == CUDA_R_16BF) return CUBLAS_COMPUTE_32F_FAST_16BF;
        return device::allow_tf32() ? CUBLAS_COMPUTE_DTYPE : CUBLAS_COMPUTE_32F;
    }    

    // Only used to size the tuner's scratch destination, so an unrecognised
    // type is rounded up rather than guessed: too large wastes a few bytes of
    // a block that is already megabytes, too small is a write past its end.
    size_t matmul_dtype_bytes(cudaDataType_t type)
    {
        if (type == CUDA_R_8I) return 1;
        if (type == CUDA_R_16F || type == CUDA_R_16BF) return 2;
        return 4;
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

    // cuDNN ships a matmul engine set of its own, and for the dense
    // benchmark's 8192x1024x1024 bf16 hidden layer it contains a kernel
    // cuBLASLt does not expose: 189.6 us at 235 W against cuBLASLt's choice
    // between 201.2 us at 169 W and 192.4 us at 265 W. An exhaustive sweep of
    // all 13,460 cuBLASLt configurations for that shape has nothing
    // comparable. cudnn_matmul::create() declines everything it has no
    // measured reason to serve -- see the guards there -- so on most shapes
    // this is one function call that returns nullptr and costs nothing.
    //
    // The candidates are appended to the SAME list the cuBLASLt algorithms
    // are in, so there is one timing loop and one selection rule rather than
    // two competing ones. They carry infinite traffic, which is not a
    // pessimistic estimate but an admission: a cuDNN engine has no tile id,
    // so tile_traffic cannot price it at all.
    void add_cudnn_candidates(LtMatmulPlan& plan,
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
        cudnn_matmul::Problem problem;
        problem.m = m;
        problem.n = n;
        problem.k = k;
        problem.transA = transA;
        problem.transB = transB;
        problem.epilogue = epilogue;
        problem.dtype_a = dtype_a;
        problem.dtype_b = dtype_b;
        problem.out_dtype = out_dtype;
        problem.lda = lda;
        problem.ldb = ldb;
        problem.ldd = ldd;
        problem.beta_is_zero = beta_is_zero;

        plan.cudnn_plan = cudnn_matmul::create(problem);
        if (!plan.cudnn_plan) return;

        const int count = cudnn_matmul::candidate_count(plan.cudnn_plan);
        for (int candidate = 0; candidate < count; ++candidate)
        {
            LtMatmulCandidate entry;
            entry.workspace_bytes = cudnn_matmul::candidate_workspace_bytes(plan.cudnn_plan, candidate);
            entry.traffic = numeric_limits<float>::infinity();
            entry.source = MatmulSource::Cudnn;
            entry.cudnn_candidate = candidate;
            plan.candidates.push_back(entry);
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
                                  lda, ldb, ldd, int(beta_is_zero),
                                   int(device::allow_tf32())};
        auto& plans = thread_state().lt_matmul_plans;
        auto it = plans.find(key);
        if (it != plans.end()) return it->second;

        throw_if(device::cuda_matmul_plan_creation_forbidden(),
                 "matmul plan forbidden (warmup incomplete).");

        detail::make_bounded_cache_room(plans, cublas_lt_plan_cache_capacity);

        LtMatmulPlan plan;
        plan.key = key;

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

        // A winner another process already timed is taken as it stands, so
        // every process on this card runs the same kernel for this shape and
        // the heuristic query below is not even made.
        if (!lt_deterministic_selection() && load_cached_lt_plan(plan))
            return plans.emplace(key, std::move(plan)).first->second;

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

        // The wide tiles exist to give the tuner something to time; with the
        // tuner off they would only be enumerated to be discarded.
        if (!lt_deterministic_selection())
            add_wide_tile_candidates(plan, heuristics, m, n, dtype_a, dtype_b, out_dtype);

        // The untuned default is the heuristic's first cuBLASLt algorithm,
        // exactly as before. cuDNN candidates are appended after it and are
        // never the default: cuDNN's own heuristic puts a 226 us engine first
        // for the shape this path was added for, where the best configuration
        // runs at 189.6, so an untimed cuDNN pick would be a downgrade. They
        // are only ever selected by autotune_lt_plan, on measured time.
        if (!plan.candidates.empty())
        {
            plan.algorithm = plan.candidates.front().algorithm;
            plan.has_algorithm = true;
            plan.workspace_bytes = plan.candidates.front().workspace_bytes;
        }

        // cuDNN engines, like the wide tiles, are only ever chosen on measured
        // time, so with the tuner off there is nothing for them to be.
        if (!lt_deterministic_selection())
            add_cudnn_candidates(plan, m, n, k, transA, transB, epilogue,
                                 dtype_a, dtype_b, out_dtype, lda, ldb, ldd, beta_is_zero);

        plan.tuned = plan.candidates.size() <= 1 || lt_deterministic_selection();
        if (plan.tuned)
        {
            plan.release_cudnn();
            plan.candidates.clear();
        }

        return plans.emplace(key, std::move(plan)).first->second;
    }

    // --- verifying a candidate from another library -----------------------
    //
    // A cuBLASLt candidate is the same kernel family answering the same
    // descriptor, so the tuner has never had to ask whether a candidate is
    // CORRECT, only which is fastest. A cuDNN candidate is a different
    // library reading the same memory through strides this file derived by
    // hand, and a stride derived wrongly does not fail: it returns a
    // confidently wrong tensor. So every cuDNN candidate is checked against
    // the cuBLASLt answer on the caller's own data before it is timed, and a
    // candidate that disagrees is dropped rather than ranked.
    //
    // The check reads three windows of the destination -- head, middle and
    // tail -- rather than all of it. That is not a sample of a random
    // process: an operand read through a wrong stride, or a bias broadcast
    // along the wrong axis, is wrong in nearly every element, so a few
    // thousand elements settle it, while the windows at the two ends are
    // where a tail-handling bug would hide. It costs three 8 KB copies per
    // candidate and no device memory at all, because it reuses the
    // destination the tuner is already writing into -- run_lt_matmul_cached
    // overwrites it with the real call the moment the tuner returns.
    constexpr size_t verification_window_elements = 4096;

    float bf16_bits_to_float(uint16_t bits)
    {
        const uint32_t widened = uint32_t(bits) << 16;
        float value = 0.0f;
        memcpy(&value, &widened, sizeof(value));
        return value;
    }

    bool read_verification_windows(const void* source, size_t elements,
                                   vector<uint16_t>& windows, cudaStream_t stream)
    {
        if (elements == 0) return false;

        const size_t window = min(elements, verification_window_elements);
        const size_t offsets[3] = {0, (elements - window) / 2, elements - window};

        windows.resize(window * 3);

        for (size_t index = 0; index < 3; ++index)
            if (cudaMemcpyAsync(windows.data() + index * window,
                                static_cast<const uint16_t*>(source) + offsets[index],
                                window * sizeof(uint16_t),
                                cudaMemcpyDeviceToHost, stream) != cudaSuccess)
            {
                device::reset_last_error();
                return false;
            }

        if (cudaStreamSynchronize(stream) != cudaSuccess)
        {
            device::reset_last_error();
            return false;
        }

        return true;
    }

    bool verification_windows_agree(const vector<uint16_t>& reference,
                                    const vector<uint16_t>& candidate)
    {
        if (reference.empty() || reference.size() != candidate.size()) return false;

        double sum_of_squares = 0.0;
        for (const uint16_t bits : reference)
        {
            const double value = bf16_bits_to_float(bits);
            sum_of_squares += value * value;
        }
        const double root_mean_square = sqrt(sum_of_squares / double(reference.size()));

        // bf16 carries about three decimal digits, so 2e-2 is the gate the
        // probe used against the same reference. Dividing by the tensor's own
        // RMS rather than by each element keeps a sample that landed on a
        // ReLU zero from manufacturing a failure.
        const double floor_value = max(1e-6, 0.01 * root_mean_square);

        for (size_t index = 0; index < reference.size(); ++index)
        {
            const double want = bf16_bits_to_float(reference[index]);
            const double got  = bf16_bits_to_float(candidate[index]);
            if (fabs(want - got) / max(floor_value, fabs(want)) > 2e-2) return false;
        }

        return true;
    }

    void autotune_lt_plan(LtMatmulPlan& plan,
                          const void* a_data, const void* b_data,
                          const void* c_data, void* d_data,
                          const void* bias_data,
                          float alpha, float beta, size_t destination_bytes,
                          cudaStream_t stream)
    {
        PROFILE_SCOPE_HOST("lt:autotune");

        if (plan.candidates.size() <= 1)
        {
            plan.release_cudnn();
            plan.candidates.clear();
            plan.tuned = true;
            return;
        }

        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &capture_status) != cudaSuccess
            || capture_status != cudaStreamCaptureStatusNone)
            return device::reset_last_error();

        plan.tuned = true;

        // alpha is deliberately absent from LtMatmulPlanKey: it is a call-time
        // scalar that changes no cuBLASLt kernel. The cuDNN graph has no alpha
        // node at all, so it can only ever serve alpha == 1, and a plan tuned
        // under one alpha may be reached later under another. Drop the cuDNN
        // candidates here rather than time them against a reference they
        // cannot reproduce; run_lt_matmul_cached repeats the same test on
        // every call, which is where the guarantee actually lives.
        if (plan.cudnn_plan && alpha != 1.0f)
        {
            plan.release_cudnn();
            erase_if(plan.candidates, [](const LtMatmulCandidate& candidate)
                     { return candidate.source == MatmulSource::Cudnn; });

            if (plan.candidates.size() <= 1)
            {
                plan.candidates.clear();
                return;
            }
        }

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
            // No scratch, so nothing can be timed, so nothing can be verified
            // either: the cuDNN engine set goes with the candidate list.
            plan.release_cudnn();
            plan.candidates.clear();
            return;
        }

        void* const destination = destination_is_operand
            ? static_cast<char*>(workspace) + destination_offset
            : d_data;

        const device::CudaEvent start(cudaEventDefault), stop(cudaEventDefault);
        // Seven, not three. The custom-option sweep can emit a dozen
        // candidates that share a tile and differ by less than the noise of
        // three launches, and the pick below then varies between processes:
        // measured, cuda-dense-infer moved over a 1.9% range run to run where
        // the published build's spread was 0.04%. A benchmark that reports a
        // different kernel each run is not reporting the library.
        constexpr int timed_runs = 7;

        vector<float> times(plan.candidates.size(), numeric_limits<float>::infinity());

        // Two anchors, because the selection below happens in two stages and
        // the stages are ordered by the quality of the evidence behind them.
        // best_lt is the fastest cuBLASLt candidate and is the ONLY anchor the
        // traffic and modelled-energy rule ever sees; best_any is the fastest
        // candidate whatever its source.
        //
        // Keeping them apart is what makes stage 1 safe: a cuDNN candidate
        // cannot move the anchor the cuBLASLt rule is written against, so
        // stage 1 selects exactly the kernel it selected before this file had
        // heard of cuDNN, on every shape.
        //
        // It is NOT, on its own, what keeps the dense-train energy win. Stage
        // 2 below can still override stage 1's pick, and stage 2 has no energy
        // term, so on any shape where stage 1 traded time for modelled energy
        // AND cuDNN qualifies, the trade can be handed back for a 2% gain.
        // Two of cuda-dense-train's three GEMMs do qualify -- the input-delta
        // product (m=inputs, n=rows, k=neurons, OP_T/OP_N, DEFAULT epilogue,
        // bf16 in and out, beta 0) and the bf16-store weight gradient
        // (DEFAULT epilogue, bf16 out) -- and the input-delta product is
        // precisely the shape whose comment in tensor_operations.cpp records
        // the tie-break buying 200.4 us at 172 W over 193.1 us at 271 W. So
        // the 1.339x figure is at risk here and has to be re-measured, not
        // assumed; OPENNN_MATMUL_CROSS_SOURCE_ANCHOR_FASTEST=1 below is the
        // A/B for it.
        size_t best_lt = plan.candidates.size();
        float  best_lt_ms = numeric_limits<float>::infinity();
        size_t best_any = plan.candidates.size();
        float  best_any_ms = numeric_limits<float>::infinity();

        const auto launch_lt = [&](size_t index)
        {
            const LtMatmulCandidate& candidate = plan.candidates[index];
            return cublasLtMatmul(Backend::get_cublas_lt_handle(), plan.matmul_descriptor,
                                  &alpha, a_data, plan.a_matrix_layout, b_data, plan.b_matrix_layout,
                                  &beta, c_data, plan.output_matrix_layout,
                                  destination, plan.output_matrix_layout,
                                  &candidate.algorithm, workspace, candidate.workspace_bytes, stream);
        };

        for (size_t index = 0; index < plan.candidates.size(); ++index)
        {
            if (plan.candidates[index].source != MatmulSource::CublasLt) continue;

            if (launch_lt(index) != CUBLAS_STATUS_SUCCESS) { device::reset_last_error(); continue; }
            device::record_event(start.get(), stream);
            bool ok = true;
            for (int i = 0; i < timed_runs && ok; ++i) ok = launch_lt(index) == CUBLAS_STATUS_SUCCESS;
            device::record_event(stop.get(), stream);
            device::synchronize_event(stop.get());
            if (!ok) { device::reset_last_error(); continue; }
            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start.get(), stop.get()));
            times[index] = ms;
            if (ms < best_lt_ms) { best_lt_ms = ms; best_lt = index; }
        }

        best_any = best_lt;
        best_any_ms = best_lt_ms;

        // --- the cuDNN candidates: verified against cuBLASLt, then timed ----
        //
        // The reference is whichever cuBLASLt kernel proved fastest, run once
        // into the same destination and read back. If there is no cuBLASLt
        // candidate that ran, there is no reference, and a cuDNN candidate is
        // dropped unmeasured: this path exists to beat cuBLASLt on a shape
        // cuBLASLt already serves, so "cuBLASLt could not run it" is not the
        // case it was built for, and taking an unchecked answer there would
        // trade a benchmark cell for the possibility of a silent wrong result.
        if (plan.cudnn_plan)
        {
            const size_t destination_elements = destination_bytes / sizeof(uint16_t);

            vector<uint16_t> reference_windows;
            vector<uint16_t> candidate_windows;

            const bool comparable =
                best_lt < plan.candidates.size()
                && !destination_is_operand
                && destination_elements > 0
                && launch_lt(best_lt) == CUBLAS_STATUS_SUCCESS
                && read_verification_windows(d_data, destination_elements, reference_windows, stream);

            if (!comparable) device::reset_last_error();

            for (size_t index = 0; comparable && index < plan.candidates.size(); ++index)
            {
                const LtMatmulCandidate& candidate = plan.candidates[index];
                if (candidate.source != MatmulSource::Cudnn) continue;

                const auto launch_cudnn = [&]
                {
                    return cudnn_matmul::run(plan.cudnn_plan, candidate.cudnn_candidate,
                                             a_data, b_data, bias_data, d_data, workspace);
                };

                if (!launch_cudnn()) continue;
                if (cudaStreamSynchronize(stream) != cudaSuccess) { device::reset_last_error(); continue; }

                if (!read_verification_windows(d_data, destination_elements, candidate_windows, stream))
                    continue;

                if (!verification_windows_agree(reference_windows, candidate_windows))
                {
                    // Loud, once per candidate, because this should never
                    // happen: it means the strides cudnn_matmul::create()
                    // derived do not describe the memory cuBLASLt was handed.
                    // The candidate is dropped either way, so the library is
                    // still correct -- but a reader who sees this line has a
                    // layout bug to find, not a slow kernel.
                    logging::warning() << "cudnn matmul: candidate "
                         << cudnn_matmul::candidate_name(plan.cudnn_plan, candidate.cudnn_candidate)
                         << " disagreed with cuBLASLt and was dropped.\n";
                    continue;
                }

                device::record_event(start.get(), stream);
                bool ok = true;
                for (int i = 0; i < timed_runs && ok; ++i) ok = launch_cudnn();
                device::record_event(stop.get(), stream);
                device::synchronize_event(stop.get());
                if (!ok) { device::reset_last_error(); continue; }

                float ms = 0.0f;
                CHECK_CUDA(cudaEventElapsedTime(&ms, start.get(), stop.get()));
                times[index] = ms;
                if (ms < best_any_ms) { best_any_ms = ms; best_any = index; }
            }
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
        // conditions now, and only the last one mentions best_lt_ms.
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
        // restores the pick by time alone -- now across BOTH sources, which is
        // what that A/B has always meant: whatever measured fastest, run it.
        const float tolerance =
            float(clamp(env_int_or("OPENNN_LT_TILE_TOLERANCE", 10), 0LL, 100LL)) / 100.0f;
        const float traffic_budget =
            float(clamp(env_int_or("OPENNN_LT_TRAFFIC_BUDGET", 120), 1LL, 10000LL)) / 10000.0f;

        // ------------------------------------------------------------------
        // STAGE 1 -- inside cuBLASLt, unchanged.
        //
        // Exactly the rule above, anchored on best_lt and looking only at
        // cuBLASLt candidates. A cuDNN candidate has infinite traffic, so the
        // `traffic > traffic_budget` test would have excluded it anyway; it is
        // skipped explicitly so that the reason is stated rather than
        // implied. The reason matters: it is not that a cuDNN engine moves too
        // much data. It is that nobody knows how much data it moves.
        // tile_traffic reads a cuBLASLt tile id and tile_power_watts is a
        // two-point fit over cuBLASLt nvjet tiles; asking either of them about
        // a cuDNN engine does not produce a conservative estimate, it produces
        // a fabricated measurement -- and this rule then uses that fabrication
        // to overrule a real one. The measured cuDNN engine draws 235 W. The
        // model would price it at 265 W and reject it on that basis.
        // ------------------------------------------------------------------
        size_t best = best_lt;

        if (tolerance > 0.0f && best_lt < plan.candidates.size())
        {
            const float allowed_ms = best_lt_ms * (1.0f + tolerance);
            const float allowed_energy = best_lt_ms * tile_power_watts(plan.candidates[best_lt].traffic);
            float least_traffic = plan.candidates[best_lt].traffic;

            for (size_t index = 0; index < plan.candidates.size(); ++index)
            {
                const LtMatmulCandidate& candidate = plan.candidates[index];

                if (candidate.source != MatmulSource::CublasLt) continue;
                if (candidate.traffic > traffic_budget) continue;
                if (candidate.traffic > least_traffic) continue;
                // Same tile, so same traffic: prefer the faster option,
                // but only when it is faster by more than the measurement can
                // resolve. The custom-option sweep above emits several
                // candidates per tile precisely because the first valid option
                // is up to 55% slower than the best one, so selecting the
                // lowest-traffic tile is not enough -- without this the pick
                // would be the first option in scan order, which is the one
                // the sweep was added to stop using.
                // Without the margin this comparison is decided by noise and
                // the chosen kernel stops being a property of the shape.
                // Candidates are enumerated in a fixed order, so ties keep the
                // first and the pick is reproducible across processes.
                constexpr float resolvable = 0.02f;
                if (candidate.traffic == least_traffic
                    && times[index] >= times[best] * (1.0f - resolvable)) continue;
                if (times[index] > allowed_ms) continue;
                if (times[index] * tile_power_watts(candidate.traffic) >= allowed_energy) continue;

                least_traffic = candidate.traffic;
                best = index;
            }
        }

        // The cuBLASLt kernel the library would run if cuDNN did not exist.
        // It is stored unconditionally, so plan.algorithm is always a valid
        // fallback no matter what stage 2 decides or what happens at run time.
        if (best < plan.candidates.size())
        {
            plan.algorithm = plan.candidates[best].algorithm;
            plan.workspace_bytes = plan.candidates[best].workspace_bytes;
            plan.has_algorithm = true;
        }

        // ------------------------------------------------------------------
        // STAGE 2 -- between kernel libraries, on measured time alone.
        //
        // THE PROBLEM THIS SOLVES, stated plainly, because it is the hard part
        // of the change and it must not be papered over. On the dense
        // inference GEMM the cuDNN engine is FASTER than the cuBLASLt kernel
        // stage 1 chose -- 189.6 us against 201.2 -- and uses MORE energy per
        // GEMM -- 44.6 mJ against 34.0. Feeding it into stage 1 as one more
        // candidate therefore changes nothing: the rule prefers the lean tile,
        // the cell keeps losing throughput, and the whole cuDNN path is dead
        // code. Something has to decide, explicitly, what six percent of
        // throughput is worth.
        //
        // WHAT THIS DOES NOT PRETEND TO DO. It does not price the trade. It
        // cannot: pricing it needs the candidate's energy, and energy is not
        // measurable here. NVML's board power is a one-second average and its
        // sample ring needs about a second of window with the ends discarded;
        // the probe spent 1.5 s per configuration to get a watt figure it
        // could stand behind. Sixty configurations at 1.5 s is ninety seconds
        // per shape, inside a plan cache that fills at warmup. A library
        // cannot spend that, and a modelled substitute is exactly the
        // fabrication stage 1's comment refuses.
        //
        // WHAT IT DOES INSTEAD. It asks the one question the tuner has real
        // evidence for -- is the difference in time REAL? -- and defers to the
        // kernel it understands whenever the answer is no.
        //
        //   * A cuDNN candidate is taken only when it beats the cuBLASLt
        //     choice by more than OPENNN_MATMUL_CROSS_SOURCE_GAIN percent,
        //     default 2. Two is not a trade rate, it is a noise floor: the
        //     comment on timed_runs above records that near-tied candidates
        //     moved this cell over a 1.9% range run to run. Below that margin
        //     the two kernels are the same speed and the library keeps the
        //     cuBLASLt one -- the one whose energy behaviour is modelled, whose
        //     tile is known, and whose selection has measured evidence behind
        //     it. Above it the difference is real and the library has no basis
        //     on which to spend it.
        //   * Nothing else. In particular, no modelled energy term: see above.
        //
        // WHAT THE MARGIN IS, AND WHAT IT IS NOT. The margin never mentions
        // this shape, this card or this margin of victory: it is a statement
        // about evidence, and any value under the 6.1% this cell offers gives
        // the same answer here. But the margin is not the load-bearing part of
        // the rule -- the ANCHOR is, and it is worth being blunt about that
        // rather than letting the 2% carry a claim it does not support.
        //
        // The anchor is times[best]: the kernel stage 1 chose. Stage 1 is
        // allowed to spend up to OPENNN_LT_TILE_TOLERANCE (10%) of time to buy
        // modelled energy, so measuring the cross-source gain against its pick
        // means a cuDNN engine has to beat a deliberately slowed-down kernel by
        // 2%, not the fastest cuBLASLt kernel by 2%. On the dense inference
        // GEMM that is the difference between a 6.1% margin (against the lean
        // 201.2 us tile stage 1 picked) and a 1.5% one (against cuBLASLt's
        // fastest at 192.4 us) -- and 1.5% is below the noise floor the margin
        // is set to, so the anchor, not the margin, is why cuDNN is taken here.
        //
        // That anchor is defensible -- "is there a measurably faster kernel
        // than the one we would actually run" is the question a caller cares
        // about -- but its consequence must be stated: where stage 1 traded
        // time for energy, stage 2 can hand the trade back without pricing it.
        // OPENNN_MATMUL_CROSS_SOURCE_ANCHOR_FASTEST=1 anchors on the fastest
        // cuBLASLt candidate instead, so cuDNN must beat ANY cuBLASLt kernel
        // by the margin; that is the stricter reading of "cuDNN reaches a
        // kernel cuBLASLt does not expose", and it is the A/B that says what
        // the loose anchor costs in energy on the training cells.
        //
        // WHAT IT COSTS, HONESTLY. On a shape where a cuDNN engine is, say, 4%
        // faster and 40% hotter, this takes the throughput and loses the
        // energy, and nothing in the library will notice. That gap is real and
        // it closes only with a per-candidate energy measurement the tuner
        // cannot afford. Until then it is a visible default with three knobs
        // rather than a decision buried in a tie-break:
        // OPENNN_MATMUL_CROSS_SOURCE_GAIN raises the bar,
        // OPENNN_MATMUL_CROSS_SOURCE_ANCHOR_FASTEST moves what the bar is
        // measured against, and OPENNN_CUDNN_MATMUL=0 removes cuDNN from the
        // candidate set outright and restores the previous behaviour exactly.
        // ------------------------------------------------------------------
        const float cross_source_gain =
            float(clamp(env_int_or("OPENNN_MATMUL_CROSS_SOURCE_GAIN", 2), 0LL, 1000LL)) / 100.0f;

        const bool anchor_on_fastest_lt =
            env_flag_enabled("OPENNN_MATMUL_CROSS_SOURCE_ANCHOR_FASTEST", false);

        size_t chosen = best;

        if (tolerance <= 0.0f)
        {
            // The A/B: pick by time alone, across every source.
            chosen = best_any;
        }
        else if (best_any < plan.candidates.size()
                 && plan.candidates[best_any].source == MatmulSource::Cudnn
                 && best < plan.candidates.size()
                 && best_any_ms * (1.0f + cross_source_gain)
                        < (anchor_on_fastest_lt ? best_lt_ms : times[best]))
        {
            chosen = best_any;
        }

        if (chosen < plan.candidates.size()
            && plan.candidates[chosen].source == MatmulSource::Cudnn)
        {
            plan.cudnn_candidate = plan.candidates[chosen].cudnn_candidate;
            plan.cudnn_workspace_bytes = plan.candidates[chosen].workspace_bytes;

            if (cudnn_matmul::verbose())
                logging::warning() << "cudnn matmul: chose "
                     << cudnn_matmul::candidate_name(plan.cudnn_plan, plan.cudnn_candidate)
                     << " at " << times[chosen] * 1000.0f / timed_runs << " us against cuBLASLt's "
                     << (best < plan.candidates.size() ? times[best] * 1000.0f / timed_runs : 0.0f)
                     << " us\n";
        }
        else
        {
            plan.release_cudnn();
        }

        plan.candidates.clear();

        // Only a verdict the timer actually reached is worth keeping: with
        // every cuBLASLt candidate failing to run there was nothing to choose
        // between, and the plan holds the untimed heuristic front.
        if (best_lt_ms < numeric_limits<float>::infinity())
            store_cached_lt_plan(plan);
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

string device::lt_plan_cache_directory() noexcept
{
    try
    {
        if (!lt_plan_cache_enabled()) return {};
        return lt_plan_cache_path().string();
    }
    catch (const exception&)
    {
        return {};
    }
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
        autotune_lt_plan(plan, a_data, b_data, c_data, d_data, bias_pointer, alpha, beta,
                         size_t(ldd ? ldd : m) * size_t(n) * matmul_dtype_bytes(out_dtype),
                         device::get_compute_stream());

    // The cuDNN overlay, when the tuner chose one. Three things are checked
    // here rather than trusted from the plan, because none of them is part of
    // LtMatmulPlanKey and so none of them is guaranteed to match the call that
    // tuned this plan: the cuDNN graph has no alpha node, no beta node and no
    // second output, so alpha must be 1, the epilogue must not be writing an
    // auxiliary tensor, and beta is already pinned by the key. A plan reached
    // with any other combination silently falls through to cuBLASLt below,
    // which is the same kernel it would have run yesterday.
    //
    // cudnn_matmul::run returns false instead of throwing, so an engine that
    // stops working -- a driver change, a workspace that could not be raised
    // -- costs one wasted launch attempt and then the cuBLASLt kernel, not an
    // exception in the middle of a training step.
    if (plan.cudnn_plan && plan.cudnn_candidate >= 0
        && alpha == 1.0f && beta == 0.0f && !aux_pointer)
    {
        void* cudnn_workspace = nullptr;
        bool have_workspace = true;
        try
        {
            cudnn_workspace = ensure_shared_scratch(plan.cudnn_workspace_bytes);
        }
        catch (const exception&)
        {
            have_workspace = false;
        }

        if (have_workspace
            && cudnn_matmul::run(plan.cudnn_plan, plan.cudnn_candidate,
                                 a_data, b_data, bias_pointer, d_data, cudnn_workspace))
            return;

        // It declined once; it will decline again. Give the shape back to
        // cuBLASLt for the life of the plan rather than pay the failed launch
        // on every forward pass.
        plan.release_cudnn();
    }

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

string device::lt_plan_cache_directory() noexcept { return {}; }

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
