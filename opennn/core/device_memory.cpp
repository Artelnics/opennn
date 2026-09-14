// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#include "opennn/core/device_backend.h"
#include "opennn/core/profiler.h"
#include "opennn/core/string_utilities.h"

#include <atomic>
#include <mutex>

namespace opennn::device
{

namespace
{

atomic_bool cuda_resources_shutting_down{false};
thread_local bool cuda_block_cache_bypassed = false;

void throw_if_auto(const Device device)
{
    throw_if(device == Device::Auto,
             "device backend expects a resolved device.");
}

#ifndef OPENNN_HAS_CUDA
[[noreturn]] void throw_cuda_unavailable()
{
    throw runtime_error("CUDA support is not compiled in.");
}
#endif

#ifdef OPENNN_HAS_CUDA
static int device_poison_mode()
{
    static const int mode = int(env_int_or("OPENNN_DEVICE_POISON", 0));
    return mode;
}

static int device_poison_byte()
{
    return device_poison_mode() == 2 ? 0x00 : 0xFF;
}

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

    bool record_pending(CachedBlock& block, DeviceStream stream) noexcept
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

void begin_cuda_shutdown() noexcept
{
    cuda_resources_shutting_down.store(true, memory_order_relaxed);
}

}
