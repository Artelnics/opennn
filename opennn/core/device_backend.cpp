// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#include "opennn/core/device_backend.h"
#include "opennn/core/string_utilities.h"

#include <atomic>

namespace opennn::device
{

namespace
{

atomic_bool cuda_allocation_growth_forbidden_runtime{false};
atomic_bool cuda_matmul_plan_creation_forbidden_runtime{false};

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

}

#ifdef OPENNN_HAS_CUDA
bool cuda_matmul_plan_creation_forbidden() noexcept
{
    return cuda_matmul_plan_creation_forbidden_runtime.load(memory_order_relaxed);
}
#endif

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
    refresh_blas_math_mode();
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

#endif

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

void set_zero_async(void* data, Index byte_count, DeviceStream stream)
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
                DeviceStream stream)
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
    const DeviceStream copy_stream = stream ? stream : get_compute_stream();
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
                DeviceStream stream)
{
    throw_if_auto(source_device);
    throw_if_auto(target_device);

    CopyKind kind = CopyKind::HostToHost;
    if (source_device == Device::CUDA && target_device == Device::CUDA) kind = CopyKind::DeviceToDevice;
    else if (source_device == Device::CUDA)                             kind = CopyKind::DeviceToHost;
    else if (target_device == Device::CUDA)                             kind = CopyKind::HostToDevice;

    copy_async(destination, source, byte_count, kind, stream);
}

void synchronize(DeviceStream stream)
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

}
