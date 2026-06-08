//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E V I C E   B A C K E N D
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"
#include "types.h"

namespace opennn::device
{

enum class CopyKind
{
    HostToHost,
    HostToDevice,
    DeviceToHost,
    DeviceToDevice
};

bool is_cuda_build() noexcept;
bool has_cuda_device() noexcept;
int cuda_compute_capability() noexcept;
pair<size_t, size_t> memory_info();
bool cuda_allocation_growth_forbidden() noexcept;
void set_cuda_allocation_growth_forbidden(bool) noexcept;

void* allocate(Device, Index);
void deallocate(Device, void*, Index);

void set_zero(void*, Index, Device);
void set_zero_async(void*, Index, cudaStream_t = nullptr);

CopyKind copy_kind(Device, Device);
void copy_async(void*, const void*, Index, CopyKind, cudaStream_t = nullptr);
void synchronize(cudaStream_t = nullptr);
void check_last_error();

#ifdef OPENNN_HAS_CUDA
struct CublasPointerModeGuard
{
    cublasHandle_t handle = nullptr;
    cublasPointerMode_t previous_mode = CUBLAS_POINTER_MODE_HOST;

    CublasPointerModeGuard(cublasHandle_t new_handle, cublasPointerMode_t mode)
        : handle(new_handle)
    {
        CHECK_CUBLAS(cublasGetPointerMode(handle, &previous_mode));
        CHECK_CUBLAS(cublasSetPointerMode(handle, mode));
    }

    CublasPointerModeGuard(const CublasPointerModeGuard&) = delete;
    CublasPointerModeGuard& operator=(const CublasPointerModeGuard&) = delete;

    ~CublasPointerModeGuard() noexcept
    {
        if (handle) cublasSetPointerMode(handle, previous_mode);
    }
};
#endif

cudaStream_t create_stream(unsigned);
void destroy_stream(cudaStream_t);

void* allocate_pinned_host(Index);
void deallocate_pinned_host(void*);

cudaEvent_t create_event(unsigned);
cudaEvent_t create_event();
void destroy_event(cudaEvent_t);
void record_event(cudaEvent_t, cudaStream_t);
void synchronize_event(cudaEvent_t);
void stream_wait_event(cudaStream_t, cudaEvent_t);

// CUDA graph capture/replay. begin_capture starts recording stream work; end_capture
// instantiates an executable graph (returned as an opaque handle); launch_graph
// replays it; destroy_graph frees it. Handles are void* so this header stays usable
// from non-CUDA translation units.
void  begin_graph_capture(cudaStream_t);
void* end_graph_capture(cudaStream_t);   // returns cudaGraphExec_t as void*
void  launch_graph(void* graph_exec, cudaStream_t);
void  destroy_graph(void* graph_exec);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
