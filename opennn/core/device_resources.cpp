// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#include "opennn/core/device_backend.h"
#include "opennn/core/log.h"
#include "opennn/core/string_utilities.h"

#include <cstdlib>
#include <memory>
#include <utility>

namespace opennn::device
{

namespace
{

#ifndef OPENNN_HAS_CUDA
[[noreturn]] void throw_cuda_unavailable()
{
    throw runtime_error("CUDA support is not compiled in.");
}
#endif

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
            logging::warning() << "CUDA graph captured: " << nodes << " nodes" << endl;
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


}
