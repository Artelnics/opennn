//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C U D N N   F R O N T E N D   U T I L I T I E S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#ifdef OPENNN_HAS_CUDA

#include <cudnn_frontend.h>

#include "opennn/core/tensor_types.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/string_utilities.h"

namespace opennn::cudnn_frontend
{
using namespace ::cudnn_frontend;

inline const auto check_status = [](auto status, const string& what) {
    throw_if(status.is_bad(),
             "cudnn-frontend {}: {}", what, status.get_message());
};

inline int device_sm_version()
{
    static const int sm = [] {
        int device = 0;
        cudaGetDevice(&device);
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        return major * 100 + minor * 10;
    }();
    return sm;
}

inline bool frontend_enabled()
{
    return device_sm_version() >= 700;
}

inline bool bn_frontend_enabled()
{
    return frontend_enabled() && device_sm_version() >= 800;
}

inline bool graph_timing_enabled()
{
    static const bool enabled = env_flag_enabled("OPENNN_GRAPH_TIMING");
    return enabled;
}

inline map<string, pair<double, long>>& graph_times()
{
    static map<string, pair<double, long>> times;
    static const bool registered = [] {
        atexit(+[] {
            double total = 0;
            for (const auto& [label, accumulated] : graph_times()) total += accumulated.first;
            cerr << format("[GRAPH_TIMING] total_gpu_ms={:.1f}\n", total);
            for (const auto& [label, accumulated] : graph_times())
                cerr << format("[GRAPH_TIMING] {:<40} total_ms={:>9.1f} calls={:>6} ms/call={:.4f}\n",
                               label, accumulated.first, accumulated.second,
                               accumulated.first / accumulated.second);
        });
        return true;
    }();
    (void)registered;
    return times;
}

template<typename TensorMap>
inline void execute_graph(graph::Graph& graph, 
                          TensorMap& tensors,
                          void* workspace, 
                          const string& what, 
                          const string& timing_label)
{
    if (timing_label.empty())
    {
        check_status(graph.execute(Backend::get_cudnn_handle(), tensors, workspace), what);
        return;
    }

    CudaEvent begin(cudaEventDefault);
    CudaEvent end(cudaEventDefault);
    device::record_event(begin, Backend::get_compute_stream());

    check_status(graph.execute(Backend::get_cudnn_handle(), tensors, workspace), what);

    device::record_event(end, Backend::get_compute_stream());
    device::synchronize_event(end);

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, begin, end));

    auto& [total, calls] = graph_times()[timing_label];
    total += milliseconds;
    ++calls;
}

inline void* shared_workspace(int64_t bytes)
{
    return bytes > 0 ? ensure_shared_scratch(size_t(bytes)) : nullptr;
}

template<typename GraphCache, typename Body>
bool run_frontend(unique_ptr<GraphCache>& cache, const char* label, Body&& body)
{
    if (!cache) cache = make_unique<GraphCache>();
    if (cache->disabled) return false;

    try
    {
        body(*cache);
        return true;
    }
    catch (const exception& e)
    {
        cache->disabled = true;
        cerr << label << ": cudnn-frontend path unavailable (" << e.what() << ").\n";
        return false;
    }
}

// The frontend path can be unavailable for two unrelated reasons: the GPU really
// lacks the required compute capability, or a plan, workspace or autotune
// allocation failed at runtime. Reporting the former when it is the latter sends
// the reader after a hardware problem that does not exist.
[[noreturn]] inline void throw_frontend_unavailable(const string& what)
{
    if (!frontend_enabled())
        throw runtime_error(what + " requires a GPU of compute capability 7.0 or higher.");

    throw runtime_error(
        what + ": no usable cuDNN plan for this shape. The cudnn-frontend message above "
        "names the cause; under memory pressure it is normally a failed workspace or "
        "autotune allocation. Reduce the batch size, or cap the convolution workspace "
        "with device::set_conv_workspace_cap().");
}

inline DataType_t to_dtype(Type t)
{
    switch (t)
    {
        case Type::FP32: return DataType_t::FLOAT;
        case Type::BF16: return DataType_t::BFLOAT16;
        default:         return DataType_t::FLOAT;
    }
}

inline vector<int64_t> nhwc_strides(int64_t c, int64_t h, int64_t w)
{
    return {h * w * c, 1, w * c, c};
}

inline shared_ptr<graph::Graph> new_graph(Type dtype = Type::FP32)
{
    auto g = make_shared<graph::Graph>();
    g->set_io_data_type(to_dtype(dtype))
      .set_intermediate_data_type(DataType_t::FLOAT)
      .set_compute_data_type(DataType_t::FLOAT);
    return g;
}

inline shared_ptr<graph::Tensor_attributes>
nhwc_tensor(graph::Graph& graph, const char* name,
            int64_t n, int64_t c, int64_t h, int64_t w)
{
    return graph.tensor(graph::Tensor_attributes()
                        .set_name(name)
                        .set_dim({n, c, h, w})
                        .set_stride(nhwc_strides(c, h, w)));
}

inline void set_nhwc_output(shared_ptr<graph::Tensor_attributes>& tensor,
                     int64_t n, int64_t c, int64_t h, int64_t w)
{
    tensor->set_output(true)
           .set_dim({n, c, h, w})
           .set_stride(nhwc_strides(c, h, w));
}

inline void finalize_attention(graph::Graph& graph, const string& tag)
{
    const cudnnHandle_t handle = Backend::get_cudnn_handle();

    check_status(graph.validate(), tag + " validate");
    check_status(graph.build_operation_graph(handle), tag + " build_operation_graph");
    check_status(graph.create_execution_plans({HeurMode_t::A}), tag + " create_execution_plans");
    check_status(graph.build_plans(handle, BuildPlanPolicy_t::HEURISTICS_CHOICE), tag + " build_plans");
}

inline shared_ptr<graph::Tensor_attributes>
seq_len_scalar(graph::Graph& graph, const char* name, int64_t batch = 1)
{
    return graph.tensor(graph::Tensor_attributes()
                        .set_name(name)
                        .set_dim({batch, 1, 1, 1})
                        .set_stride({1, 1, 1, 1})
                        .set_data_type(DataType_t::INT32));
}

inline bool finalize(graph::Graph& graph, int64_t& workspace_bytes, const string& tag,
                     bool request_autotune = false)
{
    const cudnnHandle_t handle = Backend::get_cudnn_handle();

    workspace_bytes = 0;

    check_status(graph.validate(), tag + " validate");
    check_status(graph.build_operation_graph(handle), tag + " build_operation_graph");
    check_status(graph.create_execution_plans({HeurMode_t::A, HeurMode_t::FALLBACK}),
                 tag + " create_execution_plans");

    const int64_t conv_workspace_cap = device::conv_workspace_limit_bytes();
    if (conv_workspace_cap > 0)
        graph.deselect_workspace_greater_than(conv_workspace_cap);

    // Autotuning requires an uncapped workspace. Combining
    // deselect_workspace_greater_than with BuildPlanPolicy_t::ALL crashes with an
    // access violation in this cudnn-frontend version, but only once the cap
    // actually removes plans: measured on sm_120, ResNet-50 batch 512, a 512 MiB
    // cap faults while 4 GiB and 1 TiB caps (which filter nothing) run clean.
    // Tuned-plans-within-a-budget therefore needs a different mechanism — tune
    // unbounded, then rebuild over budget — not simply dropping this condition.
    const bool autotune = request_autotune && conv_workspace_cap == 0
        && graph.build_plans(handle, BuildPlanPolicy_t::ALL).is_good();

    if (autotune) return true;

    check_status(graph.build_plans(handle, BuildPlanPolicy_t::HEURISTICS_CHOICE), tag + " build_plans");

    check_status(graph.get_workspace_size(workspace_bytes), tag + " get_workspace_size");

    return false;
}

// Autotuning is best-effort: on failure the graph keeps the plan the heuristics
// already chose. The attempt is made once per graph, so reporting a failure here
// costs at most one line per shape and is the only signal that a slower plan is
// now pinned for the rest of the process.
inline void report_autotune_skipped(const char* tag, const char* reason)
{
    cerr << (tag ? tag : "autotune")
         << ": autotune skipped, keeping the heuristic plan (" << reason << ").\n";
}

template<typename TensorMap>
inline void autotune_now(bool& pending, graph::Graph& graph,
                         TensorMap& tensors, int64_t& workspace_bytes,
                         const char* tag = nullptr)
{
    if (!pending) return;
    pending = false;

    Buffer tune_workspace{Device::CUDA};
    try
    {
        const int64_t tune_bytes = graph.get_autotune_workspace_size();
        if (tune_bytes > 0) tune_workspace.resize_bytes(Index(tune_bytes), Device::CUDA);
        check_status(graph.autotune(Backend::get_cudnn_handle(), tensors, tune_workspace.data), "autotune");
    }
    catch (const exception& e)
    {
        report_autotune_skipped(tag, e.what());
    }
    catch (...)
    {
        report_autotune_skipped(tag, "unknown error");
    }

#ifdef OPENNN_HAS_CUDA
    cudaGetLastError();
#endif

    workspace_bytes = graph.get_workspace_size();
}

template<typename TensorMap>
inline void autotune_with_scratch(bool& pending, graph::Graph& graph,
                                  const TensorMap& tensors, int64_t& workspace_bytes,
                                  const char* tag = nullptr)
{
    if (!pending) return;

    TensorMap scratch = tensors;
    vector<Buffer> buffers;
    buffers.reserve(scratch.size());

    // The scratch duplicates every tensor in the graph, so it is the largest
    // transient allocation the conv path makes. Failing to get it must not take
    // the whole cudnn-frontend path down with it: drop back to the heuristic
    // plan and keep training.
    try
    {
        for (auto& [tensor, pointer] : scratch)
        {
            if (tensor->get_is_pass_by_value()) continue;

            int64_t elements = 1;
            for (const int64_t dimension : tensor->get_dim()) elements *= dimension;

            Buffer& buffer = buffers.emplace_back(Device::CUDA);
            buffer.resize_bytes(Index(elements * int64_t(sizeof(float))), Device::CUDA);
            pointer = buffer.data;
        }
    }
    catch (const exception& e)
    {
        pending = false;
        buffers.clear();
#ifdef OPENNN_HAS_CUDA
        cudaGetLastError();
#endif
        report_autotune_skipped(tag, e.what());
        workspace_bytes = graph.get_workspace_size();
        return;
    }

    autotune_now(pending, graph, scratch, workspace_bytes, tag);
}

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
