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

#include "tensor_types.h"
#include "device_backend.h"
#include "string_utilities.h"
#include "memory_debug.h"

// cudnn-frontend graph path for fp32 convolutions and batch normalization:
// the legacy v7 API has poor NHWC-fp32 kernel coverage (backward-filter ~6x,
// batch normalization ~10x slower than the engines the graph API selects for
// the same shapes). Graphs are cached per batch size; any failure disables
// the path for that operator and falls back to the legacy implementation.
namespace opennn
{
namespace cudnn_frontend
{

inline const auto check_status = [](auto status, const string& what) {
    throw_if(status.is_bad(),
             format("cudnn-frontend {}: {}", what, status.get_message()));
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
    // Legacy v7 API is forced from code with device::set_conv_legacy(true).
    if (device::conv_legacy_forced()) return false;
    // cuDNN-frontend batchnorm/conv graph API requires SM 8.0+ (Ampere).
    // Silently skip on older hardware to avoid per-layer warning spam.
    return device_sm_version() >= 800;
}

inline bool autotune_enabled()
{
    // On by default; toggle from code with device::set_conv_autotune(false).
    return device::conv_autotune_enabled();
}

// With OPENNN_GRAPH_TIMING=1 every graph execution is timed with CUDA events
// (per-label totals printed at exit). Incompatible with set_cuda_graph(true).
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
inline void execute_graph(cudnn_frontend::graph::Graph&, TensorMap&,
                          void*, const string&, const string&)
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

// All frontend graphs draw their scratch from one shared device buffer (the
// same one the legacy conv path and cublasLt already use): the ops execute
// serially on the compute stream, so the live peak is max(individual
// workspace), not the sum. A 0-byte request passes nullptr, as cuDNN expects.
inline void* shared_workspace(int64_t bytes)
{
    return bytes > 0 ? ensure_cudnn_conv_workspace(size_t(bytes)) : nullptr;
}

// Runs a frontend-path body with the shared cache/disable/fallback protocol;
// returns false when the caller should take the legacy path instead.
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
    catch (const exception&)
    {
        cache->disabled = true;
        cerr << label << ": cudnn-frontend path unavailable (" << e.what()
             << "); falling back to the legacy cuDNN API.\n";
        return false;
    }
}

inline vector<int64_t> nhwc_strides(int64_t c, int64_t h, int64_t w)
{
    return {h * w * c, 1, w * c, c};
}

inline shared_ptr<cudnn_frontend::graph::Graph> new_graph()
{
    auto graph = make_shared<cudnn_frontend::graph::Graph>();
    graph->set_io_data_type(cudnn_frontend::DataType_t::FLOAT)
          .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
          .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
    return graph;
}

inline shared_ptr<cudnn_frontend::graph::Tensor_attributes>
nhwc_tensor(cudnn_frontend::graph::Graph&, const char*,
            int64_t n, int64_t c, int64_t h, int64_t w)
{
    return graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                        .set_name(name)
                        .set_dim({n, c, h, w})
                        .set_stride(nhwc_strides(c, h, w)));
}

inline void set_nhwc_output(shared_ptr<cudnn_frontend::graph::Tensor_attributes>&,
                     int64_t n, int64_t c, int64_t h, int64_t w)
{
    tensor->set_output(true)
           .set_dim({n, c, h, w})
           .set_stride(nhwc_strides(c, h, w));
}

// With, every candidate plan is built so the first execution
// can time them and keep the fastest (cudnn.benchmark=True);
// returns whether that mode is active for this graph.
inline bool finalize(cudnn_frontend::graph::Graph&, int64_t&, const string&,
                     bool request_autotune = false)
{
    cudnnHandle_t handle = Backend::get_cudnn_handle();

    workspace_bytes = 0;

    check_status(graph.validate(), tag + " validate");
    check_status(graph.build_operation_graph(handle), tag + " build_operation_graph");
    check_status(graph.create_execution_plans({cudnn_frontend::HeurMode_t::A, cudnn_frontend::HeurMode_t::FALLBACK}),
                 tag + " create_execution_plans");

    const bool autotune = request_autotune
        && graph.build_plans(handle, cudnn_frontend::BuildPlanPolicy_t::ALL).is_good();

    // The autotuned workspace size is fixed after plan selection (autotune_now);
    // the scratch itself comes from the shared buffer, requested at execute time.
    if (autotune) return true;

    check_status(graph.build_plans(handle, cudnn_frontend::BuildPlanPolicy_t::HEURISTICS_CHOICE), tag + " build_plans");

    check_status(graph.get_workspace_size(workspace_bytes), tag + " get_workspace_size");

    return false;
}

// On the first execution of an autotune-built graph: times every plan with a
// throwaway max-size workspace, keeps the fastest, then allocates the
// persistent workspace for the chosen plan only.
template<typename TensorMap>
inline void autotune_now(bool&, cudnn_frontend::graph::Graph&,
                         TensorMap&, int64_t&)
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
    catch (...) {}

    workspace_bytes = graph.get_workspace_size();
}

// Autotune variant for graphs with in-place or state-updating tensors (batch
// norm): times the plans on throwaway buffers so repeated execution cannot
// corrupt training data.
template<typename TensorMap>
inline void autotune_with_scratch(bool&, cudnn_frontend::graph::Graph&,
                                  const TensorMap&, int64_t&)
{
    if (!pending) return;

    TensorMap scratch = tensors;
    vector<Buffer> buffers;
    buffers.reserve(scratch.size());

    for (auto& [tensor, pointer] : scratch)
    {
        if (tensor->get_is_pass_by_value()) continue;

        int64_t elements = 1;
        for (const int64_t dimension : tensor->get_dim()) elements *= dimension;

        Buffer& buffer = buffers.emplace_back(Device::CUDA);
        buffer.resize_bytes(Index(elements * sizeof(float)), Device::CUDA);
        pointer = buffer.data;
    }

    autotune_now(pending, graph, scratch, workspace_bytes);
}

}  // namespace cudnn_frontend
}  // namespace opennn

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
