//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C U D N N   F R O N T E N D   U T I L I T I E S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#if defined(OPENNN_HAS_CUDA) && defined(HAVE_CUDNN_FRONTEND)

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
namespace cudnn_fe
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
inline void execute_graph(cudnn_frontend::graph::Graph& graph, TensorMap& tensors,
                          void* workspace, const string& what, const string& timing_label)
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
    catch (const exception& e)
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

// cuDNN-frontend tensor I/O data type for an OpenNN compute type. Activations
// (and weights) flow at this precision; intermediate accumulation and compute
// stay FP32 for accuracy (see new_graph). Per-channel stats and master
// gradients are forced to FP32 separately by their tensor builders.
inline cudnn_frontend::DataType_t fe_io_dtype(Type type)
{
    return type == Type::BF16 ? cudnn_frontend::DataType_t::BFLOAT16
                              : cudnn_frontend::DataType_t::FLOAT;
}

inline shared_ptr<cudnn_frontend::graph::Graph> new_graph(
    cudnn_frontend::DataType_t data_type = cudnn_frontend::DataType_t::FLOAT)
{
    auto graph = make_shared<cudnn_frontend::graph::Graph>();
    graph->set_io_data_type(data_type)
          .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
          .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
    return graph;
}

inline shared_ptr<cudnn_frontend::graph::Tensor_attributes>
nhwc_tensor(cudnn_frontend::graph::Graph& graph, const char* name,
            int64_t n, int64_t c, int64_t h, int64_t w)
{
    return graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                        .set_name(name)
                        .set_dim({n, c, h, w})
                        .set_stride(nhwc_strides(c, h, w)));
}

inline void set_nhwc_output(shared_ptr<cudnn_frontend::graph::Tensor_attributes>& tensor,
                     int64_t n, int64_t c, int64_t h, int64_t w)
{
    tensor->set_output(true)
           .set_dim({n, c, h, w})
           .set_stride(nhwc_strides(c, h, w));
}

// Builds the heuristic-chosen execution plan and reports its workspace size.
inline void finalize(cudnn_frontend::graph::Graph& graph, int64_t& workspace_bytes, const string& tag)
{
    cudnnHandle_t handle = Backend::get_cudnn_handle();

    workspace_bytes = 0;

    check_status(graph.validate(), tag + " validate");
    check_status(graph.build_operation_graph(handle), tag + " build_operation_graph");
    check_status(graph.create_execution_plans({cudnn_frontend::HeurMode_t::A, cudnn_frontend::HeurMode_t::FALLBACK}),
                 tag + " create_execution_plans");

    // Bound the per-conv scratch like the cublasLt path does; without this the
    // heuristic picks workspace proportional to batch (~4 MiB/sample) and OOMs
    // growing the shared buffer. FALLBACK heuristics guarantee a low/zero-
    // workspace plan stays available under the cap.
    graph.deselect_workspace_greater_than(device::conv_workspace_limit_bytes());

    check_status(graph.build_plans(handle, cudnn_frontend::BuildPlanPolicy_t::HEURISTICS_CHOICE), tag + " build_plans");
    check_status(graph.get_workspace_size(workspace_bytes), tag + " get_workspace_size");
}

}  // namespace cudnn_fe
}  // namespace opennn

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
