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
             format("Convolution cudnn-frontend {}: {}", what, status.get_message()));
};

inline bool frontend_enabled()
{
    static const bool legacy_forced = [] {
        const char* value = getenv("OPENNN_CONV_LEGACY");
        return value && value[0] == '1';
    }();
    return !legacy_forced;
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

inline shared_ptr<cudnn_frontend::graph::Graph> new_graph()
{
    auto graph = make_shared<cudnn_frontend::graph::Graph>();
    graph->set_io_data_type(cudnn_frontend::DataType_t::FLOAT)
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

inline void finalize(cudnn_frontend::graph::Graph& graph, void*& workspace, const string& tag)
{
    cudnnHandle_t handle = Backend::get_cudnn_handle();

    check_status(graph.validate(), tag + " validate");
    check_status(graph.build_operation_graph(handle), tag + " build_operation_graph");
    check_status(graph.create_execution_plans({cudnn_frontend::HeurMode_t::A}), tag + " create_execution_plans");
    check_status(graph.build_plans(handle, cudnn_frontend::BuildPlanPolicy_t::HEURISTICS_CHOICE), tag + " build_plans");

    int64_t workspace_bytes = 0;
    check_status(graph.get_workspace_size(workspace_bytes), tag + " get_workspace_size");
    if (workspace_bytes > 0)
        workspace = device::allocate(Device::CUDA, Index(workspace_bytes));
}

}  // namespace cudnn_fe
}  // namespace opennn

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
