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

#include <filesystem>
#include <fstream>

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
        case Type::Auto:
        case Type::FP32: return DataType_t::FLOAT;
        case Type::BF16: return DataType_t::BFLOAT16;
        case Type::INT8: return DataType_t::INT8;
    }

    throw invalid_argument("Unsupported data type for a cuDNN graph.");
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

// Execution-plan disk cache. Building plans dominates process startup: on
// ResNet-50 at batch 2048 the fixed cost is ~9 s, of which CUDA context and cuDNN
// handle creation are only ~0.22 s, so nearly all of it is plan building — about
// eight times one training epoch, paid again on every run. Serializing the chosen
// plan turns that into a file read (measured 9.42 s -> 6.23 s, 192 plans).
//
// A plan is only valid for the GPU, cuDNN build and workspace budget it was
// chosen under, so those name the directory while a shape-aware structural hash
// of the graph names the file. A miss, a stale file or a failed deserialize all
// fall through to the normal build, so a bad cache costs time, never correctness.
inline bool plan_cache_enabled()
{
    // On by default: a miss only costs the build that would have happened anyway.
    static const bool enabled = []
    {
        const char* setting = getenv("OPENNN_CUDNN_PLAN_CACHE");
        if (!setting || !*setting) return true;

        const string value(setting);
        return value != "0" && value != "false" && value != "off" && value != "no";
    }();

    return enabled;
}

inline const std::filesystem::path& plan_cache_directory()
{
    static const std::filesystem::path directory = []
    {
        const char* override_path = getenv("OPENNN_CUDNN_PLAN_CACHE_DIR");

        const std::filesystem::path root = override_path && *override_path
            ? std::filesystem::path(override_path)
            : std::filesystem::temp_directory_path() / "opennn-cudnn-plans";

        // cuDNN picks engines per architecture and revises them between releases,
        // so a plan may only be reloaded by the exact pair that produced it.
        return root / format("sm{}-cudnn{}", device_sm_version(), CUDNN_VERSION);
    }();

    return directory;
}

inline std::filesystem::path plan_cache_file(const graph::Graph& graph)
{
    // Graph::key() is private, so hash the structural json the same way it does.
    // "gid" is a per-process graph counter, not part of the shape or topology, so
    // leaving it in would make every run miss its own cache.
    json structure;
    graph.serialize(structure);
    structure.erase("gid");

    // The workspace cap changes which plans survive selection, so two runs with
    // different caps must not share a cached plan for the same graph.
    const size_t key = std::hash<json>{}(structure)
        ^ (std::hash<int64_t>{}(device::conv_workspace_limit_bytes()) << 1);

    return plan_cache_directory() / format("{:016x}.plan", key);
}

inline bool load_cached_plan(graph::Graph& graph, const cudnnHandle_t handle,
                             int64_t& workspace_bytes)
{
    if (!plan_cache_enabled()) return false;

    std::error_code failed;
    const std::filesystem::path file = plan_cache_file(graph);
    if (!std::filesystem::exists(file, failed) || failed) return false;

    std::ifstream stream(file, std::ios::binary);
    if (!stream) return false;

    const vector<uint8_t> blob((std::istreambuf_iterator<char>(stream)),
                                std::istreambuf_iterator<char>());
    if (blob.empty()) return false;

    // A plan from another GPU or cuDNN build lands in a different directory, but a
    // truncated file can still reach here; deserialize reports that rather than
    // throwing, and the caller then builds the plan normally. The warmup capture
    // is skipped because every graph here runs many times, so the first real call
    // primes it just as well (measured neutral: same wall clock, same final loss).
    constexpr bool enforce_precompiled = false;
    constexpr bool run_warmup = false;

    if (graph.deserialize(handle, blob, enforce_precompiled, run_warmup).is_bad()) return false;

    return graph.get_workspace_size(workspace_bytes).is_good();
}

inline void store_cached_plan(graph::Graph& graph)
{
    if (!plan_cache_enabled()) return;

    vector<uint8_t> blob;

    // Plan-only payload: the structure is rebuilt by our own graph construction,
    // so serializing it again would only make the file bigger.
    if (graph.serialize(blob, false).is_bad() || blob.empty()) return;

    std::error_code failed;
    std::filesystem::create_directories(plan_cache_directory(), failed);
    if (failed) return;

    // Write-then-rename so a concurrent reader never observes a half-written plan.
    // The temporary name only has to be unique among the writers racing for this
    // file; if two ever collide the write fails and the plan is simply rebuilt.
    static atomic<uint64_t> sequence{0};

    const std::filesystem::path file = plan_cache_file(graph);
    const std::filesystem::path pending = file.string()
        + format(".{:x}-{}.tmp", std::hash<thread::id>{}(this_thread::get_id()), sequence++);

    {
        std::ofstream stream(pending, std::ios::binary | std::ios::trunc);
        if (!stream) return;
        stream.write(reinterpret_cast<const char*>(blob.data()), std::streamsize(blob.size()));
        if (!stream) { std::filesystem::remove(pending, failed); return; }
    }

    std::filesystem::rename(pending, file, failed);
    if (failed) std::filesystem::remove(pending, failed);
}

// Attention finalizes separately from the layer graphs: it needs no workspace
// accounting and never autotunes.
inline void finalize_attention(graph::Graph& graph, const string& tag)
{
    const cudnnHandle_t handle = Backend::get_cudnn_handle();

    check_status(graph.validate(), tag + " validate");

    int64_t attention_workspace_bytes = 0;
    if (load_cached_plan(graph, handle, attention_workspace_bytes)) return;

    check_status(graph.build_operation_graph(handle), tag + " build_operation_graph");
    check_status(graph.create_execution_plans({HeurMode_t::A}), tag + " create_execution_plans");
    check_status(graph.build_plans(handle, BuildPlanPolicy_t::HEURISTICS_CHOICE), tag + " build_plans");

    store_cached_plan(graph);
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

    if (load_cached_plan(graph, handle, workspace_bytes)) return false;

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

    // An autotuned graph is measured against real data by the caller before a plan
    // is pinned, so caching here would store a plan the tuning has not chosen yet.
    if (autotune) return true;

    check_status(graph.build_plans(handle, BuildPlanPolicy_t::HEURISTICS_CHOICE), tag + " build_plans");

    check_status(graph.get_workspace_size(workspace_bytes), tag + " get_workspace_size");

    store_cached_plan(graph);

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
