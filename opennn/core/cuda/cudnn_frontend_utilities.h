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

#include "opennn/core/device_backend.h"
#include "opennn/core/profiler.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/tensor_types.h"

namespace opennn::cudnn_frontend
{
using namespace ::cudnn_frontend;

inline constexpr size_t graph_cache_capacity = 8;

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

inline profiler::Stats& graph_timing_stats()
{
    static profiler::Stats times;
    static const bool registered = [] {
        atexit(+[] {
            profiler::Stats& stats = graph_timing_stats();
            const double total_ms = stats.total_ms();
            stats.print(cerr,
                        format("total_gpu_ms={:.1f}", total_ms),
                        total_ms,
                        "GRAPH_TIMING");
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
        return check_status(graph.execute(device::get_cudnn_handle(), tensors, workspace), what);

    thread_local device::CudaEvent begin(cudaEventDefault);
    thread_local device::CudaEvent end(cudaEventDefault);
    device::record_event(begin.get(), device::get_compute_stream());

    check_status(graph.execute(device::get_cudnn_handle(), tensors, workspace), what);

    device::record_event(end.get(), device::get_compute_stream());
    device::synchronize_event(end.get());

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, begin.get(), end.get()));

    graph_timing_stats().add(timing_label, milliseconds);
}

inline void* shared_workspace(int64_t bytes)
{
    return bytes > 0 ? ensure_shared_scratch(size_t(bytes)) : nullptr;
}

template<typename GraphCache, typename Body>
bool run_frontend(GraphCache& cache, const char* label, Body&& body)
{
    const lock_guard lock(cache.access_mutex);
    if (cache.disabled) return false;

    try
    {
        body(cache);
        return true;
    }
    catch (const exception& e)
    {
        cache.disabled = true;
        cerr << label << ": cudnn-frontend path unavailable (" << e.what() << ").\n";
        return false;
    }
}

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

inline int64_t element_bytes(DataType_t dtype)
{
    switch (dtype)
    {
        case DataType_t::BFLOAT16:
        case DataType_t::HALF:      return 2;
        case DataType_t::INT8:
        case DataType_t::UINT8:
        case DataType_t::FP8_E4M3:
        case DataType_t::FP8_E5M2:  return 1;
        default:                    return 4;
    }
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

using VariantPack = unordered_map<shared_ptr<graph::Tensor_attributes>, void*>;

inline vector<int64_t> bhsd_strides(int64_t h, int64_t s, int64_t d, bool interleaved)
{
    return interleaved ? vector<int64_t>{s * h * d, d, h * d, 1}
                       : vector<int64_t>{h * s * d, s * d, d, 1};
}

inline shared_ptr<graph::Tensor_attributes>
bhsd_tensor(graph::Graph& graph, const char* name,
            int64_t b, int64_t h, int64_t s, int64_t d, bool interleaved)
{
    return graph.tensor(graph::Tensor_attributes()
                        .set_name(name)
                        .set_dim   ({b, h, s, d})
                        .set_stride(bhsd_strides(h, s, d, interleaved)));
}

inline void set_bhsd_output(shared_ptr<graph::Tensor_attributes>& tensor,
                            int64_t b, int64_t h, int64_t s, int64_t d, bool interleaved)
{
    tensor->set_output(true)
           .set_dim   ({b, h, s, d})
           .set_stride(bhsd_strides(h, s, d, interleaved));
}

inline shared_ptr<graph::Tensor_attributes>
per_channel_tensor(graph::Graph& graph, const char* name, int64_t channels)
{
    return graph.tensor(graph::Tensor_attributes()
                        .set_name(name)
                        .set_data_type(DataType_t::FLOAT)
                        .set_dim({1, channels, 1, 1})
                        .set_stride(nhwc_strides(channels, 1, 1)));
}

inline void set_per_channel_output(shared_ptr<graph::Tensor_attributes>& tensor, int64_t channels)
{
    tensor->set_output(true)
           .set_data_type(DataType_t::FLOAT)
           .set_dim({1, channels, 1, 1})
           .set_stride(nhwc_strides(channels, 1, 1));
}

template <typename... Args>
inline string timing_label(format_string<Args...> fmt, Args&&... args)
{
    if (!graph_timing_enabled()) return {};
    return format(fmt, std::forward<Args>(args)...);
}

inline shared_ptr<graph::Tensor_attributes>
scalar_tensor(graph::Graph& graph, const char* name, DataType_t dtype,
              bool pass_by_value = false, int64_t batch = 1)
{
    return graph.tensor(graph::Tensor_attributes()
                        .set_name(name)
                        .set_dim   ({batch, 1, 1, 1})
                        .set_stride({1, 1, 1, 1})
                        .set_data_type(dtype)
                        .set_is_pass_by_value(pass_by_value));
}

inline int64_t candidate_limit_from(long long value)
{
    return value <= 0 ? numeric_limits<int64_t>::max() : int64_t(value);
}

inline int64_t autotune_candidate_limit()
{
    static const int64_t limit = candidate_limit_from(env_int_or("OPENNN_AUTOTUNE_CANDIDATES", 8));
    return limit;
}

inline int64_t autotune_candidate_limit(const string& tag)
{
    static const auto per_kind = []
    {
        map<string, int64_t> limits;
        for (const char* kind : {"forward", "wgrad", "dgrad"})
        {
            string name = "OPENNN_AUTOTUNE_CANDIDATES_" + string(kind);
            for (char& c : name) c = char(toupper(c));
            if (const char* setting = getenv(name.c_str()); setting && *setting)
                limits[kind] = candidate_limit_from(env_int_or(name.c_str(), 0));
        }
        return limits;
    }();

    const auto found = per_kind.find(tag);
    return found == per_kind.end() ? autotune_candidate_limit() : found->second;
}

inline vector<HeurMode_t> heuristic_modes()
{
    static const vector<HeurMode_t> modes = []
    {
        const char* setting = getenv("OPENNN_CUDNN_HEURISTICS");
        const string value = setting ? setting : "A";
        if (value == "B")  return vector<HeurMode_t>{HeurMode_t::B, HeurMode_t::FALLBACK};
        if (value == "AB") return vector<HeurMode_t>{HeurMode_t::A, HeurMode_t::B, HeurMode_t::FALLBACK};
        return vector<HeurMode_t>{HeurMode_t::A, HeurMode_t::FALLBACK};
    }();
    return modes;
}

inline const vector<NumericalNote_t>& conv_engine_notes()
{
    static const vector<NumericalNote_t> notes = []
    {
        vector<NumericalNote_t> selected;
        const char* setting = getenv("OPENNN_CONV_ENGINE_NOTES");
        const string value = setting ? setting : "";
        if (value.find("WINOGRAD") != string::npos) selected.push_back(NumericalNote_t::WINOGRAD);
        if (value.find("FFT") != string::npos)      selected.push_back(NumericalNote_t::FFT);
        return selected;
    }();
    return notes;
}

inline bool build_top_candidates(graph::Graph& graph, int64_t limit)
{
    const int64_t count = graph.get_execution_plan_count();
    int64_t built = 0;

    for (int64_t index = 0; index < count && built < limit; ++index)
        if (graph.build_plan_at_index(index).is_good()) ++built;

    return built > 0;
}

inline bool plan_cache_enabled()
{
    static const bool enabled = env_flag_enabled("OPENNN_CUDNN_PLAN_CACHE", true);
    return enabled;
}

inline bool sdpa_autotune_enabled();
inline int64_t sdpa_workspace_cap_override_bytes();

inline const std::filesystem::path& plan_cache_directory()
{
    static const std::filesystem::path directory = []
    {
        const char* override_path = getenv("OPENNN_CUDNN_PLAN_CACHE_DIR");

        std::filesystem::path root;

        if (override_path && *override_path)
            root = std::filesystem::path(override_path);
        else
        {
            std::error_code error;
            const std::filesystem::path temporary = std::filesystem::temp_directory_path(error);
            if (error) return std::filesystem::path{};
            root = temporary / "opennn-cudnn-plans";
        }

        return root / format("sm{}-cudnn{}", device_sm_version(), CUDNN_VERSION);
    }();

    return directory;
}

inline std::filesystem::path plan_cache_file(const graph::Graph& graph)
{
    // The cache key is the serialised graph, and serialising asks cuDNN for
    // CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION -- an attribute the
    // Windows builds of cuDNN 9 do not implement. store_cached_plan() already
    // treats a failed serialise as "do not cache"; this path did not, so the
    // throw escaped through load_cached_plan(), the frontend disabled itself,
    // and every convolution became fatal. A machine that cannot serialise a
    // plan should lose the cache, not the GPU.
    json structure;
    try
    {
        graph.serialize(structure);
    }
    catch (const std::exception&)
    {
        static std::once_flag reported;
        std::call_once(reported, []{
            std::cerr << "cudnn plan cache: this cuDNN build cannot serialise "
                         "execution plans; continuing without the cache.\n"; });
        return {};
    }
    structure.erase("gid");

    size_t selection = size_t(autotune_candidate_limit());
    for (const char* kind : {"forward", "wgrad", "dgrad"})
        selection = selection * 31 + size_t(autotune_candidate_limit(kind));
    for (const HeurMode_t mode : heuristic_modes()) selection = selection * 31 + size_t(mode) + 1;
    for (const NumericalNote_t note : conv_engine_notes()) selection = selection * 31 + size_t(note) + 7;

    selection = selection * 31 + (sdpa_autotune_enabled() ? 3u : 0u);

    // The cached entry is the plan that won an autotune, and the SDPA workspace
    // cap changes which plan that is, so a capped build must not load an
    // uncapped winner from the same directory. Only the override belongs here:
    // the shape half of the cap is a function of the extents, which the
    // serialised graph already carries.
    selection = selection * 31 + size_t(sdpa_workspace_cap_override_bytes() + 2);

    const size_t key = std::hash<json>{}(structure)
        ^ (std::hash<int64_t>{}(device::conv_workspace_limit_bytes()) << 1)
        ^ (std::hash<bool>{}(device::conv_autotune_enabled()) << 2)
        ^ (std::hash<size_t>{}(selection) << 3);

    return plan_cache_directory() / format("{:016x}.plan", key);
}

inline bool load_cached_plan(graph::Graph& graph, const cudnnHandle_t handle,
                             int64_t& workspace_bytes)
{
    if (!plan_cache_enabled() || plan_cache_directory().empty()) return false;

    std::error_code failed;
    const std::filesystem::path file = plan_cache_file(graph);
    if (file.empty()) return false;
    if (!std::filesystem::exists(file, failed) || failed) return false;

    std::ifstream stream(file, std::ios::binary);
    if (!stream) return false;

    const vector<uint8_t> blob((std::istreambuf_iterator<char>(stream)),
                                std::istreambuf_iterator<char>());
    if (blob.empty()) return false;

    constexpr bool enforce_precompiled = false;
    constexpr bool run_warmup = false;

    if (graph.deserialize(handle, blob, enforce_precompiled, run_warmup).is_bad()) return false;

    return graph.get_workspace_size(workspace_bytes).is_good();
}

inline void store_cached_plan(graph::Graph& graph)
{
    if (!plan_cache_enabled() || plan_cache_directory().empty()) return;

    vector<uint8_t> blob;

    // Status-checking this is not enough: cudnn-frontend throws out of
    // serialize() rather than returning when the backend refuses the JSON
    // attribute, which is what Windows cuDNN does. The status check stays for
    // the builds that report it politely.
    try
    {
        if (graph.serialize(blob, false).is_bad() || blob.empty()) return;
    }
    catch (const std::exception&)
    {
        return;
    }

    std::error_code failed;
    std::filesystem::create_directories(plan_cache_directory(), failed);
    if (failed) return;

    static atomic<uint64_t> sequence{0};

    const std::filesystem::path file = plan_cache_file(graph);
    if (file.empty()) return;
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

inline int64_t autotune_workspace_bytes(const graph::Graph& graph)
{
    int64_t maximum = 0;
    const int64_t count = graph.get_execution_plan_count();

    for (int64_t index = 0; index < count; ++index)
    {
        int64_t bytes = 0;
        if (graph.get_workspace_size_plan_at_index(index, bytes).is_good())
            maximum = max(maximum, bytes);
    }

    return maximum;
}

inline bool sdpa_autotune_enabled()
{
    static const bool enabled = env_flag_enabled("OPENNN_SDPA_AUTOTUNE", false);
    return enabled;
}

// Megabytes, mirroring OPENNN_CONV_WORKSPACE_MB: 0 removes the cap and restores
// pick-by-time over every candidate, a positive value pins it, and unset leaves
// the shape-derived bound below. It exists so the cap stays measurable rather
// than baked in -- the transformer cell has to be re-measured before SDPA
// autotune can be turned on by default, and this is the knob that sweep varies.
inline int64_t sdpa_workspace_cap_override_bytes()
{
    static const int64_t bytes = []
    {
        const long long megabytes = env_int_or("OPENNN_SDPA_WORKSPACE_MB", -1);
        return megabytes < 0 ? int64_t(-1) : int64_t(megabytes) * 1024 * 1024;
    }();
    return bytes;
}

// A plan may not ask for more scratch than an fp32 copy of the node's own
// Q, K and V. That is the same rule the convolution path applies -- bound the
// workspace by the tensors the layer already pays for -- but it has to be
// re-derived here rather than reused, because
// device::conv_workspace_limit_bytes() is the largest *convolution* activation
// slot (forward_propagation.cpp) and the cnn benchmark pins it flat at 16 MiB.
// Neither number is a statement about an attention node, and 16 MiB would bar
// every plan the transformer cell needs.
//
// The bound is exactly the size of the fp32 dQ/dK/dV accumulators a flash
// engine legitimately keeps, so it excludes no flash plan, while an engine that
// materialises the B*H*Sq*Skv score matrix in fp32 is admitted for at most one
// such buffer and excluded outright once Sq passes 3*head_dim, where that
// matrix overtakes the accumulators and then keeps growing quadratically. At
// the transformer cell's 32 x 8 x 130 x 64 that is a 24.5 MiB cap against a
// 16.5 MiB score matrix and 12.2 MiB of bf16 Q/K/V.
inline int64_t sdpa_workspace_cap_bytes(int64_t batch, int64_t heads,
                                        int64_t q_seq, int64_t src_seq, int64_t head_dim)
{
    if (const int64_t override_bytes = sdpa_workspace_cap_override_bytes(); override_bytes >= 0)
        return override_bytes;

    const int64_t operands   = 4 * batch * heads * head_dim * (q_seq + 2 * src_seq);
    const int64_t statistics = 4 * batch * heads * q_seq;

    // The floor keeps a small shape -- a unit test, a short prompt -- from being
    // capped below a plan's fixed scratch (tile counters, semaphores, the
    // frontend's own node workspace), where every candidate is small anyway and
    // there is nothing to exclude.
    return max(operands + statistics, int64_t(4) * 1024 * 1024);
}

inline bool finalize_attention(graph::Graph& graph, const string& tag, int64_t& workspace_bytes,
                               bool allow_autotune = false, int64_t workspace_cap = 0)
{
    const cudnnHandle_t handle = device::get_cudnn_handle();

    check_status(graph.validate(), tag + " validate");

    if (load_cached_plan(graph, handle, workspace_bytes)) return false;

    check_status(graph.build_operation_graph(handle), tag + " build_operation_graph");

    if (allow_autotune && sdpa_autotune_enabled())
    {
        check_status(graph.create_execution_plans({HeurMode_t::A, HeurMode_t::B, HeurMode_t::FALLBACK}),
                     tag + " create_execution_plans");

        if (workspace_cap > 0)
            graph.deselect_workspace_greater_than(workspace_cap);

        if (build_top_candidates(graph, autotune_candidate_limit()))
        {
            // What the slot carries is the workspace the graph would execute
            // with, which is the candidate plan's; autotune_now() overwrites it
            // with the winner's and sizes its own tuning scratch from
            // autotune_workspace_bytes(). Parking that max-over-candidates here
            // instead left the slot's persistent size at the largest built
            // candidate's, and that is the size shared_workspace() grows to on
            // any path where the tuning run does not happen.
            check_status(graph.get_workspace_size(workspace_bytes), tag + " workspace");
            return true;
        }

        // Nothing survived the cap, so the heuristic plan below has to be chosen
        // without it. The cap narrows an autotune; it must never be the reason
        // attention is left with no plan at all.
        graph.deselect_workspace_greater_than(numeric_limits<int64_t>::max());
    }

    check_status(graph.create_execution_plans({HeurMode_t::A}), tag + " create_execution_plans");
    check_status(graph.build_plans(handle, BuildPlanPolicy_t::HEURISTICS_CHOICE), tag + " build_plans");
    check_status(graph.get_workspace_size(workspace_bytes), tag + " workspace");

    store_cached_plan(graph);
    return false;
}

inline shared_ptr<graph::Tensor_attributes>
seq_len_scalar(graph::Graph& graph, const char* name, int64_t batch = 1)
{
    return scalar_tensor(graph, name, DataType_t::INT32, false, batch);
}

inline bool finalize(graph::Graph& graph, int64_t& workspace_bytes, const string& tag)
{
    const cudnnHandle_t handle = device::get_cudnn_handle();
    const bool request_autotune = device::conv_autotune_enabled();

    workspace_bytes = 0;

    check_status(graph.validate(), tag + " validate");

    if (load_cached_plan(graph, handle, workspace_bytes)) return false;

    check_status(graph.build_operation_graph(handle), tag + " build_operation_graph");
    const int64_t conv_workspace_cap = device::conv_workspace_limit_bytes();

    const bool convolution_graph = tag == "forward" || tag == "wgrad" || tag == "dgrad";
    const auto prepare_candidates = [&](bool restrict_notes)
    {
        check_status(graph.create_execution_plans(heuristic_modes()), tag + " create_execution_plans");
        if (conv_workspace_cap > 0)
            graph.deselect_workspace_greater_than(conv_workspace_cap);
        if (restrict_notes)
            graph.select_numeric_notes(conv_engine_notes());
    };

    const auto build_candidates = [&]() -> bool
    {
        if (request_autotune)
            return build_top_candidates(graph, autotune_candidate_limit(tag));
        return graph.build_plans(handle, BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good();
    };

    bool restricted = convolution_graph && !conv_engine_notes().empty();
    prepare_candidates(restricted);
    bool built = build_candidates();
    if (!built && restricted)
    {
        restricted = false;
        prepare_candidates(false);
        built = build_candidates();
    }
    if (!built)
        check_status(graph.build_plans(handle, BuildPlanPolicy_t::HEURISTICS_CHOICE), tag + " build_plans");

    if (request_autotune && built) return true;

    check_status(graph.get_workspace_size(workspace_bytes), tag + " get_workspace_size");

    store_cached_plan(graph);

    return false;
}

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
    if (device::lanes_available() > 1) device::synchronize();

    Buffer tune_workspace{Device::CUDA};
    try
    {
        const int64_t tune_bytes = autotune_workspace_bytes(graph);
        if (tune_bytes > 0) tune_workspace.resize_bytes(Index(tune_bytes), Device::CUDA);
        check_status(graph.autotune(device::get_cudnn_handle(), tensors, tune_workspace.data()), "autotune");

        store_cached_plan(graph);
    }
    catch (const exception& e)
    {
        report_autotune_skipped(tag, e.what());
    }
    catch (...)
    {
        report_autotune_skipped(tag, "unknown error");
    }

    device::reset_last_error();

    workspace_bytes = graph.get_workspace_size();
}

template<typename TensorMap>
inline void autotune_with_scratch(bool& pending, graph::Graph& graph,
                                  const TensorMap& tensors, int64_t& workspace_bytes,
                                  const char* tag = nullptr)
{
    if (!pending) return;
    if (device::lanes_available() > 1) device::synchronize();

    TensorMap scratch = tensors;
    vector<Buffer> buffers;
    buffers.reserve(scratch.size());

    try
    {
        for (auto& [tensor, pointer] : scratch)
        {
            if (tensor->get_is_pass_by_value()) continue;

            int64_t elements = 1;
            for (const int64_t dimension : tensor->get_dim()) elements *= dimension;

            Buffer& buffer = buffers.emplace_back(Device::CUDA);
            buffer.resize_bytes(Index(elements * element_bytes(tensor->get_data_type())),
                                Device::CUDA);
            pointer = buffer.data();
        }
    }
    catch (const exception& e)
    {
        pending = false;
        buffers.clear();
        device::reset_last_error();
        report_autotune_skipped(tag, e.what());
        workspace_bytes = graph.get_workspace_size();
        return;
    }

    autotune_now(pending, graph, scratch, workspace_bytes, tag);
}

struct GraphSlot
{
    shared_ptr<graph::Graph> graph;
    int64_t workspace_bytes = 0;
    bool autotune_pending = false;

    explicit operator bool() const noexcept { return graph != nullptr; }
    graph::Graph& operator*() const noexcept { return *graph; }

    void build(shared_ptr<graph::Graph> built, const string& tag)
    {
        graph.reset();
        autotune_pending = finalize(*built, workspace_bytes, tag);
        graph = std::move(built);
    }

    // Attention finalizes under its own policy -- its own autotune switch, its
    // own heuristic modes and its own workspace cap -- so it cannot share
    // build()'s convolution one. Keeping it a named second entry point makes
    // that difference visible instead of hiding it behind a slot that looks
    // interchangeable. The cap comes from the caller because it is a function
    // of the node's extents, and the graph exposes those only through a full
    // serialise, not an accessor; sdpa_workspace_cap_bytes() derives it from
    // the cache key instead, and 0 leaves the candidates uncapped.
    void build_attention(shared_ptr<graph::Graph> built, const string& tag,
                         bool allow_autotune = true, int64_t workspace_cap = 0)
    {
        graph.reset();
        autotune_pending = finalize_attention(*built, tag, workspace_bytes,
                                              allow_autotune, workspace_cap);
        graph = std::move(built);
    }
};

template<typename TensorMap>
inline void run_slot(GraphSlot& slot, TensorMap& tensors, const char* what,
                     const string& timing_label, bool scratch_autotune)
{
    if (scratch_autotune)
        autotune_with_scratch(slot.autotune_pending, *slot.graph, tensors, slot.workspace_bytes, what);
    else
        autotune_now(slot.autotune_pending, *slot.graph, tensors, slot.workspace_bytes, what);
    execute_graph(*slot.graph, tensors, shared_workspace(slot.workspace_bytes), what, timing_label);
}

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
