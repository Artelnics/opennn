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
            profiler::Stats& times = graph_timing_stats();
            const double total_ms = times.total_ms();
            times.print(cerr,
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
        return check_status(graph.execute(Backend::get_cudnn_handle(), tensors, workspace), what);

    // Timing is a diagnostic; the two events live for the thread.
    thread_local CudaEvent begin(cudaEventDefault);
    thread_local CudaEvent end(cudaEventDefault);
    device::record_event(begin, device::get_compute_stream());

    check_status(graph.execute(Backend::get_cudnn_handle(), tensors, workspace), what);

    device::record_event(end, device::get_compute_stream());
    device::synchronize_event(end);

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, begin, end));

    graph_timing_stats().add(timing_label, milliseconds);
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

// The pointer bindings a graph execution consumes: one device pointer per
// Tensor_attributes the graph declared.
using VariantPack = unordered_map<shared_ptr<graph::Tensor_attributes>, void*>;

// A (B, H, S, D) tensor over memory that is laid out either that way or,
// interleaved, as (B, S, H, D) - the layout the projection GEMMs write.
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

// A (1, C, 1, 1) FP32 tensor: the per-channel parameters and statistics batch
// normalization and a convolution bias work in. The strides are the NHWC ones
// for a 1x1 image, which is what the layout amounts to.
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

// Empty unless graph timing is on, and execute_graph reads an empty label as
// "do not time" - so the format cost is not paid when timing is off, and a
// non-empty constant would silently turn timing on.
template <typename... Args>
inline string timing_label(format_string<Args...> fmt, Args&&... args)
{
    if (!graph_timing_enabled()) return {};
    return format(fmt, std::forward<Args>(args)...);
}

// A one-element tensor. Pass by value for a host scalar the graph reads
// directly; leave it false for a device pointer.
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
// Autotune measures only the plans that were built. Building every candidate
// (BuildPlanPolicy_t::ALL) compiles dozens of engines per graph and costs
// minutes of warmup per shape set - measured ~5 min per batch point for
// ResNet-50 - while the heuristics already rank the candidates and the winner
// is nearly always among the first few. So build the first K viable candidates
// in heuristic order and let autotune() skip the unbuilt slots.
// OPENNN_AUTOTUNE_CANDIDATES overrides K; 0 or a negative count means "build
// all" (the old behaviour).
inline int64_t candidate_limit_from(long long value)
{
    return value <= 0 ? numeric_limits<int64_t>::max() : int64_t(value);
}

inline int64_t autotune_candidate_limit()
{
    static const int64_t limit = candidate_limit_from(env_int_or("OPENNN_AUTOTUNE_CANDIDATES", 8));
    return limit;
}

// Per-kind candidate count: OPENNN_AUTOTUNE_CANDIDATES_<KIND> (KIND = FORWARD,
// WGRAD, DGRAD, from the graph tag) overrides the global count for that graph
// kind - the weight gradient is the largest single component of a ResNet step
// and may deserve a wider search than the rest.
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

// Which heuristic list seeds the candidates: OPENNN_CUDNN_HEURISTICS = A
// (default), B, or AB (A's list followed by B's). Modes rank engines
// differently; which is right is a per-GPU measurement.
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

// Restrict a convolution graph's candidates to engines carrying the given
// numerical notes: OPENNN_CONV_ENGINE_NOTES = WINOGRAD, FFT, or WINOGRAD,FFT.
// Those engines rank low in the heuristics and need workspace, so a top-K
// search over the default order never reaches them; this lets an A/B ask
// whether they beat implicit GEMM on a given GPU (fp32 3x3 layers, mostly).
// Empty when unset. Applied to conv graphs only - batch-norm graphs have no
// such engines and would lose every candidate.
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

// Builds candidates in heuristic order until `limit` of them exist; plans the
// workspace budget bars fail fast and do not count. True if at least one built.
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
    // On by default: a miss only costs the build that would have happened anyway.
    static const bool enabled = env_flag_enabled("OPENNN_CUDNN_PLAN_CACHE", true);
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

    // The workspace cap changes which plans survive selection, and autotune
    // changes which survivor wins, so runs that differ in either must not share
    // a cached plan for the same graph.
    size_t selection = size_t(autotune_candidate_limit());
    for (const char* kind : {"forward", "wgrad", "dgrad"})
        selection = selection * 31 + size_t(autotune_candidate_limit(kind));
    for (const HeurMode_t mode : heuristic_modes()) selection = selection * 31 + size_t(mode) + 1;
    for (const NumericalNote_t note : conv_engine_notes()) selection = selection * 31 + size_t(note) + 7;

    const size_t key = std::hash<json>{}(structure)
        ^ (std::hash<int64_t>{}(device::conv_workspace_limit_bytes()) << 1)
        ^ (std::hash<bool>{}(device::conv_autotune_enabled()) << 2)
        ^ (std::hash<size_t>{}(selection) << 3);

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

// Attention finalizes separately from the layer graphs: no workspace budget,
// no autotune; it reports the workspace the built plan needs.
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

// Attention graphs (SDPA forward/backward): the heuristics' first plan, or,
// when the caller allows it and OPENNN_SDPA_AUTOTUNE=1, the top candidates of
// the A and B heuristic lists built for autotune_now() to time on the first
// real execution (returns true in that case: the caller must run autotune_now
// before executing). Measured neutral on an RTX 3060 (cuDNN 9.24 has one
// engine for that shape); kept for GPUs where cuDNN offers more.
inline bool sdpa_autotune_enabled()
{
    static const bool enabled = env_flag_enabled("OPENNN_SDPA_AUTOTUNE", false);
    return enabled;
}

inline bool finalize_attention(graph::Graph& graph, const string& tag, int64_t& workspace_bytes,
                               bool allow_autotune = false)
{
    const cudnnHandle_t handle = Backend::get_cudnn_handle();

    check_status(graph.validate(), tag + " validate");

    if (load_cached_plan(graph, handle, workspace_bytes)) return false;

    check_status(graph.build_operation_graph(handle), tag + " build_operation_graph");

    if (allow_autotune && sdpa_autotune_enabled())
    {
        check_status(graph.create_execution_plans({HeurMode_t::A, HeurMode_t::B, HeurMode_t::FALLBACK}),
                     tag + " create_execution_plans");
        if (build_top_candidates(graph, autotune_candidate_limit()))
        {
            workspace_bytes = autotune_workspace_bytes(graph);
            return true;
        }
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

// Builds the plan(s) of a graph. Returns true when the plan choice is left to
// the first real execution (device::conv_autotune_enabled(): the top-K
// candidates are built and timed on real tensors by autotune()).
inline bool finalize(graph::Graph& graph, int64_t& workspace_bytes, const string& tag)
{
    const cudnnHandle_t handle = Backend::get_cudnn_handle();
    const bool request_autotune = device::conv_autotune_enabled();

    workspace_bytes = 0;

    check_status(graph.validate(), tag + " validate");

    if (load_cached_plan(graph, handle, workspace_bytes)) return false;

    check_status(graph.build_operation_graph(handle), tag + " build_operation_graph");
    // The workspace budget is applied in every mode, autotune included. Left
    // unbounded, cuDNN's first heuristic choice for these NHWC shapes needs a
    // scratch that grows with the batch (~4 MiB per sample on sm_86: 2 GiB at
    // batch 512, 16 GiB at 4096) and is not even the fastest engine: capping to
    // the auto budget picks a plan needing ~16 KiB per sample and measured
    // 1.5-2x faster at every batch on an RTX 3060, while the unbounded plan
    // collapsed throughput 5-30x once its scratch spilled and then OOMed.
    const int64_t conv_workspace_cap = device::conv_workspace_limit_bytes();

    // Candidate list: the heuristic modes' engines under the workspace budget,
    // optionally restricted to the requested numeric notes for convolution
    // graphs (see conv_engine_notes). A restricted list can be empty for a
    // shape - 1x1 convolutions have no Winograd engine - so the build below
    // falls back to the unrestricted list rather than leaving it without a plan.
    const bool convolution_graph = tag == "forward" || tag == "wgrad" || tag == "dgrad";
    const auto prepare_candidates = [&](bool restrict_notes)
    {
        check_status(graph.create_execution_plans(heuristic_modes()), tag + " create_execution_plans");
        if (conv_workspace_cap > 0)
            graph.deselect_workspace_greater_than(conv_workspace_cap);
        if (restrict_notes)
            graph.select_numeric_notes(conv_engine_notes());
    };

    // Plans over budget are barred and left unbuilt (nullptr), as are the
    // candidates past the top-K; autotune() skips those slots. The access
    // violation once attributed to "cap + ALL" is
    // Graph::get_autotune_workspace_size(), which dereferences every slot
    // including the unbuilt ones — see autotune_workspace_bytes() below, which
    // is why that accessor is not used.
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
    // Still nothing: report it the way build_plans does, and let the caller decide.
    if (!built)
        check_status(graph.build_plans(handle, BuildPlanPolicy_t::HEURISTICS_CHOICE), tag + " build_plans");

    // An autotuned graph is measured against real data by the caller before a plan
    // is pinned, so caching here would store a plan the tuning has not chosen yet.
    if (request_autotune && built) return true;

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

// Largest workspace any *executable* candidate needs. Graph::get_autotune_workspace_size()
// computes the same maximum but dereferences every plan slot, and a workspace cap
// leaves the barred plans as nullptr — that is the crash that used to force
// autotune to run unbounded. The per-index query goes through
// is_plan_index_executable() first, so it is safe under a cap.

template<typename TensorMap>
inline void autotune_now(bool& pending, graph::Graph& graph,
                         TensorMap& tensors, int64_t& workspace_bytes,
                         const char* tag = nullptr)
{
    if (!pending) return;
    pending = false;
    // Candidates are timed: with several lanes another lane may be running
    // now and would skew the timing (measured: a mistuned plan cost 2-9% for
    // the whole step), so the device is idle before the first measurement.
    if (device::lanes_available() > 1) device::synchronize();

    Buffer tune_workspace{Device::CUDA};
    try
    {
        const int64_t tune_bytes = autotune_workspace_bytes(graph);
        if (tune_bytes > 0) tune_workspace.resize_bytes(Index(tune_bytes), Device::CUDA);
        check_status(graph.autotune(Backend::get_cudnn_handle(), tensors, tune_workspace.data()), "autotune");

        // The winner is now the candidate; persist it so the next process loads
        // the tuned plan instead of re-tuning (or, worse, settling for the
        // heuristic one under the same key).
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

// A built graph with what running it needs: the plan's workspace and whether
// finalize() left the plan choice to the first real execution (autotune).
struct GraphSlot
{
    shared_ptr<graph::Graph> graph;
    int64_t workspace_bytes = 0;
    bool autotune_pending = false;

    explicit operator bool() const noexcept { return graph != nullptr; }
    graph::Graph& operator*() const noexcept { return *graph; }

    // finalize() a freshly assembled graph and take it; on a throw the slot is
    // left empty (finalize reports the failure).
    void build(shared_ptr<graph::Graph> built, const string& tag)
    {
        graph.reset();
        autotune_pending = finalize(*built, workspace_bytes, tag);
        graph = std::move(built);
    }
};

// Autotune once on real tensors, then execute. `scratch_autotune` times the
// candidates on scratch copies of the outputs (graphs whose outputs feed the
// step, or accumulate); the plain form times them in place.
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
