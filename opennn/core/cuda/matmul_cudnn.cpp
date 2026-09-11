// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#include "opennn/core/cuda/matmul_cudnn.h"
#include "opennn/core/log.h"

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/cudnn_frontend_utilities.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/string_utilities.h"

namespace opennn::matmul::cudnn
{

namespace fe = ::cudnn_frontend;
namespace house = ::opennn::cudnn_frontend;

namespace
{

// The variant pack is keyed by uid, as in cudnn_matmul_probe.cu, rather than
// by tensor pointer as the convolution and attention graphs are. Those graphs
// keep their Tensor_attributes alive in the cache entry beside the graph and
// index the pack with them; here the pack is rebuilt on every call with new
// device pointers, and four integers are cheaper to carry and impossible to
// dangle. It is the same pack the frontend accepts either way.
constexpr int64_t uid_left   = 1;   // cuBLASLt's B operand: the samples
constexpr int64_t uid_right  = 2;   // cuBLASLt's A operand: the weights
constexpr int64_t uid_bias   = 3;
constexpr int64_t uid_output = 4;

bool enabled()
{
    static const bool on = env_flag_enabled("OPENNN_CUDNN_MATMUL", true);
    return on;
}

// The workspace cap is deliberately the same 32 MiB cuBLASLt is already
// allowed to ask for during its own candidate search
// (cublas_lt_workspace_search_bytes), so a cuDNN plan can never claim more
// scratch than a cuBLASLt algorithm was already permitted to claim.
//
// That bounds the shared-scratch high-water mark; it does not leave it
// untouched, and the benchmark reports peak memory. Two ways it can move:
// autotune_matmul_plan sizes its timing scratch to the largest workspace over ALL
// candidates, cuDNN's included, so a fat cuDNN engine raises the buffer even
// on a shape where it goes on to lose; and when a cuDNN engine wins,
// run_lt_matmul_cached ensures ITS workspace on every call, which may exceed
// the cuBLASLt algorithm's. The shared-scratch Buffer never shrinks, so both
// are permanent for the process. Lower this knob if peak memory moves.
int64_t workspace_cap_bytes()
{
    static const int64_t bytes =
        int64_t(clamp(env_int_or("OPENNN_CUDNN_MATMUL_WORKSPACE_MB", 32), 0LL, 4096LL))
        * 1024 * 1024;
    return bytes;
}

// Below this the GEMM is not where the time is, and enumerating and building
// sixty cuDNN engine configurations costs more warmup than the kernel can
// ever return. The bound is stated in arithmetic rather than in extents
// because that is what the cost is proportional to: the probe's evidence is a
// 17.2 GFLOP contraction, and the tile-power model the selection rule leans
// on was fitted on an L2-resident compute-bound GEMM of that size. Under a
// few GFLOP a bf16 GEMM on this class of part is dominated by launch and
// epilogue, where nothing here has been measured at all.
double minimum_gflop()
{
    static const double gflop =
        double(clamp(env_int_or("OPENNN_CUDNN_MATMUL_MIN_GFLOP", 8), 0LL, 1000000LL));
    return gflop;
}

int64_t minimum_dimension()
{
    static const int64_t dimension =
        int64_t(clamp(env_int_or("OPENNN_CUDNN_MATMUL_MIN_DIM", 128), 1LL, 1000000LL));
    return dimension;
}

// 0 builds every configuration cuDNN offers, which is what the probe did and
// what found the winner. It is a knob because the winner's position in the
// enumeration is a property of the cuDNN build, not something this file can
// assume: capping the search is a way to trade warmup for the risk of missing
// an engine, and that trade should be the operator's to make, not a constant.
int64_t candidate_limit()
{
    static const int64_t limit =
        int64_t(clamp(env_int_or("OPENNN_CUDNN_MATMUL_CANDIDATES", 0), 0LL, 4096LL));
    return limit;
}

string plan_name(fe::graph::Graph& graph, int64_t index)
{
    string name;
    try
    {
        if (graph.get_plan_name_at_index(index, name).is_bad()) return "?";
    }
    catch (...)
    {
        return "?";
    }
    return name;
}

}

class Plan
{
public:

    struct Candidate
    {
        int64_t index = -1;
        int64_t workspace_bytes = 0;
        string  name;
    };

    shared_ptr<fe::graph::Graph> graph;
    vector<Candidate> candidates;
    unordered_map<int64_t, void*> pack;
    bool uses_bias = false;
};

bool verbose() noexcept
{
    static const bool on = env_flag_enabled("OPENNN_CUDNN_MATMUL_VERBOSE", false);
    return on;
}

namespace
{

struct EpilogueFeatures
{
    bool bias = false;
    bool relu = false;
};

optional<EpilogueFeatures> supported_problem(const Problem& problem)
{
    if(!enabled() || !house::frontend_enabled() || house::device_sm_version() < 800)
        return nullopt;
    if(!problem.beta_is_zero || problem.dtype_a != CUDA_R_16BF
       || problem.dtype_b != CUDA_R_16BF || problem.out_dtype != CUDA_R_16BF)
        return nullopt;
    if(problem.m <= 0 || problem.n <= 0 || problem.k <= 0
       || problem.m < minimum_dimension() || problem.n < minimum_dimension()
       || problem.k < minimum_dimension() || problem.m % 8 != 0 || problem.k % 8 != 0)
        return nullopt;
    if(2.0 * double(problem.m) * double(problem.n) * double(problem.k)
       < minimum_gflop() * 1e9)
        return nullopt;

    if(problem.epilogue == CUBLASLT_EPILOGUE_DEFAULT) return EpilogueFeatures{};
    if(problem.epilogue == CUBLASLT_EPILOGUE_BIAS) return EpilogueFeatures{true, false};
    if(problem.epilogue == CUBLASLT_EPILOGUE_RELU) return EpilogueFeatures{false, true};
    if(problem.epilogue == CUBLASLT_EPILOGUE_RELU_BIAS)
        return EpilogueFeatures{true, true};
    return nullopt;
}

struct MatmulLayout
{
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t left_sample;
    int64_t left_inner;
    int64_t right_inner;
    int64_t right_neuron;
    int64_t output_leading;
};

optional<MatmulLayout> make_layout(const Problem& problem)
{
    const int64_t m = problem.m;
    const int64_t n = problem.n;
    const int64_t k = problem.k;
    const int64_t lda = problem.lda ? problem.lda
                                    : (problem.transA == CUBLAS_OP_N ? m : k);
    const int64_t ldb = problem.ldb ? problem.ldb
                                    : (problem.transB == CUBLAS_OP_N ? k : n);
    const int64_t ldd = problem.ldd ? problem.ldd : m;
    if(lda % 8 != 0 || ldb % 8 != 0 || ldd % 8 != 0) return nullopt;

    return MatmulLayout{
        m, n, k,
        problem.transB == CUBLAS_OP_N ? ldb : 1,
        problem.transB == CUBLAS_OP_N ? 1 : ldb,
        problem.transA == CUBLAS_OP_N ? lda : 1,
        problem.transA == CUBLAS_OP_N ? 1 : lda,
        ldd
    };
}

int64_t batch_stride(int64_t first_extent, int64_t first_stride,
                     int64_t second_extent, int64_t second_stride)
{
    return max(first_extent * first_stride, second_extent * second_stride);
}

shared_ptr<fe::graph::Graph> build_graph(const MatmulLayout& layout,
                                         EpilogueFeatures epilogue,
                                         cudnnHandle_t handle)
{
    auto graph = make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    auto left = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("input").set_dim({1, layout.n, layout.k})
        .set_stride({batch_stride(layout.n, layout.left_sample,
                                  layout.k, layout.left_inner),
                     layout.left_sample, layout.left_inner}).set_uid(uid_left));
    auto right = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("weights").set_dim({1, layout.k, layout.m})
        .set_stride({batch_stride(layout.k, layout.right_inner,
                                  layout.m, layout.right_neuron),
                     layout.right_inner, layout.right_neuron}).set_uid(uid_right));
    auto value = graph->matmul(left, right, fe::graph::Matmul_attributes()
        .set_name("gemm").set_compute_data_type(fe::DataType_t::FLOAT));

    if(epilogue.bias)
    {
        auto bias = graph->tensor(fe::graph::Tensor_attributes()
            .set_name("bias").set_dim({1, 1, layout.m})
            .set_stride({layout.m, layout.m, 1}).set_uid(uid_bias));
        value = graph->pointwise(value, bias, fe::graph::Pointwise_attributes()
            .set_name("bias_add").set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    }
    if(epilogue.relu)
        value = graph->pointwise(value, fe::graph::Pointwise_attributes()
            .set_name("relu").set_mode(fe::PointwiseMode_t::RELU_FWD)
            .set_compute_data_type(fe::DataType_t::FLOAT));

    value->set_output(true).set_dim({1, layout.n, layout.m})
         .set_stride({batch_stride(layout.n, layout.output_leading, layout.m, 1),
                      layout.output_leading, 1})
         .set_data_type(fe::DataType_t::BFLOAT16).set_uid(uid_output);

    if(graph->validate().is_bad() || graph->build_operation_graph(handle).is_bad())
        return nullptr;
    if(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B,
                                      fe::HeurMode_t::FALLBACK}).is_bad())
        return nullptr;
    graph->deselect_workspace_greater_than(workspace_cap_bytes());
    return graph;
}

void build_candidates(Plan& plan, fe::graph::Graph& graph,
                      cudnnHandle_t handle, int64_t requested_index)
{
    const int64_t offered = graph.get_execution_plan_count();
    const int64_t first = requested_index < 0 ? 0 : requested_index;
    const int64_t last = requested_index < 0 ? offered
                                             : min(offered, requested_index + 1);
    for(int64_t index = first; index < last; ++index)
    {
        if(candidate_limit() > 0
           && int64_t(plan.candidates.size()) >= candidate_limit())
            break;
        try
        {
            if(graph.build_plan_at_index(handle, index).is_bad()) continue;
            int64_t bytes = 0;
            if(graph.get_workspace_size_plan_at_index(index, bytes).is_bad()
               || bytes > workspace_cap_bytes())
                continue;
            plan.candidates.push_back({index, bytes, plan_name(graph, index)});
        }
        catch(...)
        {
            device::reset_last_error();
        }
    }
}

bool plan_creation_allowed()
{
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if(cudaStreamIsCapturing(device::get_compute_stream(), &capture) == cudaSuccess
       && capture == cudaStreamCaptureStatusNone)
        return true;
    device::reset_last_error();
    return false;
}

}

Plan* create(const Problem& problem, int64_t plan_index) noexcept
{
    try
    {
        const optional<EpilogueFeatures> epilogue = supported_problem(problem);
        const optional<MatmulLayout> layout = epilogue ? make_layout(problem) : nullopt;
        if(!layout || !plan_creation_allowed()) return nullptr;

        const cudnnHandle_t handle = device::get_cudnn_handle();
        if(!handle) return nullptr;
        shared_ptr<fe::graph::Graph> graph = build_graph(*layout, *epilogue, handle);
        if(!graph) return nullptr;

        auto plan = make_unique<Plan>();
        plan->uses_bias = epilogue->bias;
        build_candidates(*plan, *graph, handle, plan_index);
        device::reset_last_error();
        if(plan->candidates.empty()) return nullptr;

        const int64_t offered = graph->get_execution_plan_count();
        plan->graph = std::move(graph);
        plan->pack = {{uid_left, nullptr}, {uid_right, nullptr}, {uid_output, nullptr}};
        if(epilogue->bias) plan->pack[uid_bias] = nullptr;

        if(verbose())
            logging::info() << format(
                "cudnn matmul {}x{}x{} epilogue {}: {} of {} engine configurations built\n",
                problem.m, problem.n, problem.k, int(problem.epilogue),
                plan->candidates.size(), offered);
        return plan.release();
    }
    catch(const exception& error)
    {
        device::reset_last_error();
        if(verbose()) logging::warning() << "cudnn matmul: declined ("
                                        << error.what() << ")\n";
        return nullptr;
    }
    catch(...)
    {
        device::reset_last_error();
        return nullptr;
    }
}

void destroy(Plan* plan) noexcept
{
    delete plan;
}

int candidate_count(const Plan* plan) noexcept
{
    return plan ? int(plan->candidates.size()) : 0;
}

CandidateInfo candidate(const Plan* plan, int index) noexcept
{
    if(!plan || index < 0 || index >= int(plan->candidates.size())) return {};
    const Plan::Candidate& value = plan->candidates[size_t(index)];
    return {value.index, size_t(value.workspace_bytes), value.name};
}

bool run(Plan* plan, int candidate,
         const void* a, const void* b, const void* bias, void* d,
         void* workspace) noexcept
{
    if (!plan || !plan->graph) return false;
    if (candidate < 0 || candidate >= int(plan->candidates.size())) return false;
    if (!a || !b || !d) return false;
    if (plan->uses_bias && !bias) return false;

    try
    {
        // The pack's keys were fixed in create(), so these are assignments
        // into an existing map and not allocations: this runs on every
        // forward pass of every layer that ends up here.
        plan->pack[uid_left]   = const_cast<void*>(b);
        plan->pack[uid_right]  = const_cast<void*>(a);
        plan->pack[uid_output] = d;
        if (plan->uses_bias) plan->pack[uid_bias] = const_cast<void*>(bias);

        const auto status = plan->graph->execute_plan_at_index(
            device::get_cudnn_handle(), plan->pack, workspace,
            plan->candidates[size_t(candidate)].index);

        if (status.is_bad())
        {
            device::reset_last_error();
            return false;
        }
        return true;
    }
    catch (...)
    {
        device::reset_last_error();
        return false;
    }
}

}

#endif
