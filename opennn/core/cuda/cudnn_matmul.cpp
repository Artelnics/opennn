//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C U D N N   M A T M U L   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/cuda/cudnn_matmul.h"
#include "opennn/core/log.h"

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/cudnn_frontend_utilities.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/string_utilities.h"

namespace opennn::cudnn_matmul
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
// autotune_lt_plan sizes its timing scratch to the largest workspace over ALL
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

Plan* create(const Problem& problem) noexcept
{
    return create(problem, -1);
}

Plan* create(const Problem& problem, const int64_t plan_index) noexcept
{
    try
    {
        if (!enabled()) return nullptr;
        if (!house::frontend_enabled()) return nullptr;

        // bf16 tensor cores start at sm_80. Below that the whole premise --
        // a bf16 contraction with an fp32 accumulator -- is emulated, and
        // nothing about the measurement transfers.
        if (house::device_sm_version() < 800) return nullptr;

        // --- what this path is allowed to serve -------------------------
        //
        // Every one of these is a refusal, not a fallback of last resort:
        // the cuBLASLt path already serves all of them correctly, and the
        // only reason to route a shape here is that somebody measured a
        // cuDNN engine winning it. bf16 in and out with an fp32 accumulator
        // is what was measured and verified against a cuBLASLt reference;
        // fp32 and int8 were not, so they are declined rather than guessed.
        if (!problem.beta_is_zero) return nullptr;
        if (problem.dtype_a != CUDA_R_16BF) return nullptr;
        if (problem.dtype_b != CUDA_R_16BF) return nullptr;
        if (problem.out_dtype != CUDA_R_16BF) return nullptr;

        bool with_bias = false;
        bool with_relu = false;
        switch (problem.epilogue)
        {
        case CUBLASLT_EPILOGUE_DEFAULT:                              break;
        case CUBLASLT_EPILOGUE_BIAS:      with_bias = true;          break;
        case CUBLASLT_EPILOGUE_RELU:      with_relu = true;          break;
        case CUBLASLT_EPILOGUE_RELU_BIAS: with_bias = with_relu = true; break;

        // Everything else -- the AUX epilogues that also write a
        // pre-activation or a ReLU bitmask, DRELU, GELU -- would need a
        // second graph output and a second correctness argument. Declined.
        default: return nullptr;
        }

        const int64_t m = problem.m;
        const int64_t n = problem.n;
        const int64_t k = problem.k;

        if (m <= 0 || n <= 0 || k <= 0) return nullptr;
        if (m < minimum_dimension() || n < minimum_dimension() || k < minimum_dimension())
            return nullptr;

        // Eight bf16 elements is sixteen bytes, which is what cuDNN's fast
        // matmul engines want on the contiguous extents. m and k are those
        // extents -- k along a sample's inputs, m along a row of neurons --
        // while n only counts rows and never has to divide anything. The row
        // strides are checked further down, once the leading dimensions are
        // resolved: an aligned extent behind an unaligned stride still puts
        // every row after the first on a bad boundary. cuDNN would decline
        // these itself, politely; declining here costs nothing and keeps a
        // ragged shape from paying for sixty build attempts that all fail.
        if (m % 8 != 0 || k % 8 != 0) return nullptr;

        if (2.0 * double(m) * double(n) * double(k) < minimum_gflop() * 1e9) return nullptr;

        // Building a plan finalizes cuDNN backend descriptors and can
        // allocate; neither may happen inside a stream capture. The caller
        // (get_lt_matmul_plan) is already barred from creating plans in
        // steady state by cuda_matmul_plan_creation_forbidden, so this is
        // the second gate, not the only one.
        cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(device::get_compute_stream(), &capturing) != cudaSuccess
            || capturing != cudaStreamCaptureStatusNone)
        {
            device::reset_last_error();
            return nullptr;
        }

        const cudnnHandle_t handle = device::get_cudnn_handle();
        if (!handle) return nullptr;

        // --- cuBLASLt's column-major statement, restated row-major -------
        //
        // cuBLASLt computes, in column-major storage,
        //
        //     D[m][n] = op(A)[m][k] * op(B)[k][n]
        //
        // and cuDNN's matmul contracts row-major tensors. Transposing the
        // whole statement turns one into the other:
        //
        //     D^T[n][m] = op(B)^T[n][k] * op(A)^T[k][m]
        //
        // so cuDNN's LEFT operand is cuBLASLt's B -- the batch of samples --
        // and cuDNN's RIGHT operand is cuBLASLt's A -- the weights. No data
        // moves: a column-major m x n matrix with leading dimension ld IS a
        // row-major n x m matrix with row stride ld, which is why D^T is the
        // destination OpenNN already has.
        //
        // The strides below are derived from transA, transB and the leading
        // dimensions rather than assumed, because assuming is exactly what
        // cost a debugging round on the probe. Worked through for the dense
        // benchmark's hidden layer, where linear_forward_lt_gpu calls with
        // m = neurons, n = samples, k = inputs, transA = transB = N and all
        // three leading dimensions derived:
        //
        //   left  (samples) dim {1, 8192, 1024} stride {8388608, 1024, 1}
        //   right (weights) dim {1, 1024, 1024} stride {1048576, 1024, 1}
        //   out             dim {1, 8192, 1024} stride {8388608, 1024, 1}
        //
        // which is byte-for-byte the graph cudnn_matmul_probe.cu built and
        // verified.
        //
        // NOTE FOR ANYONE HOLDING THE PROBE'S REPORT, because the two do not
        // agree and this one is right. Those weight strides are the literal
        // the probe labels "B-transposed", and NOTHING IS TRANSPOSED HERE.
        // The probe's "B-as-stored" sweep assumed OpenNN holds a Dense weight
        // matrix neuron-major, [neurons][inputs] with each neuron's inputs
        // contiguous. It does not. It holds [inputs][neurons]: Dense's own
        // expression writer indexes the matrix i * outputs_number + j with i
        // the input and j the neuron, and linear_forward_lt_gpu hands it to
        // cuBLASLt as operand A with transA = N and lda = m, which is the
        // same statement. So {k * m, lda, 1} is the layout the library
        // already has, reached with no copy at all; copying the probe's
        // "as-stored" literal instead would have contracted the transpose of
        // the weight matrix and returned a confidently wrong answer that no
        // status code anywhere would have reported.
        //
        // The recommendation in the probe's report therefore inverts. Its two
        // winners measured 189.60 us / 235.4 W ("as-stored") and 189.80 us /
        // 227.7 W ("transposed"), and the second one is the layout OpenNN
        // actually has. The library gets the cooler kernel for free, and the
        // 8 W that was supposed to be waiting behind a future weight-layout
        // change is already collected. The other variant is the one that
        // would now need a transposed copy, and it is hotter, so there is
        // nothing left there worth anyone's time.
        const int64_t lda = problem.lda ? int64_t(problem.lda)
                                        : (problem.transA == CUBLAS_OP_N ? m : k);
        const int64_t ldb = problem.ldb ? int64_t(problem.ldb)
                                        : (problem.transB == CUBLAS_OP_N ? k : n);
        const int64_t ldd = problem.ldd ? int64_t(problem.ldd) : m;

        if (lda % 8 != 0 || ldb % 8 != 0 || ldd % 8 != 0) return nullptr;

        // left = op(B)^T, dims {1, n, k}: stride along samples, then along k.
        const int64_t left_sample = (problem.transB == CUBLAS_OP_N) ? ldb : 1;
        const int64_t left_inner  = (problem.transB == CUBLAS_OP_N) ? 1   : ldb;

        // right = op(A)^T, dims {1, k, m}: stride along k, then along neurons.
        const int64_t right_inner  = (problem.transA == CUBLAS_OP_N) ? lda : 1;
        const int64_t right_neuron = (problem.transA == CUBLAS_OP_N) ? 1   : lda;

        const auto batch_stride = [](int64_t d1, int64_t s1, int64_t d2, int64_t s2)
        {
            return std::max(d1 * s1, d2 * s2);
        };

        auto graph = make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::BFLOAT16)
              .set_intermediate_data_type(fe::DataType_t::FLOAT)
              .set_compute_data_type(fe::DataType_t::FLOAT);

        auto left = graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("input")
                                  .set_dim({1, n, k})
                                  .set_stride({batch_stride(n, left_sample, k, left_inner),
                                               left_sample, left_inner})
                                  .set_uid(uid_left));

        auto right = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("weights")
                                   .set_dim({1, k, m})
                                   .set_stride({batch_stride(k, right_inner, m, right_neuron),
                                                right_inner, right_neuron})
                                   .set_uid(uid_right));

        auto value = graph->matmul(left, right, fe::graph::Matmul_attributes()
                                                .set_name("gemm")
                                                .set_compute_data_type(fe::DataType_t::FLOAT));

        if (with_bias)
        {
            // One value per neuron, the same m numbers under every one of the
            // n rows -- broadcast along rows, which is exactly what
            // CUBLASLT_EPILOGUE_BIAS does. bf16, because
            // get_lt_matmul_plan sets CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE to
            // the output type and linear_forward_lt_gpu stages an fp32 bias
            // into bf16 before the call.
            auto bias = graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("bias")
                                      .set_dim({1, 1, m})
                                      .set_stride({m, m, 1})
                                      .set_uid(uid_bias));

            value = graph->pointwise(value, bias, fe::graph::Pointwise_attributes()
                                                  .set_name("bias_add")
                                                  .set_mode(fe::PointwiseMode_t::ADD)
                                                  .set_compute_data_type(fe::DataType_t::FLOAT));
        }

        if (with_relu)
            value = graph->pointwise(value, fe::graph::Pointwise_attributes()
                                            .set_name("relu")
                                            .set_mode(fe::PointwiseMode_t::RELU_FWD)
                                            .set_compute_data_type(fe::DataType_t::FLOAT));

        value->set_output(true)
              .set_dim({1, n, m})
              .set_stride({batch_stride(n, ldd, m, 1), ldd, 1})
              .set_data_type(fe::DataType_t::BFLOAT16)
              .set_uid(uid_output);

        if (graph->validate().is_bad())               return nullptr;
        if (graph->build_operation_graph(handle).is_bad()) return nullptr;

        // All three heuristic modes, as in the probe. Mode A alone is what
        // the rest of the library asks for, and for this shape mode A's
        // first pick runs at 226 us where the best configuration runs at
        // 189.6; the engine that wins is reached only by enumerating.
        if (graph->create_execution_plans({fe::HeurMode_t::A,
                                           fe::HeurMode_t::B,
                                           fe::HeurMode_t::FALLBACK}).is_bad())
            return nullptr;

        graph->deselect_workspace_greater_than(workspace_cap_bytes());

        auto plan = make_unique<Plan>();
        plan->uses_bias = with_bias;

        const int64_t offered = graph->get_execution_plan_count();
        const int64_t limit = candidate_limit();

        // A recorded winner is rebuilt alone: building every configuration
        // cuDNN offers is the search, and the search has been done.
        const int64_t first = plan_index < 0 ? 0 : plan_index;
        const int64_t last  = plan_index < 0 ? offered : min(offered, plan_index + 1);

        for (int64_t index = first; index < last; ++index)
        {
            if (limit > 0 && int64_t(plan->candidates.size()) >= limit) break;

            // Every failure here is ordinary: cuDNN offers configurations it
            // cannot build for a given shape and says so. Swallow it and
            // move to the next, exactly as the probe does, so that "no
            // engine was any good" and "no engine ran" stay distinguishable
            // through candidate_count() rather than through silence.
            try
            {
                if (graph->build_plan_at_index(handle, index).is_bad()) continue;
            }
            catch (...)
            {
                device::reset_last_error();
                continue;
            }

            int64_t bytes = 0;
            try
            {
                if (graph->get_workspace_size_plan_at_index(index, bytes).is_bad()) continue;
            }
            catch (...)
            {
                device::reset_last_error();
                continue;
            }

            if (bytes > workspace_cap_bytes()) continue;

            plan->candidates.push_back({index, bytes, plan_name(*graph, index)});
        }

        device::reset_last_error();

        if (plan->candidates.empty()) return nullptr;

        plan->graph = std::move(graph);
        plan->pack[uid_left]   = nullptr;
        plan->pack[uid_right]  = nullptr;
        plan->pack[uid_output] = nullptr;
        if (with_bias) plan->pack[uid_bias] = nullptr;

        if (verbose())
            logging::info() << format("cudnn matmul {}x{}x{} epilogue {}: {} of {} engine "
                           "configurations built\n",
                           problem.m, problem.n, problem.k, int(problem.epilogue),
                           plan->candidates.size(), offered);

        return plan.release();
    }
    catch (const exception& e)
    {
        device::reset_last_error();
        if (verbose()) logging::warning() << "cudnn matmul: declined (" << e.what() << ")\n";
        return nullptr;
    }
    catch (...)
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

int64_t candidate_plan_index(const Plan* plan, int candidate) noexcept
{
    if (!plan || candidate < 0 || candidate >= int(plan->candidates.size())) return -1;
    return plan->candidates[size_t(candidate)].index;
}

size_t candidate_workspace_bytes(const Plan* plan, int candidate) noexcept
{
    if (!plan || candidate < 0 || candidate >= int(plan->candidates.size())) return 0;
    return size_t(plan->candidates[size_t(candidate)].workspace_bytes);
}

string candidate_name(const Plan* plan, int candidate)
{
    if (!plan || candidate < 0 || candidate >= int(plan->candidates.size())) return "?";
    return plan->candidates[size_t(candidate)].name;
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

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
