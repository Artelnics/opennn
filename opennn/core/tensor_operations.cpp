//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   O P E R A T I O N S   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/tensor_operations.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/cuda/cutlass_narrow_gemm.cuh"
#include "opennn/core/profiler.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/cuda/kernel_activation.cuh"
#include "opennn/core/cuda/kernel_normalization.cuh"
#include "opennn/core/cuda/kernel_cast.cuh"
#include "opennn/core/cuda/kernel_quantization.cuh"
#include "opennn/core/cuda/kernel_tensor.cuh"

#include <atomic>
#include <mutex>
#include <omp.h>

#ifdef EIGEN_USE_MKL_ALL
#include <mkl_cblas.h>
#include <mkl_service.h>
#include <mkl_vml.h>
#endif

namespace opennn
{

template<bool FuseRelu>
static void add_bias_span(float* const __restrict y, const float* const __restrict b,
                          Index first, Index last, Index columns)
{
    for (Index i = first; i < last; ++i)
    {
        float* const row = y + i * columns;

        for (Index j = 0; j < columns; ++j)
        {
            const float value = row[j] + b[j];
            if constexpr (FuseRelu) row[j] = value > 0.0f ? value : 0.0f;
            else                    row[j] = value;
        }
    }
}

// Column sums, threaded. The serial `colwise().sum()` Eigen gives you reads
// the whole delta -- 16 MiB at batch 4096 by 1024 -- at one core's bandwidth,
// about 6.7 GB/s here, which made the bias gradient 9.5% of a training step
// for arithmetic that is pure streaming. Each block reduces into its own
// scratch row and one serial pass adds those up: `columns` is a layer width,
// so both the scratch and the final reduction are negligible beside the read
// they parallelise.
static void column_sums(const float* const __restrict values, float* const __restrict sums,
                        Index rows, Index columns, Index first, Index last)
{
    for (Index j = 0; j < columns; j++) sums[j] = 0.0f;

    for (Index i = first; i < last; i++)
    {
        const float* const row = values + i * columns;

        for (Index j = 0; j < columns; j++) sums[j] += row[j];
    }
}

// Eigen unless the application asked for MKL. A plain build has no MKL paths
// to reach, and a build that does have them still waits to be told.
static bool mkl_dispatch_enabled()
{
    return Configuration::instance().get_blas() == Blas::Mkl;
}

static Index gemm_block_rows(Index rows, Index threads)
{
    static const Index requested_rows = []
    {
        const char* const requested = getenv("OPENNN_GEMM_BLOCK");

        return requested ? Index(atoll(requested)) : 0;
    }();

    if (requested_rows > 0) return requested_rows;

    // What matters is the block *count*, not the block height. Sweeping both
    // across batch 1024 to 8192, the optimum sits at two blocks per thread
    // every time -- 32 rows at batch 1024, 64 at 2048, 128 at 4096. Fixing the
    // height instead costs 42% at batch 1024 and 36% at 2048, which is what
    // tuning the constant against a single batch size buys you.
    //
    // Powers of two on purpose. Batch sizes are powers of two, so these divide
    // the batch evenly, and a ragged final block costs far more than the
    // imbalance it saves: 192 rows on a 4096-row batch loses 20% against 128.
    const Index wanted = max<Index>(8, rows / max<Index>(1, threads * 2));

    Index height = 8;

    while (height * 2 <= wanted) height *= 2;

    return max<Index>(8, min(height, rows));
}

#ifdef EIGEN_USE_MKL_ALL


static bool try_activation_forward(TensorView& output, ActivationFunction function)
{
    if (!mkl_dispatch_enabled()) return false;

    if (function != ActivationFunction::Tanh || !output.is_fp32()) return false;

    float* values = output.as<float>();
    const int size = to_int(output.size());

    vsTanh(size, values, values);

    return true;
}

static atomic<long long> mkl_linear_calls{0};
static atomic<long long> mkl_linear_refusals{0};

struct MklLinearReport
{
    ~MklLinearReport()
    {
        if (getenv("OPENNN_MKL_REPORT"))
            fprintf(stderr, "mkl_linear_forward_calls=%lld refusals=%lld\n",
                    mkl_linear_calls.load(), mkl_linear_refusals.load());
    }
};

static MklLinearReport mkl_linear_report;

enum class GemmParallelism { Mkl, Pool, Omp, Contract };

static GemmParallelism gemm_parallelism()
{
    static const GemmParallelism mode = []
    {
        const char* const requested = getenv("OPENNN_GEMM_MODE");
        if (!requested) return GemmParallelism::Contract;
        if (string(requested) == "pool") return GemmParallelism::Pool;
        if (string(requested) == "mkl") return GemmParallelism::Mkl;
        if (string(requested) == "omp") return GemmParallelism::Omp;
        return GemmParallelism::Contract;
    }();

    return mode;
}

struct BiasRelu
{
    const float* bias = nullptr;
    bool relu = true;

    template<typename Index, typename Scalar>
    void operator()(const Eigen::internal::blas_data_mapper<Scalar, Index, Eigen::ColMajor>& output,
                    const Eigen::TensorContractionParams& params,
                    Index i, Index j, Index rows, Index columns) const
    {
        EIGEN_UNUSED_VARIABLE(params);
        EIGEN_UNUSED_VARIABLE(j);

        for (Index column = 0; column < columns; ++column)
            for (Index row = 0; row < rows; ++row)
            {
                const float value = output(row, column) + bias[i + row];
                output(row, column) = relu && value < 0.0f ? 0.0f : value;
            }
    }
};

class ContractionScratch final : public Eigen::Allocator
{
public:

    ~ContractionScratch() override
    {
        for (const auto& block : blocks)
            Eigen::internal::aligned_free(block.first);
    }

    void* allocate(size_t bytes) const override
    {
        const lock_guard<mutex> lock(guard);

        for (size_t i = 0; i < free_list.size(); ++i)
            if (blocks.at(free_list[i]) >= bytes)
            {
                void* const reused = free_list[i];
                free_list[i] = free_list.back();
                free_list.pop_back();

                return reused;
            }

        void* const fresh = Eigen::internal::aligned_malloc(bytes);
        if (fresh) blocks.emplace(fresh, bytes);

        return fresh;
    }

    void deallocate(void* buffer) const override
    {
        if (!buffer) return;

        const lock_guard<mutex> lock(guard);

        if (blocks.count(buffer)) free_list.push_back(buffer);
        else                      Eigen::internal::aligned_free(buffer);
    }

private:

    mutable mutex guard;
    mutable unordered_map<void*, size_t> blocks;
    mutable vector<void*> free_list;
};

static Eigen::ThreadPoolDevice contraction_device()
{
    static ContractionScratch scratch;
    return Eigen::ThreadPoolDevice(get_device().getPool(),
                                   get_device().numThreads(),
                                   &scratch);
}

static void contract_linear_forward(int m, int n, int k, const float* a, const float* b, float* c,
                                    const float* bias, bool fuse_relu)
{
    using RowTensor = Eigen::Tensor<float, 2, Eigen::RowMajor>;

    const Eigen::TensorMap<const RowTensor> left(a, m, k);
    const Eigen::TensorMap<const RowTensor> right(b, k, n);
    Eigen::TensorMap<RowTensor> out(c, m, n);

    const Eigen::array<Eigen::IndexPair<int>, 1> dimensions = {Eigen::IndexPair<int>(1, 0)};

    out.device(contraction_device()) = left.contract(right, dimensions, BiasRelu{bias, fuse_relu});
}


static double gemm_contract_flops()
{
    static const double flops = []
    {
        const char* const requested = getenv("OPENNN_GEMM_CONTRACT_FLOPS");
        const double requested_flops = requested ? atof(requested) : -1.0;

        return requested_flops >= 0.0 ? requested_flops : 8.0e9;
    }();

    return flops;
}

static double gemm_min_flops()
{
    static const double flops = []
    {
        const char* const requested = getenv("OPENNN_GEMM_MIN_FLOPS");
        const double requested_flops = requested ? atof(requested) : -1.0;

        return requested_flops >= 0.0 ? requested_flops : 64.0 * 1024.0 * 1024.0;
    }();

    return flops;
}

static Index gemm_min_output()
{
    static const Index elements = []
    {
        const char* const requested = getenv("OPENNN_GEMM_MIN_OUTPUT");
        const Index requested_elements = requested ? Index(atoll(requested)) : -1;

        return requested_elements >= 0 ? requested_elements : Index(256) * 1024;
    }();

    return elements;
}

static int gemm_guided_setting()
{
    static const int setting = []
    {
        const char* const requested = getenv("OPENNN_GEMM_GUIDED");

        return requested ? atoi(requested) : -1;
    }();

    return setting;
}

static Index gemm_mkl_threads()
{
    static const Index threads = []
    {
        const char* const requested = getenv("OPENNN_GEMM_MKL_THREADS");
        const Index requested_threads = requested ? Index(atoll(requested)) : 0;

        return requested_threads > 0 ? requested_threads : 1;
    }();

    return threads;
}

// Runs `body` on `workers` threads of a full-size team. Asking for a smaller
// team (`num_threads(workers)`) would make libgomp shrink its pool to that size
// and grow it back with `pthread_create` at the next full region -- see
// Backend::set_threads_number. The threads left over go straight to the
// barrier.
template<typename Body>
static void parallel_workers(Index workers, Body&& body)
{
    #pragma omp parallel
    {
        if (Index(omp_get_thread_num()) < workers) body();
    }
}

static void sgemm_rows(int m, int n, int k, const float* a, const float* b, float* c)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0f, a, k, b, n, 0.0f, c, n);
}

struct PackedWeights
{
    float* data = nullptr;
    size_t bytes = 0;

    ~PackedWeights() { if (data) mkl_free(data); }

    float* reserve(size_t wanted)
    {
        if (wanted <= bytes) return data;

        if (data) mkl_free(data);
        data = static_cast<float*>(mkl_malloc(wanted, 64));
        bytes = data ? wanted : 0;

        return data;
    }
};

static bool gemm_pack_weights(Index)
{
    static const int requested = []
    {
        const char* const setting = getenv("OPENNN_GEMM_PACK");

        return setting ? atoi(setting) : -1;
    }();

    if (requested >= 0) return requested != 0;

    // Off. Packing B once per call and running `cblas_sgemm_compute` per block
    // measures slower than letting each block call `sgemm_rows` on the
    // unpacked weights, at every block height tried -- 4.9% slower at 64 rows.
    // The weights here are small enough that MKL's own copy costs less than
    // the pack it would replace. `OPENNN_GEMM_PACK=1` brings it back.
    return false;
}

template<typename Work>
static bool blocked_rows(Index rows, double work_flops, Work&& work)
{
    const Index threads = get_device().numThreads();
    const Index block_rows = gemm_block_rows(rows, threads);
    const Index blocks = (rows + block_rows - 1) / block_rows;

    const Index workers = max<Index>(1, min<Index>(threads / gemm_mkl_threads(), blocks / 2));

    if (gemm_parallelism() == GemmParallelism::Mkl
        || workers < 2
        || omp_in_parallel()
        || work_flops < gemm_min_flops())
        return false;

    atomic<Index> next_block{0};

    parallel_workers(workers, [&]
    {
        mkl_set_num_threads_local(to_int(gemm_mkl_threads()));

        for (Index block = next_block++; block < blocks; block = next_block++)
        {
            const Index first = block * block_rows;

            work(first, min<Index>(block_rows, rows - first));
        }
    });

    mkl_set_num_threads_local(0);

    return true;
}

static bool backward_input_delta(int rows, int in_features, int out_features,
                                 const float* delta, const float* weights, float* input_delta,
                                 float beta)
{
    PROFILE_SCOPE_HOST("cpu:bwd_input_delta");

    const auto slice = [&](Index first, Index count)
    {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    to_int(count), in_features, out_features, 1.0f,
                    delta + first * Index(out_features), out_features,
                    weights, out_features,
                    beta, input_delta + first * Index(in_features), in_features);
    };

    const double flops = double(rows) * double(in_features) * double(out_features);

    if (blocked_rows(Index(rows), flops, slice)) return true;

    slice(0, Index(rows));

    return true;
}

static bool backward_weight_gradient(int rows, int in_features, int out_features,
                                     const float* input, const float* delta, float* weight_gradient)
{
    PROFILE_SCOPE_HOST("cpu:bwd_weight_gradient");

    const auto tile = [&](Index m_first, Index m_count, Index n_first, Index n_count)
    {
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    to_int(m_count), to_int(n_count), rows, 1.0f,
                    input + m_first, in_features,
                    delta + n_first, out_features,
                    0.0f, weight_gradient + m_first * Index(out_features) + n_first, out_features);
    };

    const double flops = double(rows) * double(in_features) * double(out_features);
    const Index threads = get_device().numThreads();
    const Index workers = max<Index>(1, threads / gemm_mkl_threads());

    // The batch is this product's reduction axis, so unlike the forward GEMM
    // it cannot be the axis we split. Splitting only `in_features` -- what this
    // did before -- hands every block the whole delta as B, so the work grows
    // with the block count instead of dividing by it. Splitting both output
    // axes keeps the tile count while shrinking what each tile reads: for a
    // 1024x1024 gradient over a 4096-row batch, an 8x4 grid touches 201 MiB
    // where 32x1 touched 554 MiB.
    constexpr Index smallest_tile = 32;
    const Index tiles_wanted = max<Index>(1, workers * 2);

    Index m_blocks = 1;
    Index n_blocks = 1;

    while (m_blocks * n_blocks < tiles_wanted)
    {
        const bool m_has_room = Index(in_features) / (m_blocks * 2) >= smallest_tile;
        const bool n_has_room = Index(out_features) / (n_blocks * 2) >= smallest_tile;

        if (m_has_room && (m_blocks <= n_blocks || !n_has_room)) m_blocks *= 2;
        else if (n_has_room)                                     n_blocks *= 2;
        else break;
    }

    const Index tiles = m_blocks * n_blocks;

    if (gemm_parallelism() == GemmParallelism::Mkl
        || omp_in_parallel()
        || tiles < 2
        || flops < gemm_min_flops())
    {
        tile(0, Index(in_features), 0, Index(out_features));

        return true;
    }

    const Index m_block = (Index(in_features) + m_blocks - 1) / m_blocks;
    const Index n_block = (Index(out_features) + n_blocks - 1) / n_blocks;

    atomic<Index> next_tile{0};

    parallel_workers(min<Index>(workers, tiles), [&]
    {
        mkl_set_num_threads_local(to_int(gemm_mkl_threads()));

        for (Index index = next_tile++; index < tiles; index = next_tile++)
        {
            const Index m_first = (index / n_blocks) * m_block;
            const Index n_first = (index % n_blocks) * n_block;

            if (m_first >= Index(in_features) || n_first >= Index(out_features)) continue;

            tile(m_first, min<Index>(m_block, Index(in_features) - m_first),
                 n_first, min<Index>(n_block, Index(out_features) - n_first));
        }
    });

    mkl_set_num_threads_local(0);

    return true;
}

static void blocked_linear_forward(int m, int n, int k, const float* a, const float* b, float* c,
                                   const float* bias, bool fuse_relu)
{
    const bool has_bias = bias != nullptr;

    if (gemm_parallelism() == GemmParallelism::Contract
        && has_bias
        && Index(k) >= 64
        && double(m) * double(n) * double(k) >= gemm_contract_flops()
        && Index(m) * Index(n) >= gemm_min_output())
        return contract_linear_forward(m, n, k, a, b, c, bias, fuse_relu);

    const Index threads = get_device().numThreads();
    const Index block_rows = gemm_block_rows(Index(m), threads);
    const Index blocks = (Index(m) + block_rows - 1) / block_rows;

    const Index workers = max<Index>(1, min<Index>(threads / gemm_mkl_threads(), blocks / 2));

    const auto epilogue = [&](Index first, Index last)
    {
        if (!bias) return;

        if (fuse_relu) add_bias_span<true>(c, bias, first, last, n);
        else           add_bias_span<false>(c, bias, first, last, n);
    };

    if (gemm_parallelism() == GemmParallelism::Mkl
        || workers < 2
        || omp_in_parallel()
        || Index(m) * Index(n) < gemm_min_output())
    {
        sgemm_rows(m, n, k, a, b, c);
        return epilogue(0, m);
    }

    atomic<Index> next_block{0};

    thread_local PackedWeights panel;
    const float* packed = nullptr;

    if (gemm_pack_weights(block_rows) && blocks > 1)
    {
        const int packed_rows = to_int(block_rows);
        float* const buffer = panel.reserve(cblas_sgemm_pack_get_size(CblasBMatrix, packed_rows, n, k));

        if (buffer)
        {
            cblas_sgemm_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans,
                             packed_rows, n, k, 1.0f, b, n, buffer);
            packed = buffer;
        }
    }

    const auto take_blocks = [&]
    {
        mkl_set_num_threads_local(to_int(gemm_mkl_threads()));

        for (Index block = next_block++; block < blocks; block = next_block++)
        {
            const Index first = block * block_rows;
            const int rows = to_int(min<Index>(block_rows, Index(m) - first));

            if (packed && Index(rows) == block_rows)
                cblas_sgemm_compute(CblasRowMajor, CblasNoTrans, CblasPacked,
                                    rows, n, k, a + first * Index(k), k, packed, n,
                                    0.0f, c + first * Index(n), n);
            else
                sgemm_rows(rows, n, k, a + first * Index(k), b, c + first * Index(n));

            epilogue(first, first + rows);
        }
    };

    atomic<Index> next_row{0};

    const auto take_share = [&]
    {
        mkl_set_num_threads_local(to_int(gemm_mkl_threads()));

        for (;;)
        {
            Index first = next_row.load(memory_order_relaxed);
            Index rows = 0;

            do
            {
                if (first >= Index(m)) return;
                rows = max<Index>(8, (Index(m) - first) / (2 * workers));
                rows = min<Index>(rows, Index(m) - first);
            }
            while (!next_row.compare_exchange_weak(first, first + rows));

            sgemm_rows(to_int(rows), n, k, a + first * Index(k), b, c + first * Index(n));
            epilogue(first, first + rows);
        }
    };

    const bool guided = gemm_guided_setting() >= 0 ? gemm_guided_setting() == 1
                                                   : packed == nullptr;

    if (gemm_parallelism() == GemmParallelism::Omp)
    {
        if (guided) parallel_workers(workers, take_share);
        else        parallel_workers(workers, take_blocks);
    }
    else
    {
        const Eigen::TensorOpCost cost(0.0, 0.0, double(block_rows) * double(n) * double(k));

        get_device().parallelFor(workers, cost, [&](Index, Index) { take_blocks(); });
    }

    mkl_set_num_threads_local(0);
}

static bool try_linear_forward(const TensorView& input,
                                const TensorView& weights,
                                const TensorView& bias,
                                TensorView& output,
                                bool fuse_relu)
{
    if (!mkl_dispatch_enabled()) return false;

    ++mkl_linear_refusals;
    if (!input.is_fp32()
        || !weights.is_fp32()
        || !bias.is_fp32()
        || !output.is_fp32()
        || input.get_shape().get_rank() == 0
        || weights.get_shape().get_rank() != 2
        || bias.get_shape().get_rank() != 1)
        return false;

    const Index input_columns = input.get_shape().back();
    const Index output_columns = weights.get_shape().back();

    if (input_columns <= 0
        || output_columns <= 0
        || input.size() % input_columns != 0
        || weights.get_shape()[0] != input_columns
        || bias.size() != output_columns)
        return false;

    const Index rows = input.size() / input_columns;

    if (rows <= 0 || output.size() != rows * output_columns)
        return false;

    const int m = to_int(rows);
    const int n = to_int(output_columns);
    const int k = to_int(input_columns);

    {
    PROFILE_SCOPE_HOST(n >= 1024 && k >= 1024 ? "cpu:sgemm_wide"
                     : (k < 64 ? "cpu:sgemm_thin_k" : "cpu:sgemm_thin_n"));
    blocked_linear_forward(m, n, k, input.as<float>(), weights.as<float>(),
                           output.as<float>(), bias.as<float>(), fuse_relu);
    }
    --mkl_linear_refusals;
    ++mkl_linear_calls;
    return true;
}

static bool try_linear_backward(const TensorView& output_delta, const TensorView& input,
                                const TensorView& weights, const TensorView& weight_gradient,
                                TensorView& input_delta, bool accumulate, const TensorView* addend)
{
    if (!mkl_dispatch_enabled()) return false;

    static const bool eigen_backward = []
    {
        const char* const requested = getenv("OPENNN_BACKWARD");

        return requested && string(requested) == "eigen";
    }();

    if (eigen_backward
        || addend
        || !output_delta.is_fp32() || !input.is_fp32() || !weights.is_fp32()
        || !weight_gradient.is_fp32()
        || weights.get_shape().get_rank() != 2)
        return false;

    const Index out_features = weights.get_shape().back();
    const Index in_features = weights.get_shape()[0];

    if (in_features <= 0 || out_features <= 0
        || output_delta.size() % out_features != 0
        || input.size() % in_features != 0
        || weight_gradient.size() != in_features * out_features)
        return false;

    const Index rows = output_delta.size() / out_features;

    if (rows <= 0 || input.size() != rows * in_features) return false;

    const bool wants_input_delta = input_delta.get_data() && !input_delta.empty();

    if (wants_input_delta
        && (!input_delta.is_fp32() || input_delta.size() != rows * in_features))
        return false;

    backward_weight_gradient(to_int(rows), to_int(in_features), to_int(out_features),
                             input.as<float>(), output_delta.as<float>(), weight_gradient.as<float>());

    if (wants_input_delta)
        backward_input_delta(to_int(rows), to_int(in_features), to_int(out_features),
                             output_delta.as<float>(), weights.as<float>(), input_delta.as<float>(),
                             accumulate ? 1.0f : 0.0f);

    return true;
}

#else

static bool try_activation_forward(TensorView&, ActivationFunction)  { return false; }
static bool try_linear_forward(const TensorView&, const TensorView&,
                               const TensorView&, TensorView&, bool) { return false; }
static bool try_linear_backward(const TensorView&, const TensorView&, const TensorView&,
                                const TensorView&, TensorView&, bool, const TensorView*) { return false; }

#endif

bool blas_mkl_available()
{
#ifdef EIGEN_USE_MKL_ALL
    return true;
#else
    return false;
#endif
}

const EnumMap<ActivationFunction>& activation_function_map()
{
    static const EnumMap<ActivationFunction> map{
        {ActivationFunction::Identity,  "Identity"},
        {ActivationFunction::Sigmoid,   "Sigmoid"},
        {ActivationFunction::Tanh,      "Tanh"},
        {ActivationFunction::ReLU,      "ReLU"},
        {ActivationFunction::Softmax,   "Softmax"},
        {ActivationFunction::LeakyReLU, "LeakyReLU"},
        {ActivationFunction::GELU,      "GELU"},
        {ActivationFunction::GELUTanh,  "GELUTanh"},
        {ActivationFunction::SiLU,      "SiLU"},
        {ActivationFunction::SiLU,      "Swish"},

        {ActivationFunction::Identity,  "Linear"},
        {ActivationFunction::Sigmoid,   "Logistic"},
        {ActivationFunction::Tanh,      "HyperbolicTangent"},
        {ActivationFunction::ReLU,      "RectifiedLinear"},
        {ActivationFunction::ReLU,      "ScaledExponentialLinear"}
    };
    return map;
}

bool activation_needs_input(ActivationFunction function)
{
    return is_one_of(function, ActivationFunction::GELU,
                     ActivationFunction::GELUTanh, ActivationFunction::SiLU);
}

const string& activation_function_to_string(ActivationFunction function)
{
    return activation_function_map().to_string(function);
}

ActivationFunction activation_function_from_string(const string& name)
{
    return activation_function_map().from_string(name);
}

#define OPENNN_GPU_OPS(X) \
    X(copy_gpu, (const TensorView&, TensorView&)) \
    X(add_gpu, (const TensorView&, const TensorView&, TensorView&)) \
    X(multiply_gpu, (const TensorView&, bool, const TensorView&, bool, TensorView&, float, float)) \
    X(softmax_gpu, (TensorView&)) \
    X(softmax_backward_gpu, (const TensorView&, TensorView&, float)) \
    X(activation_forward_gpu, (TensorView&, ActivationFunction)) \
    X(activation_backward_gpu, (const TensorView&, TensorView&, ActivationFunction)) \
    X(linear_forward_gpu, (const TensorView&, const TensorView&, const TensorView&, TensorView&, cublasLtEpilogue_t, TensorView*, const TensorView&, ActivationFunction)) \
    X(linear_backward_gpu, (const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, TensorView&, bool, const LinearBackwardOptions&))

#define OPENNN_DECLARE_GPU_OP(name, sig) static void name sig;
OPENNN_GPU_OPS(OPENNN_DECLARE_GPU_OP)
#undef OPENNN_DECLARE_GPU_OP

static void require_tensor(const TensorView& tensor, string_view operation, string_view role)
{
    const Shape& shape = tensor.get_shape();
    throw_if(shape.empty(), "{}: {} must have a shape.", operation, role);
    throw_if(any_of(shape.begin(), shape.end(), [](Index dim) { return dim < 0; }),
             "{}: {} has a negative dimension.", operation, role);
    throw_if(tensor.get_device() == Device::Auto, "{}: {} has unresolved device metadata.", operation, role);
    throw_if(tensor.get_type() == Type::Auto, "{}: {} has unresolved dtype metadata.", operation, role);
    throw_if(tensor.size() > 0 && !tensor.get_data(), "{}: {} has no storage.", operation, role);
}

static void require_same_device(const TensorView& reference, const TensorView& tensor,
                                string_view operation)
{
    throw_if(reference.get_device() != tensor.get_device(),
             "{}: all tensors must be on the same device.", operation);
}

static void require_same_type(const TensorView& reference, const TensorView& tensor,
                              string_view operation)
{
    throw_if(reference.get_type() != tensor.get_type(),
             "{}: tensor dtypes are incompatible.", operation);
}

static void require_same_shape(const TensorView& reference, const TensorView& tensor,
                               string_view operation)
{
    throw_if(reference.get_shape() != tensor.get_shape(),
             "{}: tensor shapes are incompatible.", operation);
}

static void require_optional_tensor(const TensorView& reference, const TensorView& tensor,
                                    string_view operation, string_view role)
{
    if (tensor.empty()) return;
    require_tensor(tensor, operation, role);
    require_same_device(reference, tensor, operation);
}

static void require_fp32_or_bf16(const TensorView& tensor, string_view operation, string_view role)
{
    throw_if(!tensor.is_fp32() && !tensor.is_bf16(),
             "{}: {} must use FP32 or BF16 storage.", operation, role);
}

static void require_int8_linear(const TensorView& input, const TensorView& output,
                                const TensorView& weight_scale, string_view operation)
{
    throw_if(!input.is_cuda() || !input.is_bf16() || !output.is_bf16(),
             "{}: INT8 weights require CUDA BF16 activations.", operation);
    throw_if(weight_scale.empty() || !weight_scale.is_fp32()
             || weight_scale.get_shape().get_rank() != 1
             || weight_scale.size() != output.get_shape().back(),
             "{}: INT8 weights require one FP32 scale per output feature.", operation);
}

static void require_cpu_fp32(const TensorView& tensor, string_view operation, string_view role)
{
    throw_if(tensor.get_device() != Device::CPU || !tensor.is_fp32(),
             "{}: CPU {} must use FP32 storage.", operation, role);
}

static Index matrix_count(const TensorView& tensor)
{
    const Shape& shape = tensor.get_shape();
    const size_t rank = shape.get_rank();
    return tensor.size() / (shape[rank - 2] * shape[rank - 1]);
}

static void require_matching_linear_prefix(const TensorView& input, const TensorView& output,
                                           string_view operation)
{
    const Shape& input_shape = input.get_shape();
    const Shape& output_shape = output.get_shape();

    throw_if(input_shape.get_rank() != output_shape.get_rank(),
             "{}: input and output ranks do not match.", operation);
    for (size_t i = 0; i + 1 < input_shape.get_rank(); ++i)
        throw_if(input_shape[i] != output_shape[i],
                 "{}: input and output leading dimensions do not match.", operation);
}

static void validate_linear_io(const TensorView& input, const TensorView& weights,
                               const TensorView& output, bool transposed_weights,
                               string_view operation)
{
    require_tensor(input, operation, "input");
    require_tensor(weights, operation, "weights");
    require_tensor(output, operation, "output");
    require_same_device(input, weights, operation);
    require_same_device(input, output, operation);

    const Shape& input_shape = input.get_shape();
    const Shape& weights_shape = weights.get_shape();
    const Shape& output_shape = output.get_shape();

    throw_if(input_shape.get_rank() < 1, "{}: input rank must be at least one.", operation);
    throw_if(weights_shape.get_rank() != 2, "{}: weights must be a matrix.", operation);
    throw_if(output_shape.get_rank() < 1, "{}: output rank must be at least one.", operation);

    const Index input_features  = input_shape.back();
    const Index weight_inputs   = transposed_weights ? weights_shape[1] : weights_shape[0];
    const Index output_features = transposed_weights ? weights_shape[0] : weights_shape[1];
    throw_if(input_features <= 0 || output_features <= 0,
             "{}: feature dimensions must be positive.", operation);
    throw_if(weight_inputs != input_features,
             "{}: input and weight feature dimensions do not match.", operation);
    throw_if(output_shape.back() != output_features,
             "{}: output feature dimension does not match the weights.", operation);
    require_matching_linear_prefix(input, output, operation);
}

static void validate_linear_types(const TensorView& input, const TensorView& weights,
                                  const TensorView& output, string_view operation)
{
    if (!input.is_cuda())
    {
        require_cpu_fp32(input, operation, "input");
        require_cpu_fp32(weights, operation, "weights");
        return require_cpu_fp32(output, operation, "output");
    }

    require_fp32_or_bf16(input, operation, "input");
    require_fp32_or_bf16(output, operation, "output");
    throw_if(!weights.is_int8() && weights.get_type() != output.get_type(),
             "{}: non-quantized weights and output must use the same dtype.", operation);
}

void copy(const TensorView& source, TensorView& destination)
{
    require_tensor(source, "copy", "source");
    require_tensor(destination, "copy", "destination");
    require_same_shape(source, destination, "copy");
    require_same_device(source, destination, "copy");
    require_same_type(source, destination, "copy");

    if (source.is_cuda()) { copy_gpu(source, destination); return; }
    memcpy(destination.get_data(), source.get_data(), source.byte_size());
}

void add(const TensorView& input_1,
         const TensorView& input_2,
         TensorView& output)
{
    require_tensor(input_1, "add", "first input");
    require_tensor(input_2, "add", "second input");
    require_tensor(output, "add", "output");
    require_same_shape(input_1, input_2, "add");
    require_same_shape(input_1, output, "add");
    require_same_device(input_1, input_2, "add");
    require_same_device(input_1, output, "add");
    require_same_type(input_1, input_2, "add");
    require_same_type(input_1, output, "add");
    if (input_1.is_cuda()) { add_gpu(input_1, input_2, output); return; }

    require_cpu_fp32(input_1, "add", "input");
    output.as_vector().noalias() = input_1.as_vector() + input_2.as_vector();
}

static void multiply_cpu(const TensorView& input_a, bool transpose_a,
                  const TensorView& input_b, bool transpose_b,
                  TensorView& output,
                  float alpha, float beta)
{
    const Shape& shape = input_a.get_shape();
    const size_t rank = shape.get_rank();

    const Index batch_count = matrix_count(input_a);

    const bool parallel = output.size() >= 65536;

    #pragma omp parallel for schedule(static) if(parallel)
    for (Index batch_index = 0; batch_index < batch_count; ++batch_index)
    {
        const MatrixMap matrix_a = input_a.as_matrix(batch_index);
        const MatrixMap matrix_b = input_b.as_matrix(batch_index);
        MatrixMap matrix_output = output.as_matrix(batch_index);

        auto gemm_like = [&](auto A, auto B)
        {
            if (beta == 0.0f)
                matrix_output.noalias() = alpha * (A * B);
            else
                matrix_output.noalias() = alpha * (A * B) + beta * matrix_output;
        };

        if (!transpose_a && !transpose_b)       gemm_like(matrix_a,             matrix_b);
        else if (transpose_a && !transpose_b)   gemm_like(matrix_a.transpose(), matrix_b);
        else if (!transpose_a && transpose_b)   gemm_like(matrix_a,             matrix_b.transpose());
        else                                    gemm_like(matrix_a.transpose(), matrix_b.transpose());
    }
}

void multiply(const TensorView& input_a, Transpose transpose_a,
              const TensorView& input_b, Transpose transpose_b,
              TensorView& output,
              float alpha, float beta)
{
    require_tensor(input_a, "multiply", "first input");
    require_tensor(input_b, "multiply", "second input");
    require_tensor(output, "multiply", "output");
    require_same_device(input_a, input_b, "multiply");
    require_same_device(input_a, output, "multiply");

    const Shape& shape_a = input_a.get_shape();
    const Shape& shape_b = input_b.get_shape();
    const Shape& output_shape = output.get_shape();

    throw_if(shape_a.get_rank() < 2 || shape_b.get_rank() < 2 || output_shape.get_rank() < 2,
             "multiply: all tensors must have rank two or greater.");
    require_fp32_or_bf16(input_a, "multiply", "first input");
    require_fp32_or_bf16(input_b, "multiply", "second input");
    require_fp32_or_bf16(output, "multiply", "output");
    if (!input_a.is_cuda())
    {
        require_cpu_fp32(input_a, "multiply", "first input");
        require_same_type(input_a, input_b, "multiply");
        require_same_type(input_a, output, "multiply");
    }

    const size_t rank_a = shape_a.get_rank();
    const size_t rank_b = shape_b.get_rank();
    const Index rows_a = shape_a[rank_a - 2];
    const Index cols_a = shape_a[rank_a - 1];
    const Index rows_b = shape_b[rank_b - 2];
    const Index cols_b = shape_b[rank_b - 1];
    const Index rows_output = output_shape[output_shape.get_rank() - 2];
    const Index cols_output = output_shape.back();
    throw_if(rows_a <= 0 || cols_a <= 0 || rows_b <= 0 || cols_b <= 0
             || rows_output <= 0 || cols_output <= 0,
             "multiply: matrix dimensions must be positive.");

    const bool transpose_first = transpose_a == Transpose::Yes;
    const bool transpose_second = transpose_b == Transpose::Yes;
    const Index inner_a = transpose_first ? rows_a : cols_a;
    const Index inner_b = transpose_second ? cols_b : rows_b;
    const Index result_rows = transpose_first ? cols_a : rows_a;
    const Index result_columns = transpose_second ? rows_b : cols_b;
    throw_if(inner_a != inner_b, "multiply: inner matrix dimensions do not match.");

    const bool flattened_cuda_rhs = input_a.is_cuda() && rank_a > 2 && rank_b == 2;
    if (flattened_cuda_rhs)
    {
        throw_if(transpose_first, "multiply: a flattened CUDA left operand cannot be transposed.");
        const Index flat_rows = input_a.size() / cols_a;
        throw_if(output_shape.back() != result_columns
                 || output.size() != flat_rows * result_columns,
                 "multiply: output shape does not match the flattened matrix product.");
    }
    else
    {
        throw_if(rank_a != rank_b || rank_a != output_shape.get_rank(),
                 "multiply: batched operands and output must have matching ranks.");
        for (size_t i = 0; i + 2 < rank_a; ++i)
            throw_if(shape_a[i] != shape_b[i] || shape_a[i] != output_shape[i],
                     "multiply: batch dimensions do not match.");
        throw_if(matrix_count(input_a) != matrix_count(input_b)
                 || matrix_count(input_a) != matrix_count(output),
                 "multiply: matrix batch counts do not match.");
        throw_if(output_shape[output_shape.get_rank() - 2] != result_rows
                 || output_shape.back() != result_columns,
                 "multiply: output matrix dimensions do not match the product.");
    }

    if (input_a.is_cuda()) { multiply_gpu(input_a, transpose_first, input_b, transpose_second, output, alpha, beta); return; }
    multiply_cpu(input_a, transpose_first, input_b, transpose_second, output, alpha, beta);
}

static void softmax_cpu(TensorView& output)
{
    MatrixMap output_matrix = output.as_flat_matrix();
    const Index rows = output_matrix.rows();

    const bool parallel = output_matrix.size() >= 65536;

    #pragma omp parallel for schedule(static) if(parallel)
    for (Index i = 0; i < rows; ++i)
    {
        const float max_val = output_matrix.row(i).maxCoeff();
        output_matrix.row(i).array() = (output_matrix.row(i).array() - max_val).exp();
        output_matrix.row(i) /= output_matrix.row(i).sum();
    }
}

void softmax(TensorView& output)
{
    if (output.empty()) return;

    require_tensor(output, "softmax", "output");
    throw_if(output.get_shape().back() <= 0, "softmax: the channel dimension must be positive.");
    require_fp32_or_bf16(output, "softmax", "output");
    if (output.is_cuda()) { softmax_gpu(output); return; }

    require_cpu_fp32(output, "softmax", "output");
    softmax_cpu(output);
}

static void softmax_backward_cpu(const TensorView& outputs, TensorView& delta, float alpha)
{
    const MatrixMap output_matrix = outputs.as_flat_matrix();
    MatrixMap delta_matrix = delta.as_flat_matrix();
    const Index rows = delta_matrix.rows();

    const bool parallel = delta_matrix.size() >= 65536;

    #pragma omp parallel for schedule(static) if(parallel)
    for (Index i = 0; i < rows; ++i)
    {
        const float dot = output_matrix.row(i).dot(delta_matrix.row(i));
        delta_matrix.row(i).array() =
            alpha * output_matrix.row(i).array() * (delta_matrix.row(i).array() - dot);
    }
}

void softmax_backward(const TensorView& outputs, TensorView& delta, float alpha)
{
    if (delta.empty()) return;

    require_tensor(outputs, "softmax_backward", "outputs");
    require_tensor(delta, "softmax_backward", "delta");
    require_same_device(outputs, delta, "softmax_backward");
    throw_if(outputs.size() != delta.size()
             || outputs.get_shape().back() != delta.get_shape().back(),
             "softmax_backward: outputs and delta must have the same shape.");
    require_fp32_or_bf16(delta, "softmax_backward", "delta");
    if (delta.is_cuda()) { softmax_backward_gpu(outputs, delta, alpha); return; }

    require_cpu_fp32(outputs, "softmax_backward", "outputs");
    require_cpu_fp32(delta, "softmax_backward", "delta");
    softmax_backward_cpu(outputs, delta, alpha);
}

template <typename Apply>
static void for_each_activation_chunk(Index size, Apply apply)
{
    constexpr Index chunk_size = 16384;
    constexpr Index parallel_threshold = 65536;

    if (size < parallel_threshold)
    {
        apply(Index(0), size);
        return;
    }

    const Index chunks = (size + chunk_size - 1) / chunk_size;

    #pragma omp parallel for schedule(static)
    for (Index chunk = 0; chunk < chunks; ++chunk)
    {
        const Index begin = chunk * chunk_size;
        apply(begin, min(chunk_size, size - begin));
    }
}

static void activation_forward_cpu(TensorView& output, ActivationFunction function)
{
    if (try_activation_forward(output, function)) return;

    using enum ActivationFunction;

    if (function == Identity || function == Softmax) return;

    VectorMap flat = output.as_vector();
    float* const data = flat.data();

    for_each_activation_chunk(flat.size(), [data, function](Index begin, Index count)
    {
        auto a = Map<ArrayXf>(data + begin, count);

        switch (function)
        {
        case Identity:
        case Softmax:
            return;
        case Sigmoid:
            a = (1.0f + (-a).exp()).inverse();
            return;
        case Tanh:
            a = a.tanh();
            return;
        case ReLU:
            a = a.cwiseMax(0.0f);
            return;
        case LeakyReLU:
            a = (a >= 0.0f).select(a, a * LEAKY_RELU_SLOPE);
            return;
        case GELU:
            a = a.unaryExpr([](float x) { return gelu_value(x); });
            return;
        case GELUTanh:
            a = 0.5f * a * (1.0f + (SQRT_2_OVER_PI * (a + GELU_TANH_CUBIC * a * a * a)).tanh());
            return;
        case SiLU:
            a = a / (1.0f + (-a).exp());
            return;
        }
    });
}

static void activation_backward_cpu(const TensorView& outputs, TensorView& delta, ActivationFunction function)
{
    using enum ActivationFunction;

    if (function == Identity || function == Softmax) return;

    auto outputs_flat = outputs.as_vector();
    VectorMap delta_flat = delta.as_vector();

    const float* const outputs_data = outputs_flat.data();
    float* const delta_data = delta_flat.data();

    for_each_activation_chunk(delta_flat.size(), [outputs_data, delta_data, function](Index begin, Index count)
    {
        const auto y = Map<const ArrayXf>(outputs_data + begin, count);
        auto d = Map<ArrayXf>(delta_data + begin, count);

        switch (function)
        {
        case Identity:
        case Softmax:
            return;
        case Sigmoid:
            d *= y * (1.0f - y);
            return;
        case Tanh:
            d *= (1.0f - y.square());
            return;
        case ReLU:
            d = (y > 0.0f).select(d, 0.0f);
            return;
        case LeakyReLU:
            d = (y >= 0.0f).select(d, d * LEAKY_RELU_SLOPE);
            return;
        case GELU:
            d *= y.unaryExpr([](float x) { return gelu_derivative(x); });
            return;
        case GELUTanh:
            d *= y.unaryExpr([](float x) { return gelu_tanh_derivative(x); });
            return;
        case SiLU:
            d *= y.unaryExpr([](float x) { return silu_derivative(x); });
            return;
        }
    });
}

void activation_forward(TensorView& output, ActivationFunction function)
{
    if (function == ActivationFunction::Identity || output.empty()) return;
    if (function == ActivationFunction::Softmax) { softmax(output); return; }

    require_tensor(output, "activation_forward", "output");
    require_fp32_or_bf16(output, "activation_forward", "output");
    if (output.is_cuda()) { activation_forward_gpu(output, function); return; }

    require_cpu_fp32(output, "activation_forward", "output");
    activation_forward_cpu(output, function);
}

void activation_backward(const TensorView& outputs, TensorView& delta, ActivationFunction function)
{
    if (is_one_of(function, ActivationFunction::Identity, ActivationFunction::Softmax)
        || outputs.empty()) return;

    require_tensor(outputs, "activation_backward", "outputs");
    require_tensor(delta, "activation_backward", "delta");
    require_same_shape(outputs, delta, "activation_backward");
    require_same_device(outputs, delta, "activation_backward");
    require_same_type(outputs, delta, "activation_backward");
    require_fp32_or_bf16(outputs, "activation_backward", "outputs");
    if (outputs.is_cuda()) { activation_backward_gpu(outputs, delta, function); return; }

    require_cpu_fp32(outputs, "activation_backward", "outputs");
    activation_backward_cpu(outputs, delta, function);
}

static void linear_forward_cpu(const TensorView& input, const TensorView& weights, const TensorView& bias,
                        TensorView& output, cublasLtEpilogue_t epilogue)
{
    PROFILE_SCOPE_HOST("cpu:linear_fwd");

    const bool fuse_relu = epilogue == CUBLASLT_EPILOGUE_RELU_BIAS
                        || epilogue == CUBLASLT_EPILOGUE_RELU;

    if (try_linear_forward(input, weights, bias, output, fuse_relu)) return;

    auto output_matrix = output.as_flat_matrix();

    {
        PROFILE_SCOPE_HOST("cpu:eigen_gemm");
        output_matrix.noalias() = input.as_flat_matrix() * weights.as_matrix();
    }

    const bool has_bias = !bias.empty();

    if (!has_bias && !fuse_relu) return;

    PROFILE_SCOPE_HOST("cpu:eigen_epilogue");

    if (!has_bias)
    {
        output.as_vector().array() = output.as_vector().array().cwiseMax(0.0f);
        return;
    }

    const Index rows = Index(output_matrix.rows());
    const Index columns = Index(output_matrix.cols());

    if (fuse_relu) add_bias_span<true>(output.as<float>(), bias.as<float>(), 0, rows, columns);
    else           add_bias_span<false>(output.as<float>(), bias.as<float>(), 0, rows, columns);
}

static void linear_backward_cpu(const TensorView& output_delta, const TensorView& input, const TensorView& weights,
                         const TensorView& weight_gradient, const TensorView& bias_gradient,
                         TensorView& input_delta, bool accumulate, const TensorView* addend)
{
    {
    PROFILE_SCOPE_HOST("cpu:bwd_bias");
    if (!bias_gradient.empty())
    {
        const auto delta_matrix = output_delta.as_flat_matrix();
        const Index rows = Index(delta_matrix.rows());
        const Index columns = Index(delta_matrix.cols());
        const Index block_rows = gemm_block_rows(rows, get_device().numThreads());
        const Index blocks = (rows + block_rows - 1) / block_rows;

        if (blocks < 2 || omp_in_parallel() || !output_delta.is_fp32() || !bias_gradient.is_fp32())
        {
            bias_gradient.as_vector().noalias() = delta_matrix.colwise().sum();
        }
        else
        {
            const float* const values = output_delta.as<float>();
            vector<float> partials(size_t(blocks) * size_t(columns));

            #pragma omp parallel for schedule(static)
            for (Index block = 0; block < blocks; block++)
            {
                const Index first = block * block_rows;

                column_sums(values, partials.data() + size_t(block) * size_t(columns),
                            rows, columns, first, min<Index>(rows, first + block_rows));
            }

            float* const sums = bias_gradient.as<float>();

            for (Index j = 0; j < columns; j++) sums[j] = partials[size_t(j)];

            for (Index block = 1; block < blocks; block++)
            {
                const float* const partial = partials.data() + size_t(block) * size_t(columns);

                for (Index j = 0; j < columns; j++) sums[j] += partial[j];
            }
        }
    }
    }

    PROFILE_SCOPE_HOST("cpu:bwd_gemms");

    if (try_linear_backward(output_delta, input, weights, weight_gradient,
                            input_delta, accumulate, addend))
        return;

    weight_gradient.as_matrix().noalias() = input.as_flat_matrix().transpose() * output_delta.as_flat_matrix();

    if (!input_delta.get_data() || input_delta.empty()) return;

    auto input_delta_mat = input_delta.as_flat_matrix();
    const auto product   = output_delta.as_flat_matrix() * weights.as_matrix().transpose();

    if (accumulate)   input_delta_mat.noalias() += product;
    else if (addend)  input_delta_mat.noalias()  = product + addend->as_flat_matrix();
    else              input_delta_mat.noalias()  = product;
}

void linear_forward(const TensorView& input, const TensorView& weights, const TensorView& bias,
                    TensorView& output, cublasLtEpilogue_t epilogue, TensorView* pre_activation,
                    const TensorView& weight_scale, ActivationFunction fused_activation)
{
    constexpr string_view operation = "linear_forward";
    validate_linear_io(input, weights, output, false, operation);
    validate_linear_types(input, weights, output, operation);

    require_optional_tensor(input, bias, operation, "bias");
    if (!bias.empty())
    {
        throw_if(bias.get_shape().get_rank() != 1 || bias.size() != output.get_shape().back(),
                 "linear_forward: bias shape does not match the output features.");
        throw_if(bias.get_type() != output.get_type() && !(output.is_bf16() && bias.is_fp32()),
                 "linear_forward: bias dtype is incompatible with the output.");
    }

    if (pre_activation)
    {
        require_tensor(*pre_activation, operation, "pre-activation output");
        require_same_device(input, *pre_activation, operation);
        if (epilogue == CUBLASLT_EPILOGUE_GELU_AUX_BIAS)
        {
            require_same_shape(output, *pre_activation, operation);
            require_same_type(output, *pre_activation, operation);
        }
        else if (epilogue == CUBLASLT_EPILOGUE_RELU_AUX_BIAS)
        {
            const Index rows = output.size() / output.get_shape().back();
            throw_if(!pre_activation->is_int8() || pre_activation->get_shape().get_rank() != 2
                     || pre_activation->get_shape()[0] != rows
                     || pre_activation->get_shape()[1] * 8 != output.get_shape().back(),
                     "linear_forward: ReLU mask shape or dtype is incompatible with the output.");
        }
        else
        {
            throw runtime_error("linear_forward: auxiliary output requires an auxiliary epilogue.");
        }
    }

    require_optional_tensor(input, weight_scale, operation, "weight scale");
    if (weights.is_int8())
        require_int8_linear(input, output, weight_scale, operation);

    if (input.is_cuda())
    {
        linear_forward_gpu(input, weights, bias, output, epilogue, pre_activation, weight_scale,
                           fused_activation);
        return;
    }

    throw_if(weights.is_int8(), "linear_forward: INT8 weights are CUDA-only.");

    throw_if(epilogue == CUBLASLT_EPILOGUE_GELU_AUX_BIAS,
             "linear_forward: the GELU_AUX_BIAS epilogue is CUDA-only.");

    linear_forward_cpu(input, weights, bias, output, epilogue);

    if (fused_activation != ActivationFunction::Identity)
        activation_forward(output, fused_activation);
}

void linear_backward(const TensorView& output_delta, const TensorView& input, const TensorView& weights,
                     const TensorView& weight_gradient, const TensorView& bias_gradient,
                     TensorView& input_delta, bool accumulate_input_delta,
                     const LinearBackwardOptions& options)
{
    constexpr string_view operation = "linear_backward";
    const TensorView* drelu_mask = options.drelu_mask;
    const TensorView* addend = options.addend && !options.addend->empty() ? options.addend : nullptr;
    bool* const fused_input_relu = options.fused_input_relu;
    validate_linear_io(input, weights, output_delta, false, operation);
    require_fp32_or_bf16(output_delta, operation, "output delta");
    require_fp32_or_bf16(input, operation, "input");
    throw_if(weights.get_type() != output_delta.get_type(),
             "linear_backward: weights and output delta must use the same dtype.");
    throw_if(weights.is_int8(), "linear_backward: INT8 weights are inference-only.");

    require_tensor(weight_gradient, operation, "weight gradient");
    require_same_device(input, weight_gradient, operation);
    require_same_shape(weights, weight_gradient, operation);
    throw_if(!weight_gradient.is_fp32(), "linear_backward: weight gradient must use FP32 storage.");

    require_optional_tensor(input, bias_gradient, operation, "bias gradient");
    if (!bias_gradient.empty())
        throw_if(!bias_gradient.is_fp32() || bias_gradient.get_shape().get_rank() != 1
                 || bias_gradient.size() != output_delta.get_shape().back(),
                 "linear_backward: bias gradient must be an FP32 output-feature vector.");

    require_optional_tensor(input, input_delta, operation, "input delta");
    if (!input_delta.empty())
    {
        require_same_shape(input, input_delta, operation);
        require_same_type(input, input_delta, operation);
    }

    if (!output_delta.is_cuda())
    {
        require_cpu_fp32(output_delta, operation, "output delta");
        require_cpu_fp32(input, operation, "input");
        require_cpu_fp32(weights, operation, "weights");
    }

    if (drelu_mask)
    {
        require_tensor(*drelu_mask, operation, "DReLU mask");
        require_same_device(output_delta, *drelu_mask, operation);
        const Index rows = input.size() / input.get_shape().back();
        throw_if(!drelu_mask->is_int8() || drelu_mask->get_shape().get_rank() != 2
                 || drelu_mask->get_shape()[0] != rows
                 || drelu_mask->get_shape()[1] * 8 != input.get_shape().back(),
                 "linear_backward: DReLU mask shape or dtype is incompatible with the input.");
    }

    throw_if(drelu_mask && (!output_delta.is_cuda() || accumulate_input_delta),
             "linear_backward: the DRELU fused input-delta path is CUDA, non-accumulating only.");

    if (addend)
    {
        require_tensor(*addend, operation, "input delta addend");
        require_same_shape(input_delta, *addend, operation);
        require_same_type(input_delta, *addend, operation);
        require_same_device(input_delta, *addend, operation);
        throw_if(accumulate_input_delta || input_delta.empty(),
                 "linear_backward: the input delta addend needs a non-accumulating input delta.");
    }

    if (output_delta.is_cuda())
        return linear_backward_gpu(output_delta, input, weights, weight_gradient, bias_gradient,
                                   input_delta, accumulate_input_delta,
                                   {drelu_mask, addend, fused_input_relu});
    if (fused_input_relu) *fused_input_relu = false;
    linear_backward_cpu(output_delta, input, weights, weight_gradient, bias_gradient,
                        input_delta, accumulate_input_delta, addend);
}

#ifdef OPENNN_HAS_CUDA

constexpr Index int8_dequant_budget_bytes = Index(32) * 1024 * 1024;

static void w8a16_linear_rows(Index rows, Index in_features, Index out_features,
                              bool weights_out_major,
                              const bfloat16* x, const int8_t* weights, const float* scales,
                              const bfloat16* bias, bfloat16* y)
{
    for (Index row = 0; row < rows; row += W8A16_MAX_M)
        w8a16_linear_cuda<bfloat16>(to_int(min(Index(W8A16_MAX_M), rows - row)),
                                    to_int(in_features), to_int(out_features), weights_out_major,
                                    x + row * in_features, weights, scales, bias,
                                    y + row * out_features);
}

#endif

void linear_forward_transposed(const TensorView& input, const TensorView& embed_weight, TensorView& output,
                          const TensorView& weight_scale)
{
    constexpr string_view operation = "linear_forward_transposed";
    validate_linear_io(input, embed_weight, output, true, operation);
    validate_linear_types(input, embed_weight, output, operation);
    require_optional_tensor(input, weight_scale, operation, "weight scale");

    if (embed_weight.is_int8())
        require_int8_linear(input, output, weight_scale, operation);

#ifdef OPENNN_HAS_CUDA
    if (input.is_cuda() && embed_weight.is_int8())
    {
        throw_if(weight_scale.empty() || !input.is_bf16() || !output.is_bf16(),
                 "linear_forward_transposed: INT8 weights require BF16 activations and a per-channel scale vector.");

        const Index in_features  = embed_weight.get_shape().back();
        const Index out_features = embed_weight.size() / in_features;
        const Index rows = input.size() / in_features;

        if (rows <= W8A16_MAX_M)
            return w8a16_linear_rows(rows, in_features, out_features, true,
                                     input.as<bfloat16>(), embed_weight.as<int8_t>(),
                                     weight_scale.as<float>(), nullptr, output.as<bfloat16>());

        const Index tile_rows = min(out_features,
            max(Index(1), int8_dequant_budget_bytes / (in_features * Index(sizeof(bfloat16)))));
        bfloat16* dequantized = ensure_int8_dequant_workspace(tile_rows * in_features);

        for (Index j0 = 0; j0 < out_features; j0 += tile_rows)
        {
            const Index tile = min(tile_rows, out_features - j0);
            w8_dequant_cuda<bfloat16>(tile, in_features, true,
                                      embed_weight.as<int8_t>() + j0 * in_features,
                                      weight_scale.as<float>() + j0, dequantized);
            gemm_strided_batched_cuda(CUBLAS_OP_T, CUBLAS_OP_N,
                                      to_int(tile), to_int(rows), to_int(in_features),
                                      dequantized, CUDA_R_16BF, to_int(in_features), 0,
                                      input.get_data(), CUDA_R_16BF, to_int(in_features), 0,
                                      output.as<bfloat16>() + j0, CUDA_R_16BF, to_int(out_features), 0,
                                      1);
        }
        return;
    }
#endif

    if (input.is_cuda()) { multiply(input, Transpose::No, embed_weight, Transpose::Yes, output, 1.0f, 0.0f); return; }
    output.as_flat_matrix().noalias() =
        input.as_flat_matrix() * embed_weight.as_matrix().transpose();
}

#ifdef OPENNN_HAS_CUDA

static void copy_gpu(const TensorView& source, TensorView& destination)
{
    device::copy_async(destination.get_data(), source.get_data(), source.byte_size(),
                       device::CopyKind::DeviceToDevice,
                       device::get_compute_stream());
}

static void add_gpu(const TensorView& input_1,
             const TensorView& input_2,
             TensorView& output)
{

    if (input_1.is_fp32() && input_2.is_fp32() && output.is_fp32())
        return add_relu_cuda(output.size(), input_1.as<float>(), input_2.as<float>(),
                              false, output.as<float>());

    CHECK_CUDNN(cudnnOpTensor(device::get_cudnn_handle(),
                              device::get_op_tensor_add_descriptor(),
                              &one, input_1.get_descriptor(), input_1.get_data(),
                              &one, input_2.get_descriptor(), input_2.get_data(),
                              &zero, output.get_descriptor(), output.get_data()));
}

static void multiply_gpu(const TensorView& input_a, bool transpose_a,
                  const TensorView& input_b, bool transpose_b,
                  TensorView& output,
                  float alpha, float beta)
{
    const size_t rank_a = input_a.get_rank();
    const size_t rank_b = input_b.get_rank();

    int rows_a = to_int(input_a.get_shape()[rank_a - 2]);
    const int cols_a = to_int(input_a.get_shape()[rank_a - 1]);
    const int rows_b = to_int(input_b.get_shape()[rank_b - 2]);
    const int cols_b = to_int(input_b.get_shape()[rank_b - 1]);

    if (rank_b == 2 && rank_a > 2)
        rows_a = to_int(input_a.size() / cols_a);

    const int cols_out = transpose_b ? rows_b : cols_b;
    const int rows_out = transpose_a ? cols_a : rows_a;
    const int inner_dim = transpose_a ? rows_a : cols_a;

    const cublasOperation_t operation_b = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t operation_a = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    const long long stride_a = 1LL * rows_a * cols_a;
    const long long stride_b = 1LL * rows_b * cols_b;
    const int batch_count = to_int(input_a.size() / stride_a);
    const long long stride_output = output.get_shape()[output.get_rank() - 2]
                                  * output.get_shape()[output.get_rank() - 1];

    gemm_strided_batched_cuda(operation_b, operation_a,
                              cols_out, rows_out, inner_dim,
                              input_b.get_data(), input_b.cuda_dtype(), cols_b, stride_b,
                              input_a.get_data(), input_a.cuda_dtype(), cols_a, stride_a,
                              output.get_data(), output.cuda_dtype(), cols_out, stride_output,
                              batch_count,
                              alpha, beta);
}

template<typename Apply>
static void for_each_softmax_chunk(const TensorView& reference, Apply apply)
{
    const Index channels = reference.get_shape().back();
    const Index rows = reference.size() / channels;
    const Index max_rows = Index(numeric_limits<int>::max()) / channels;

    for (Index row = 0; row < rows; row += max_rows)
        apply(row, min(max_rows, rows - row));
}

static TensorView softmax_chunk(const TensorView& view, Index first, Index count)
{
    const Index channels = view.get_shape().back();

    if (first == 0 && count * channels == view.size()) return view;

    return TensorView(static_cast<char*>(view.get_data()) + first * channels * type_bytes(view.get_type()),
                      Shape{count, channels},
                      view.get_type(), view.get_device());
}

static void softmax_gpu(TensorView& output)
{
    for_each_softmax_chunk(output, [&](Index first, Index count)
    {
        const TensorView chunk = softmax_chunk(output, first, count);

        CHECK_CUDNN(cudnnSoftmaxForward(device::get_cudnn_handle(),
                                        CUDNN_SOFTMAX_ACCURATE,
                                        CUDNN_SOFTMAX_MODE_CHANNEL,
                                        &one,
                                        chunk.get_descriptor(), chunk.get_data(),
                                        &zero,
                                        chunk.get_descriptor(), chunk.get_data()));
    });
}

static void softmax_backward_gpu(const TensorView& outputs, TensorView& delta, float alpha)
{
    for_each_softmax_chunk(delta, [&](Index first, Index count)
    {
        const TensorView output_chunk = softmax_chunk(outputs, first, count);
        const TensorView delta_chunk = softmax_chunk(delta, first, count);

        CHECK_CUDNN(cudnnSoftmaxBackward(device::get_cudnn_handle(),
                                         CUDNN_SOFTMAX_ACCURATE,
                                         CUDNN_SOFTMAX_MODE_CHANNEL,
                                         &alpha,
                                         output_chunk.get_descriptor(), output_chunk.get_data(),
                                         delta_chunk.get_descriptor(), delta_chunk.get_data(),
                                         &zero,
                                         delta_chunk.get_descriptor(), delta_chunk.get_data()));
    });
}

static void activation_forward_gpu(TensorView& output, ActivationFunction function)
{
    output.dispatch([&]<typename T>()
    {
        activation_forward_cuda<T>(output.size(), output.as<T>(), static_cast<int>(function));
    });
    device::check_last_error();
}

static void activation_backward_gpu(const TensorView& outputs, TensorView& delta, ActivationFunction function)
{
    delta.dispatch([&]<typename T>()
    {
        activation_backward_cuda<T>(delta.size(), outputs.as<T>(), delta.as<T>(), static_cast<int>(function));
    });

    device::check_last_error();
}

static Index gemm_row_chunk()
{
    static const Index rows = []
    {
        const char* const requested = getenv("OPENNN_GEMM_ROW_CHUNK");
        const long long requested_rows = requested ? atoll(requested) : -1;

        return requested_rows >= 0 ? Index(requested_rows) : Index(16384);
    }();

    return rows;
}

static void linear_forward_lt_gpu(const TensorView& input, const TensorView& weights, const TensorView& bias,
                                  TensorView& output, cublasLtEpilogue_t epilogue,
                                  TensorView* pre_activation)
{
    const int input_columns  = to_int(input.flat_columns());
    const int output_columns = to_int(weights.flat_columns());
    const int total_rows     = to_int(input.flat_rows());

    const void* input_for_gemm = data_for_gemm_dtype(input, weights.get_type());
    const cudaDataType_t io_type = output.cuda_dtype();

    const void* bias_for_gemm = (bias.get_data() && output.is_bf16() && bias.is_fp32())
        ? bias_for_gemm_bf16(bias)
        : bias.get_data();

    const Index chunk = pre_activation ? Index(0) : gemm_row_chunk();
    const bool chunked = chunk > 0 && Index(total_rows) > chunk;

    const bool narrow_k_applies = !pre_activation
        && (epilogue == CUBLASLT_EPILOGUE_RELU_BIAS || epilogue == CUBLASLT_EPILOGUE_BIAS)
        && input.is_bf16() && weights.is_bf16() && output.is_bf16()
        && bias.get_data() && bias.is_bf16();

    if (!chunked && narrow_k_applies
        && narrow_k_linear_forward_cutlass(total_rows, input_columns, output_columns,
                                           input_for_gemm, weights.get_data(), bias_for_gemm,
                                           output.get_data(),
                                           epilogue == CUBLASLT_EPILOGUE_RELU_BIAS,
                                           device::get_compute_stream()))
        return;

    try
    {
        if (chunked)
        {
            const Index input_row_bytes  = Index(input_columns) * type_bytes(weights.get_type());
            const Index output_row_bytes = Index(output_columns) * type_bytes(output.get_type());

            const char* const input_base = static_cast<const char*>(input_for_gemm);
            char* const output_base = static_cast<char*>(output.get_data());

            for (Index start = 0; start < Index(total_rows); start += chunk)
            {
                const int rows = to_int(min(chunk, Index(total_rows) - start));

                if (narrow_k_applies
                    && narrow_k_linear_forward_cutlass(rows, input_columns, output_columns,
                                                       input_base + start * input_row_bytes,
                                                       weights.get_data(), bias_for_gemm,
                                                       output_base + start * output_row_bytes,
                                                       epilogue == CUBLASLT_EPILOGUE_RELU_BIAS,
                                                       device::get_compute_stream()))
                    continue;

                run_lt_matmul_cached(
                    output_columns, rows, input_columns,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    epilogue,
                    weights.get_data(),
                    input_base + start * input_row_bytes,
                    output_base + start * output_row_bytes,
                    bias_for_gemm,
                    io_type, io_type,
                    nullptr);
            }
        }
        else
        run_lt_matmul_cached(
            output_columns, total_rows, input_columns,
            CUBLAS_OP_N, CUBLAS_OP_N,
            epilogue,
            weights.get_data(), input_for_gemm, output.get_data(), bias_for_gemm,
            io_type, io_type,
            pre_activation ? pre_activation->get_data() : nullptr);
    }
    catch (const runtime_error& e)
    {
        if (epilogue == CUBLASLT_EPILOGUE_GELU_AUX_BIAS && pre_activation)
        {
            linear_forward_lt_gpu(input, weights, bias, *pre_activation,
                                  CUBLASLT_EPILOGUE_BIAS, nullptr);
            copy_gpu(*pre_activation, output);
            return activation_forward_gpu(output, ActivationFunction::GELUTanh);
        }

        throw runtime_error(format("cuBLASLt GEMM {}x{}x{} ({}) failed: {}",
                                   output_columns, total_rows, input_columns,
                                   output.is_bf16() ? "bf16" : "fp32", e.what()));
    }
}

static bool aligned_for_vectors(const void* pointer)
{
    return reinterpret_cast<uintptr_t>(pointer) % 16 == 0;
}

static bool single_output_layer_shape(const TensorView& input, const TensorView& weights, Index lanes)
{
    if (weights.get_shape().back() != 1) return false;
    if (input.get_type() != weights.get_type()) return false;
    if (!input.is_bf16() && !input.is_fp32()) return false;

    const Index per_vector = Index(16) / Index(type_bytes(input.get_type()));
    if (input.flat_columns() % (lanes * per_vector) != 0) return false;

    return aligned_for_vectors(input.get_data()) && aligned_for_vectors(weights.get_data());
}

static bool single_output_reduction_applies(const TensorView& input, const TensorView& weights,
                                            const TensorView& bias, const TensorView& output,
                                            cublasLtEpilogue_t epilogue, const TensorView* pre_activation)
{
    static const bool enabled = env_flag_enabled("OPENNN_LINEAR_SINGLE_OUTPUT", true);
    if (!enabled || pre_activation) return false;

    if (epilogue != CUBLASLT_EPILOGUE_DEFAULT && epilogue != CUBLASLT_EPILOGUE_BIAS) return false;
    if (input.get_type() != output.get_type()) return false;
    if (bias.get_data() && bias.get_type() != input.get_type()) return false;

    return single_output_layer_shape(input, weights, 1);
}

static void linear_forward_gpu(const TensorView& input, const TensorView& weights, const TensorView& bias,
                               TensorView& output, cublasLtEpilogue_t epilogue,
                               TensorView* pre_activation, const TensorView& weight_scale,
                               ActivationFunction fused_activation)
{
    PROFILE_SCOPE("op:linear_fwd " + to_string(weights.flat_columns()) + "x"
                  + to_string(input.flat_columns()) + "x" + to_string(input.flat_rows()));
    if (!weights.is_int8()
        && single_output_reduction_applies(input, weights, bias, output, epilogue, pre_activation))
    {
        const Index features = input.get_shape().back();
        const Index rows = input.size() / features;
        const void* bias_data = (epilogue == CUBLASLT_EPILOGUE_BIAS) ? bias.get_data() : nullptr;

        input.dispatch([&]<typename T>() {
            linear_forward_single_output_cuda<T>(rows, features,
                                                 input.as<T>(), weights.as<T>(),
                                                 static_cast<const T*>(bias_data),
                                                 int(fused_activation),
                                                 output.as<T>());
        });
        return;
    }

    const auto run_fused_activation = [&]
    {
        if (fused_activation != ActivationFunction::Identity)
            activation_forward_gpu(output, fused_activation);
    };

    if (!weights.is_int8())
    {
        linear_forward_lt_gpu(input, weights, bias, output, epilogue, pre_activation);
        run_fused_activation();
        return;
    }

    throw_if(weight_scale.empty() || !input.is_bf16() || !output.is_bf16(),
             "linear_forward: INT8 weights require BF16 activations and a per-channel scale vector.");

    const int input_columns  = to_int(input.flat_columns());
    const int output_columns = to_int(weights.flat_columns());
    const int total_rows     = to_int(input.flat_rows());

    const bool gemv_path = (total_rows <= W8A16_MAX_M
                            || weights.byte_size() > int8_dequant_budget_bytes)
        && (epilogue == CUBLASLT_EPILOGUE_DEFAULT || epilogue == CUBLASLT_EPILOGUE_BIAS)
        && (!bias.get_data() || bias.is_bf16());

    if (gemv_path)
    {
        w8a16_linear_rows(total_rows, input_columns, output_columns, false,
                          input.as<bfloat16>(), weights.as<int8_t>(), weight_scale.as<float>(),
                          epilogue == CUBLASLT_EPILOGUE_BIAS && bias.get_data()
                              ? bias.as<bfloat16>() : nullptr,
                          output.as<bfloat16>());
        return run_fused_activation();
    }

    bfloat16* dequantized = ensure_int8_dequant_workspace(weights.size());
    w8_dequant_cuda<bfloat16>(input_columns, output_columns, false, weights.as<int8_t>(),
                              weight_scale.as<float>(), dequantized);
    const TensorView dequantized_weights(dequantized, weights.get_shape(),
                                         Type::BF16, Device::CUDA);
    linear_forward_lt_gpu(input, dequantized_weights, bias, output, epilogue, pre_activation);
    run_fused_activation();
}

static bool single_output_backward_applies(const TensorView& output_delta, const TensorView& input,
                                           const TensorView& weights, const TensorView& weight_gradient,
                                           const TensorView& bias_gradient, const TensorView& input_delta,
                                           bool accumulate_input_delta,
                                           const LinearBackwardOptions& options)
{
    static const bool enabled = env_flag_enabled("OPENNN_SINGLE_OUTPUT_KERNEL", true);

    if (!enabled || options.drelu_mask || options.addend || accumulate_input_delta) return false;
    if (input.get_type() != output_delta.get_type()) return false;
    if (!weight_gradient.is_fp32()) return false;
    if (!bias_gradient.empty() && (!bias_gradient.is_fp32() || bias_gradient.size() != 1)) return false;
    if (!input_delta.empty()
        && (input_delta.get_type() != input.get_type() || !aligned_for_vectors(input_delta.get_data())))
        return false;

    return single_output_layer_shape(input, weights, 32);
}

static void linear_backward_gpu(const TensorView& output_delta, const TensorView& input, const TensorView& weights,
                         const TensorView& weight_gradient, const TensorView& bias_gradient,
                         TensorView& input_delta, bool accumulate_input_delta,
                         const LinearBackwardOptions& options)
{
    const TensorView* const drelu_mask = options.drelu_mask;
    const TensorView* const addend = options.addend;

    if (single_output_backward_applies(output_delta, input, weights, weight_gradient,
                                       bias_gradient, input_delta, accumulate_input_delta,
                                       options))
    {
        PROFILE_SCOPE("op:linear_bwd_single_output");
        const Index features = input.flat_columns();
        const Index rows = input.flat_rows();
        const bool has_input_delta = !input_delta.empty() && input_delta.get_data();
        const bool fuse_relu = options.fused_input_relu && has_input_delta;

        bool done = false;
        input.dispatch([&]<typename T>() {
            done = linear_backward_single_output_cuda<T>(
                rows, features,
                output_delta.as<T>(), input.as<T>(), weights.as<T>(),
                has_input_delta ? input_delta.as<T>() : nullptr,
                fuse_relu,
                weight_gradient.as<float>(),
                bias_gradient.empty() ? nullptr : bias_gradient.as<float>());
        });
        if (done)
        {
            if (options.fused_input_relu) *options.fused_input_relu = fuse_relu;
            return;
        }
    }

    const int input_columns  = to_int(input.flat_columns());
    const int output_columns = to_int(output_delta.flat_columns());
    const int total_rows     = to_int(input.flat_rows());

    const void* input_for_gemm = data_for_gemm_dtype(input, weights.get_type());

    if (options.fused_input_relu) *options.fused_input_relu = false;

    const bool has_bias = bias_gradient.size() > 0;

    static atomic<bool> bf16_fp32_store_supported{true};

    static const bool force_staged = env_flag_enabled("OPENNN_WGRAD_STAGED", false);

    const bool direct_fp32_store = !output_delta.is_bf16()
        || (bf16_fp32_store_supported.load(memory_order_relaxed) && !force_staged);

    bool stored = false;
    {
    PROFILE_SCOPE("op:linear_bwd_wgrad " + to_string(output_columns) + "x" + to_string(input_columns) + "x" + to_string(total_rows));

    const bool skinny_wgrad = Index(output_columns) * Index(input_columns) <= Index(64) * 1024
                           && Index(total_rows) >= 4 * Index(max(output_columns, input_columns));
    if (skinny_wgrad)
    {
        const TensorView input_2d(const_cast<void*>(input_for_gemm), Shape{total_rows, input_columns},
                                  weights.get_type(), Device::CUDA);
        const TensorView output_delta_2d(output_delta.get_data(), Shape{total_rows, output_columns},
                                         output_delta.get_type(), Device::CUDA);
        TensorView weight_gradient_2d(weight_gradient.get_data(), Shape{input_columns, output_columns},
                                      Type::FP32, Device::CUDA);
        multiply(input_2d, Transpose::Yes, output_delta_2d, Transpose::No, weight_gradient_2d, 1.0f, 0.0f);

        if (has_bias)
        {
            device::set_zero_async(bias_gradient.get_data(),
                                   bias_gradient.size() * Index(sizeof(float)),
                                   device::get_compute_stream());
            output_delta.dispatch([&]<typename T>() {
                bias_grad_sum_cuda<T>(total_rows, output_columns,
                                      output_delta.as<T>(), bias_gradient.as<float>());
            });
        }
        stored = true;
    }
    else if (direct_fp32_store)
    {
        try
        {
            run_lt_matmul_cached(
                output_columns, input_columns, total_rows,
                CUBLAS_OP_N, CUBLAS_OP_T,
                has_bias ? CUBLASLT_EPILOGUE_BGRADA : CUBLASLT_EPILOGUE_DEFAULT,
                output_delta.get_data(), input_for_gemm, weight_gradient.get_data(),
                has_bias ? bias_gradient.as<float>() : nullptr,
                output_delta.cuda_dtype(),
                CUDA_R_32F);
            stored = true;
        }
        catch (const exception&)
        {
            if (!output_delta.is_bf16()) throw;
            bf16_fp32_store_supported.store(false, memory_order_relaxed);
            cerr << "linear_backward: cuBLASLt has no BF16-in/FP32-out weight-gradient "
                    "epilogue here; using BF16 store + cast for the rest of the process.\n";
            device::reset_last_error();
        }
    }

    if (!stored)
    {
        bfloat16* dw_bf16 = ensure_bf16_gradient_workspace(weight_gradient.size());
        run_lt_matmul_cached(
            output_columns, input_columns, total_rows,
            CUBLAS_OP_N, CUBLAS_OP_T,
            CUBLASLT_EPILOGUE_DEFAULT,
            output_delta.get_data(), input_for_gemm, dw_bf16, nullptr,
            output_delta.cuda_dtype(),
            CUDA_R_16BF);
        cast_bf16_to_fp32(weight_gradient.size(), dw_bf16, weight_gradient.as<float>());

        if (has_bias)
        {
            device::set_zero_async(bias_gradient.get_data(),
                                   bias_gradient.size() * Index(sizeof(float)),
                                   device::get_compute_stream());
            bias_grad_sum_cuda<bfloat16>(total_rows, output_columns,
                                         output_delta.as<bfloat16>(), bias_gradient.as<float>());
        }
    }
    }

    if (!input_delta.get_data() || input_delta.empty()) return;

    PROFILE_SCOPE("op:linear_bwd_dx " + to_string(output_columns) + "x" + to_string(input_columns) + "x" + to_string(total_rows));
    if (drelu_mask || addend)
        return run_lt_matmul_cached(
                   input_columns, total_rows, output_columns,
                   CUBLAS_OP_T, CUBLAS_OP_N,
                   drelu_mask ? CUBLASLT_EPILOGUE_DRELU : CUBLASLT_EPILOGUE_DEFAULT,
                   weights.get_data(), output_delta.get_data(), input_delta.get_data(), nullptr,
                   output_delta.cuda_dtype(), input_delta.cuda_dtype(),
                   drelu_mask ? drelu_mask->get_data() : nullptr,
                   addend ? addend->get_data() : nullptr);

    multiply(output_delta, Transpose::No, weights, Transpose::Yes, input_delta, 1.0f,
             accumulate_input_delta ? 1.0f : 0.0f);
}

#else

#define OPENNN_STUB_GPU_OP(name, sig) OPENNN_CUDA_STUB(void, name, sig)
OPENNN_GPU_OPS(OPENNN_STUB_GPU_OP)
#undef OPENNN_STUB_GPU_OP

#endif


MatrixR append_rows(const MatrixR& starting_matrix, const MatrixR& block)
{
    if (starting_matrix.size() == 0)
        return block;
    if (block.size() == 0)
        return starting_matrix;

    throw_if(starting_matrix.cols() != block.cols(),
             "append_rows: Column mismatch ({} vs {})",
             starting_matrix.cols(), block.cols());

    MatrixR final_matrix(starting_matrix.rows() + block.rows(), starting_matrix.cols());

    final_matrix.topRows(starting_matrix.rows()) = starting_matrix;
    final_matrix.bottomRows(block.rows()) = block;

    return final_matrix;
}


MatrixR append_columns(const MatrixR& first_matrix, const MatrixR& second_matrix)
{
    MatrixR result(first_matrix.rows(), first_matrix.cols() + second_matrix.cols());
    result.leftCols(first_matrix.cols()) = first_matrix;
    result.rightCols(second_matrix.cols()) = second_matrix;
    return result;
}


VectorR slice_rows(const VectorR& values, const vector<Index>& indices)
{
    VectorR result(ssize(indices));

    for (Index i = 0; i < ssize(indices); ++i)
        result(i) = values(indices[i]);

    return result;
}


MatrixR slice_rows(const MatrixR& matrix, const vector<Index>& indices)
{
    MatrixR result(ssize(indices), matrix.cols());

    for (Index i = 0; i < ssize(indices); ++i)
        result.row(i) = matrix.row(indices[i]);

    return result;
}


pair<MatrixR, MatrixR> slice_rows(const pair<MatrixR, MatrixR>& matrices, const vector<Index>& indices)
{
    return {slice_rows(matrices.first, indices), slice_rows(matrices.second, indices)};
}


pair<MatrixR, MatrixR> append_rows(const pair<MatrixR, MatrixR>& matrices, const pair<MatrixR, MatrixR>& blocks)
{
    return {append_rows(matrices.first, blocks.first), append_rows(matrices.second, blocks.second)};
}


MatrixR append_columns(const pair<MatrixR, MatrixR>& matrices)
{
    return append_columns(matrices.first, matrices.second);
}


MatrixR calculate_distances(const MatrixR& points)
{
    const VectorR squared_norms = points.rowwise().squaredNorm();

    MatrixR squared_distances = -2.0f * points * points.transpose();
    squared_distances.colwise() += squared_norms;
    squared_distances.rowwise() += squared_norms.transpose();

    return squared_distances.cwiseMax(0.0f).cwiseSqrt();
}


VectorI get_nearest_points(const MatrixR& matrix, const VectorR& point, Index neighbors_number)
{
    const Index rows = matrix.rows();

    const VectorR distances = (matrix.rowwise() - point.transpose()).rowwise().norm();

    vector<pair<float, Index>> pairs(rows);

    for (Index i = 0; i < rows; ++i)
        pairs[i] = {distances(i), i};

    if (neighbors_number > rows)
        neighbors_number = rows;

    partial_sort(pairs.begin(), pairs.begin() + neighbors_number, pairs.end());

    VectorI result(neighbors_number);
    transform(pairs.begin(), pairs.begin() + neighbors_number, result.data(),
              [](const auto& p) { return p.second; });
    return result;
}


bool row_dominates(const MatrixR& values, const Index a, const Index b)
{
    bool strictly_better = false;

    for (Index j = 0; j < values.cols(); ++j)
    {
        const float difference = values(a, j) - values(b, j);

        if (difference < 0.0f) return false;
        if (difference > 0.0f) strictly_better = true;
    }

    return strictly_better;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
