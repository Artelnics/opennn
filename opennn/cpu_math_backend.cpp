//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C P U   M A T H   B A C K E N D
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "cpu_math_backend.h"

#ifdef EIGEN_USE_MKL_ALL
#include <mkl_cblas.h>
#include <mkl_vml.h>
#endif

namespace opennn::cpu_math
{

#ifdef EIGEN_USE_MKL_ALL

static bool env_enabled(const char* name)
{
    const char* value = getenv(name);
    return value && (strcmp(value, "1") == 0
                  || strcmp(value, "true") == 0
                  || strcmp(value, "TRUE") == 0);
}

static bool fast_vml_enabled()    { static const bool enabled = env_enabled("OPENNN_MKL_FAST_VML");    return enabled; }
static bool packed_gemm_enabled() { static const bool enabled = env_enabled("OPENNN_MKL_PACKED_GEMM"); return enabled; }

struct PackedGemmCache
{
    const void* source = nullptr;
    int m = 0;
    int n = 0;
    int k = 0;
    vector<float> data;
};

static const float* get_packed_b(const TensorView& weights, int m, int n, int k)
{
    static thread_local vector<PackedGemmCache> caches;

    for (PackedGemmCache& cache : caches)
        if (cache.source == weights.data
            && cache.m == m
            && cache.n == n
            && cache.k == k)
            return cache.data.data();

    PackedGemmCache cache;
    cache.source = weights.data;
    cache.m = m;
    cache.n = n;
    cache.k = k;

    cache.data.resize(cblas_sgemm_pack_get_size(CblasBMatrix, m, n, k));
    cblas_sgemm_pack(CblasRowMajor,
                     CblasBMatrix,
                     CblasNoTrans,
                     m,
                     n,
                     k,
                     1.0f,
                     weights.as<float>(),
                     n,
                     cache.data.data());

    caches.push_back(move(cache));
    return caches.back().data.data();
}

static void tanh_forward(TensorView& output)
{
    float* values = output.as<float>();
    const int size = to_int(output.size());

    if (fast_vml_enabled())
        vmsTanh(size, values, values, VML_EP);
    else
        vsTanh(size, values, values);
}

static void linear_gemm(const TensorView& input,
                        const TensorView& weights,
                        TensorView& output,
                        int m,
                        int n,
                        int k)
{
    if (packed_gemm_enabled() && n > 1)
    {
        cblas_sgemm_compute(CblasRowMajor,
                            CblasNoTrans,
                            CblasPacked,
                            m,
                            n,
                            k,
                            input.as<float>(),
                            k,
                            get_packed_b(weights, m, n, k),
                            n,
                            0.0f,
                            output.as<float>(),
                            n);
        return;
    }

    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                m,
                n,
                k,
                1.0f,
                input.as<float>(),
                k,
                weights.as<float>(),
                n,
                0.0f,
                output.as<float>(),
                n);
}

static void add_bias(TensorView& output, const TensorView& bias, Index rows, Index columns, bool fuse_relu)
{
    float* y = output.as<float>();
    const float* b = bias.as<float>();

    if (!fuse_relu && columns > 1)
    {
        static thread_local vector<float> ones;
        if (ssize(ones) < rows) ones.assign(size_t(rows), 1.0f);
        cblas_sger(CblasRowMajor,
                   to_int(rows),
                   to_int(columns),
                   1.0f,
                   ones.data(),
                   1,
                   b,
                   1,
                   y,
                   to_int(columns));
        return;
    }

    const bool parallel_bias = rows * columns >= 65536;

    #pragma omp parallel for schedule(static) if(parallel_bias)
    for (Index i = 0; i < rows; ++i)
    {
        float* row = y + i * columns;
        for (Index j = 0; j < columns; ++j)
        {
            const float value = row[j] + b[j];
            row[j] = fuse_relu ? max(value, 0.0f) : value;
        }
    }
}

bool try_activation_forward(TensorView& output, ActivationFunction function)
{
    if (function != ActivationFunction::Tanh) return false;

    tanh_forward(output);
    return true;
}

bool try_linear_forward(const TensorView& input,
                        const TensorView& weights,
                        const TensorView& bias,
                        TensorView& output,
                        cublasLtEpilogue_t epilogue)
{
    const Index input_columns = input.shape.back();
    const Index output_columns = weights.shape.back();
    const Index rows = input.size() / input_columns;

    linear_gemm(input,
                weights,
                output,
                to_int(rows),
                to_int(output_columns),
                to_int(input_columns));

    add_bias(output, bias, rows, output_columns, epilogue == CUBLASLT_EPILOGUE_RELU_BIAS);
    return true;
}

#else

bool try_activation_forward(TensorView&, ActivationFunction)
{
    return false;
}

bool try_linear_forward(const TensorView&, const TensorView&, const TensorView&, TensorView&, cublasLtEpilogue_t)
{
    return false;
}

#endif

}
