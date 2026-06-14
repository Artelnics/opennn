//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C P U   M A T H   B A C K E N D
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "cpu_math_backend.h"
#include "tensor_operations.h"
#include "string_utilities.h"

#ifdef EIGEN_USE_MKL_ALL
#include <mkl_cblas.h>
#include <mkl_vml.h>
#endif

namespace opennn::cpu_math
{

#ifdef EIGEN_USE_MKL_ALL

static bool fast_vml_enabled() { static const bool enabled = env_flag_enabled("OPENNN_MKL_FAST_VML"); return enabled; }

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
    if (function != ActivationFunction::Tanh || output.type != Type::FP32) return false;

    float* values = output.as<float>();
    const int size = to_int(output.size());

    if (fast_vml_enabled())
        vmsTanh(size, values, values, VML_EP);
    else
        vsTanh(size, values, values);

    return true;
}

bool try_linear_forward(const TensorView& input,
                        const TensorView& weights,
                        const TensorView& bias,
                        TensorView& output,
                        bool fuse_relu)
{
    if (input.type != Type::FP32
        || weights.type != Type::FP32
        || bias.type != Type::FP32
        || output.type != Type::FP32
        || input.shape.rank == 0
        || weights.shape.rank != 2
        || bias.shape.rank != 1)
        return false;

    const Index input_columns = input.shape.back();
    const Index output_columns = weights.shape.back();

    if (input_columns <= 0
        || output_columns <= 0
        || input.size() % input_columns != 0
        || weights.shape[0] != input_columns
        || bias.size() != output_columns)
        return false;

    const Index rows = input.size() / input_columns;

    if (rows <= 0 || output.size() != rows * output_columns)
        return false;

    const int m = to_int(rows);
    const int n = to_int(output_columns);
    const int k = to_int(input_columns);

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

    add_bias(output, bias, rows, output_columns, fuse_relu);
    return true;
}

#else

bool try_activation_forward(TensorView&, ActivationFunction)
{
    return false;
}

bool try_linear_forward(const TensorView&, const TensorView&, const TensorView&, TensorView&, bool)
{
    return false;
}

#endif

}
