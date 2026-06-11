//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E R R O R   F U N C T I O N S   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "error_functions.h"
#include "device_backend.h"

namespace opennn
{

#ifdef OPENNN_HAS_CUDA

static float sum_squared_diff_cuda(const TensorView& input, const TensorView& target, float* workspace)
{
    const int total_size = to_int(input.size());

    input.dispatch([&](auto tag) {
        using TIn = decltype(tag);
        diff_to_fp32_cuda<TIn>(input.size(), input.as<TIn>(), target.as_float(), workspace);
    });

    float sum_squared = 0.0f;
    CHECK_CUBLAS(cublasSdot(Backend::get_cublas_handle(), total_size,
                            workspace, 1, workspace, 1, &sum_squared));
    return sum_squared;
}

static void scaled_diff_cuda(const TensorView& input, const TensorView& target, float scale, const TensorView& output)
{
    visit_type_pair<Type::FP32, Type::BF16>(input.type, output.type, [&](auto in, auto out) {
        using TIn  = typename decltype(in)::type;
        using TOut = typename decltype(out)::type;
        scaled_diff_cuda_typed<TIn, TOut>(input.size(),
                                          input.as<TIn>(),
                                          target.as_float(),
                                          scale,
                                          output.as<TOut>());
    });
}

static float sum_abs_cuda(const float* data, Index size)
{
    float sum = 0.0f;
    CHECK_CUBLAS(cublasSasum(Backend::get_cublas_handle(), to_int(size), data, 1, &sum));
    return sum;
}

static float squared_norm_cuda(const float* data, Index size)
{
    float dot = 0.0f;
    CHECK_CUBLAS(cublasSdot(Backend::get_cublas_handle(), to_int(size),
                            data, 1, data, 1, &dot));
    return dot;
}

static void cross_entropy_3d_gradient_device_count_cuda(const TensorView& input,
                                                        const TensorView& target,
                                                        const TensorView& input_delta,
                                                        const float* active_tokens_count_device)
{
    const Index vocabulary_size = input.shape.back();

    input.dispatch([&](auto tag) {
        using T = decltype(tag);
        cross_entropy_3d_multiple_backward_device_count_cuda<T>(
            static_cast<size_t>(input.size()), to_int(vocabulary_size),
            input.as<T>(), target.as<float>(), input_delta.as<T>(),
            active_tokens_count_device);
    });
}

#else

#define OPENNN_CUDA_STUBS(X) \
    X(float, sum_squared_diff_cuda, (const TensorView&, const TensorView&, float*)) \
    X(void,  scaled_diff_cuda, (const TensorView&, const TensorView&, float, const TensorView&)) \
    X(float, sum_abs_cuda, (const float*, Index)) \
    X(float, squared_norm_cuda, (const float*, Index)) \
    X(void,  cross_entropy_3d_gradient_device_count_cuda, (const TensorView&, const TensorView&, const TensorView&, const float*))

#define OPENNN_CUDA_STUB(ret, name, sig) static ret name sig { throw runtime_error(#name " requires CUDA support."); }
OPENNN_CUDA_STUBS(OPENNN_CUDA_STUB)
#undef OPENNN_CUDA_STUB
#undef OPENNN_CUDA_STUBS

template<typename T> void binary_cross_entropy_cuda(const Index, float*, const float*, const T*, const float)
{ throw runtime_error("binary_cross_entropy_cuda requires CUDA support."); }

template<typename T> void binary_cross_entropy_gradient_cuda(const Index, T*, const float*, const T*, const float, const float)
{ throw runtime_error("binary_cross_entropy_gradient_cuda requires CUDA support."); }

template<typename T> void multiple_cross_entropy_cuda(const Index, float*, const float*, const T*, const float)
{ throw runtime_error("multiple_cross_entropy_cuda requires CUDA support."); }

template<typename T> void multiple_cross_entropy_gradient_cuda(const Index, T*, const float*, const T*, const float)
{ throw runtime_error("multiple_cross_entropy_gradient_cuda requires CUDA support."); }

template<typename T> void weighted_squared_error_cuda(const Index, float*, const float*, const T*, const float, const float)
{ throw runtime_error("weighted_squared_error_cuda requires CUDA support."); }

template<typename T> void weighted_squared_error_gradient_cuda(const Index, T*, const float*, const T*, const float, const float, const float)
{ throw runtime_error("weighted_squared_error_gradient_cuda requires CUDA support."); }

template<typename T> void cross_entropy_3d_multiple_forward_cuda(const Index, const int, const T*, const float*, float*, float*, float*, const float)
{ throw runtime_error("cross_entropy_3d_multiple_forward_cuda requires CUDA support."); }

template<typename T> void cross_entropy_3d_multiple_backward_cuda(const Index, const int, const T*, const float*, T*, const float)
{ throw runtime_error("cross_entropy_3d_multiple_backward_cuda requires CUDA support."); }

#endif

void mean_squared_error(const TensorView& input, const TensorView& target, float& error,
                        float* workspace_device)
{
    const Index batch_size = input.shape[0];
    if (input.is_cuda())
    {
        error = sum_squared_diff_cuda(input, target, workspace_device) / to_int(2 * batch_size);
        return;
    }
    error = (input.as_vector() - target.as_vector()).squaredNorm() / to_type(2 * batch_size);
}

void mean_squared_error_gradient(const TensorView& input, const TensorView& target, const TensorView& input_delta)
{
    const Index batch_size = input.shape[0];
    if (input.is_cuda())
    {
        scaled_diff_cuda(input, target, 1.0f / to_int(batch_size), input_delta);
        return;
    }
    input_delta.as_vector().noalias() = (input.as_vector() - target.as_vector()) / to_type(batch_size);
}

void normalized_squared_error(const TensorView& input, const TensorView& target, float coefficient, float& error,
                              float* workspace_device)
{
    if (input.is_cuda())
    {
        error = sum_squared_diff_cuda(input, target, workspace_device) / (2.0f * (coefficient + EPSILON));
        return;
    }
    error = (input.as_vector() - target.as_vector()).squaredNorm() / (2.0f * (coefficient + EPSILON));
}

void normalized_squared_error_gradient(const TensorView& input, const TensorView& target, float coefficient, const TensorView& input_delta)
{
    if (input.is_cuda())
    {
        scaled_diff_cuda(input, target, 1.0f / (static_cast<float>(coefficient) + EPSILON), input_delta);
        return;
    }
    input_delta.as_vector().noalias() = (input.as_vector() - target.as_vector()) / (coefficient + EPSILON);
}

void weighted_squared_error(const TensorView& input, const TensorView& target, float positive_weight, float negative_weight, float& error,
                            [[maybe_unused]] float* workspace_device)
{
    if (input.is_cuda()) {
        input.dispatch([&](auto tag) {
            using T = decltype(tag);
            weighted_squared_error_cuda<T>(input.size(),
                                           workspace_device,
                                           target.as<float>(),
                                           input.as<T>(),
                                           positive_weight,
                                           negative_weight);

            error = 0.5f * sum_abs_cuda(workspace_device, input.size());
        });
        return;
    }

    const auto inputs = input.as_vector().array();
    const auto targets = target.as_vector().array();
    const auto squared_error = (inputs - targets).square();
    error = 0.5f * (targets == 1.0f).select(squared_error * positive_weight, squared_error * negative_weight).sum();
}

void weighted_squared_error_gradient(const TensorView& input, const TensorView& target, float positive_weight, float negative_weight, float coefficient, const TensorView& input_delta)
{
    if (input.is_cuda()) {
        input.dispatch([&](auto tag) {
            using T = decltype(tag);
            weighted_squared_error_gradient_cuda<T>(input.size(),
                input_delta.as<T>(), target.as<float>(), input.as<T>(), positive_weight, negative_weight, coefficient);
        });
        return;
    }

    const auto inputs = input.as_vector().array();
    const auto targets = target.as_vector().array();
    const auto difference = inputs - targets;
    input_delta.as_vector().array()
        = (targets == 1.0f).select(positive_weight * difference, negative_weight * difference) * coefficient;
}

void binary_cross_entropy(const TensorView& input, const TensorView& target, float& error,
                          [[maybe_unused]] float* workspace_device)
{
    if (input.is_cuda()) {
        input.dispatch([&](auto tag) {
            using T = decltype(tag);
            binary_cross_entropy_cuda<T>(input.size(),
                workspace_device, target.as<float>(), input.as<T>(), EPSILON);
            error = sum_abs_cuda(workspace_device, input.size()) / input.shape[0];
        });
        return;
    }
    const Index samples_number = input.shape[0];

    const MatrixMap outputs = input.as_matrix();
    const MatrixMap targets = target.as_matrix();

    const auto clamped_outputs = outputs.array().cwiseMax(EPSILON).cwiseMin(1.0f - EPSILON);

    error = -(targets.array() * clamped_outputs.log() + (1.0f - targets.array()) * (1.0f - clamped_outputs).log()).sum()
            / to_type(samples_number);

    if (isnan(error) || isinf(error)) error = 10.0f;
}

void categorical_cross_entropy(const TensorView& input, const TensorView& target, float& error,
                               [[maybe_unused]] float* workspace_device)
{
    if (input.is_cuda()) {
        input.dispatch([&](auto tag) {
            using T = decltype(tag);
            multiple_cross_entropy_cuda<T>(input.size(),
                workspace_device, target.as<float>(), input.as<T>(), EPSILON);
            error = sum_abs_cuda(workspace_device, input.size()) / input.shape[0];
        });
        return;
    }
    const Index samples_number = input.shape[0];

    const MatrixMap outputs = input.as_matrix();
    const MatrixMap targets = target.as_matrix();

    error = (targets.array() * (outputs.array() + EPSILON).log()).sum() / to_type(-samples_number);

    if (isnan(error) || isinf(error)) error = 10.0f;
}

void cross_entropy_gradient(const TensorView& input, const TensorView& target, const TensorView& input_delta)
{
    if (input.is_cuda()) {
        input.dispatch([&](auto tag) {
            using T = decltype(tag);
            const Index num_classes = input.shape.back();
            const float scale = 1.0f / static_cast<float>(input.shape[0]);
            if (num_classes == 1)
                binary_cross_entropy_gradient_cuda<T>(input.size(),
                    input_delta.as<T>(), target.as<float>(), input.as<T>(), EPSILON, scale);
            else
                multiple_cross_entropy_gradient_cuda<T>(input.size(),
                    input_delta.as<T>(), target.as<float>(), input.as<T>(), scale);
        });
        return;
    }
    const Index samples_number = input.shape[0];
    const Index num_classes = input.shape.back();

    const MatrixMap outputs = input.as_matrix();
    const MatrixMap targets = target.as_matrix();
    MatrixMap gradients = input_delta.as_matrix();

    if (num_classes == 1)
        gradients.array() = (-targets.array() / (outputs.array() + EPSILON)
                             + (1.0f - targets.array()) / (1.0f - outputs.array() + EPSILON))
                            / to_type(samples_number);
    else
        gradients = (outputs - targets) / to_type(samples_number);
}

void minkowski_error(const TensorView& input, const TensorView& target, float power, float& error, float* workspace_device)
{
    throw_if(workspace_device,
             "minkowski_error: GPU implementation not available.");

    const Index batch_size = input.shape[0];
    error = (input.as_vector() - target.as_vector()).array().abs().pow(power).sum() / to_type(power * batch_size);
}

void minkowski_error_gradient(const TensorView& input,
                              const TensorView& target,
                              float power,
                              const TensorView& input_delta,
                              bool on_gpu)
{
    throw_if(on_gpu,
             "minkowski_error_gradient: GPU implementation not available.");

    const Index batch_size = input.shape[0];
    const VectorR difference_vec = input.as_vector() - target.as_vector();
    const float scale = 1.0f / to_type(batch_size);
    const float exponent = power - 1.0f;
    input_delta.as_vector().array() = scale
        * difference_vec.array().sign()
        * difference_vec.array().abs().unaryExpr([exponent](float x) { return pow(x, exponent); });
}

void cross_entropy_3d(const TensorView& input, const TensorView& target, float& error,
                      Index& active_tokens_out, Index& correct_tokens_out, float* errors_device)
{
    const Index vocabulary_size = input.shape.back();

#ifdef OPENNN_HAS_CUDA
    if (input.is_cuda()) {
        input.dispatch([&](auto tag) {
            using T = decltype(tag);
            const size_t token_count = static_cast<size_t>(input.size() / vocabulary_size);

            float* valid_mask_device   = errors_device + token_count;
            float* correct_mask_device = errors_device + 2 * token_count;

            cross_entropy_3d_multiple_forward_cuda<T>(token_count, to_int(vocabulary_size),
                input.as<T>(), target.as<float>(),
                errors_device, valid_mask_device, correct_mask_device, EPSILON);

            thread_local Buffer device_results(Device::CUDA);
            device_results.grow_to(Index(3 * sizeof(float)));
            float* device_results_ptr = device_results.as<float>();

            cublasHandle_t handle = Backend::get_cublas_handle();
            {
                device::CublasPointerModeGuard pointer_mode(handle, CUBLAS_POINTER_MODE_DEVICE);
                CHECK_CUBLAS(cublasSasum(handle, to_int(token_count), errors_device,       1, device_results_ptr + 0));
                CHECK_CUBLAS(cublasSasum(handle, to_int(token_count), valid_mask_device,   1, device_results_ptr + 1));
                CHECK_CUBLAS(cublasSasum(handle, to_int(token_count), correct_mask_device, 1, device_results_ptr + 2));
            }

            float host_results[3];
            cudaStream_t stream = Backend::get_compute_stream();
            device::copy_async(host_results, device_results_ptr,
                               3 * Index(sizeof(float)),
                               device::CopyKind::DeviceToHost,
                               stream);
            device::synchronize(stream);

            const float sum_loss      = host_results[0];
            const float active_count  = host_results[1];
            const float correct_count = host_results[2];

            active_tokens_out = static_cast<Index>(active_count);
            correct_tokens_out = static_cast<Index>(correct_count);
            error = active_count > 0 ? sum_loss / active_count : 0.0f;
        });
        return;
    }
#endif

    (void)errors_device;

    const Index token_count = input.size() / vocabulary_size;
    const MatrixMap outputs_flat = input.as_flat_matrix();
    const VectorMap targets_flat = target.as_vector();

    float total_log_loss = 0;
    Index active_tokens = 0;
    Index correct_tokens = 0;

    #pragma omp parallel for reduction(+:total_log_loss, active_tokens, correct_tokens)
    for (Index token_index = 0; token_index < token_count; ++token_index)
    {
        const Index target_index = static_cast<Index>(targets_flat(token_index));
        
        if (target_index <= 0 || target_index >= vocabulary_size) continue;

        total_log_loss -= log(outputs_flat(token_index, target_index) + EPSILON);
        ++active_tokens;

        Index best_index;
        outputs_flat.row(token_index).maxCoeff(&best_index);
        if (best_index == target_index) ++correct_tokens;
    }

    active_tokens_out = active_tokens;
    correct_tokens_out = correct_tokens;
    error = active_tokens > 0 ? total_log_loss / to_type(active_tokens) : 0.0f;
}

void cross_entropy_3d_gradient(const TensorView& input, const TensorView& target, const TensorView& input_delta,
                               Index active_tokens_count)
{
    const Index vocabulary_size = input.shape.back();

    if (input.is_cuda()) {
        input.dispatch([&](auto tag) {
            using T = decltype(tag);
            const float scale = active_tokens_count > 0 ? 1.0f / static_cast<float>(active_tokens_count) : 0.0f;
            cross_entropy_3d_multiple_backward_cuda<T>(static_cast<size_t>(input.size()), to_int(vocabulary_size),
                input.as<T>(), target.as<float>(), input_delta.as<T>(), scale);
        });
        return;
    }
    const Index token_count = input.size() / vocabulary_size;
    MatrixMap gradients_flat = input_delta.as_flat_matrix();
    const MatrixMap outputs_flat = input.as_flat_matrix();
    const VectorMap targets_flat = target.as_vector();

    const float scale = active_tokens_count > 0 ? 1.0f / to_type(active_tokens_count) : 0.0f;

    #pragma omp parallel for
    for (Index token_index = 0; token_index < token_count; ++token_index)
    {
        const Index target_index = static_cast<Index>(targets_flat(token_index));
        if (target_index > 0 && target_index < vocabulary_size)
        {
            gradients_flat.row(token_index).noalias() = scale * outputs_flat.row(token_index);
            gradients_flat(token_index, target_index) -= scale;
        }
        else
        {
            gradients_flat.row(token_index).setZero();
        }
    }
}

void cross_entropy_3d_gradient_device_count(const TensorView& input, const TensorView& target, const TensorView& input_delta,
                                            const float* active_tokens_count_device)
{
    if (input.is_cuda())
    {
        cross_entropy_3d_gradient_device_count_cuda(input, target, input_delta, active_tokens_count_device);
        return;
    }

    Index active_tokens_count = 0;
    const Index token_count = (input.size() / input.shape.back());
    const VectorMap targets_flat = target.as_vector();
    for (Index token_index = 0; token_index < token_count; ++token_index)
    {
        const Index target_index = static_cast<Index>(targets_flat(token_index));
        if (target_index > 0 && target_index < input.shape.back()) ++active_tokens_count;
    }

    cross_entropy_3d_gradient(input, target, input_delta, active_tokens_count);
}

void l1_regularization(const TensorView& parameters, float lambda, float& penalty)
{
    if (parameters.is_cuda())
    {
        penalty = lambda * sum_abs_cuda(parameters.as<float>(), parameters.size());
        return;
    }
    penalty = lambda * parameters.as_vector().lpNorm<1>();
}

void l1_regularization_gradient(const TensorView& parameters, float lambda, const TensorView& gradient)
{
#ifdef OPENNN_HAS_CUDA
    if (parameters.is_cuda())
    {
        l1_gradient_cuda<float>(parameters.size(), gradient.as<float>(), parameters.as<float>(), lambda);
        return;
    }
#endif
    gradient.as_vector().array() += lambda * parameters.as_vector().array().sign();
}

void l2_regularization(const TensorView& parameters, float lambda, float& penalty)
{
    if (parameters.is_cuda())
    {
        penalty = 0.5f * lambda * squared_norm_cuda(parameters.as<float>(), parameters.size());
        return;
    }
    penalty = 0.5f * lambda * parameters.as_vector().squaredNorm();
}

void l2_regularization_gradient(const TensorView& parameters, float lambda, const TensorView& gradient)
{
#ifdef OPENNN_HAS_CUDA
    if (parameters.is_cuda())
    {
        const int total_size = to_int(parameters.size());
        CHECK_CUBLAS(cublasAxpyEx(Backend::get_cublas_handle(), total_size,
                                  &lambda,         CUDA_R_32F,
                                  parameters.data, CUDA_R_32F, 1,
                                  gradient.data,   CUDA_R_32F, 1,
                                  CUDA_R_32F));
        return;
    }
#endif
    gradient.as_vector().noalias() += lambda * parameters.as_vector();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
