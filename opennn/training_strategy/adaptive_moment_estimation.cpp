//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/training_strategy/adaptive_moment_estimation.h"

#include "opennn/core/device_backend.h"
#include "opennn/core/profiler.h"
#include "opennn/dataset/batch.h"
#include "opennn/dataset/dataset.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/core/string_utilities.h"
#include "opennn/training_strategy/kernel_optimizers.cuh"
#include "opennn/training_strategy/loss.h"

namespace opennn
{

namespace
{

bool bf16_first_moment_default()
{
    static const bool enabled = env_flag_enabled("OPENNN_ADAM_BF16_MOMENT", true);
    return enabled;
}

#ifdef OPENNN_HAS_CUDA
// Adam does no arithmetic worth counting; it streams. Per parameter it reads the
// gradient, reads and rewrites both moments and the parameter itself, and writes
// a BF16 mirror when one exists. Counting those passes is what puts the
// optimizer step on the same footing as the layer kernels in the OPENNN_PROFILE
// bandwidth column, and it is usually a larger share than people expect.
double adam_bytes(const vector<BackPropagation::GradientSlice>& slices,
                  bool has_mirror,
                  bool first_moment_bf16)
{
    constexpr double float_bytes = double(sizeof(float));

    const double first_moment_bytes =
        first_moment_bf16 ? double(sizeof(bfloat16)) : float_bytes;

    const double per_parameter = float_bytes                                     // gradient, read
                               + 2.0 * first_moment_bytes                        // first moment, read and written
                               + 2.0 * float_bytes                               // second moment, read and written
                               + 2.0 * float_bytes                               // parameters, read and written
                               + (has_mirror ? double(sizeof(bfloat16)) : 0.0);  // mirror, written

    double parameters = 0.0;
    for (const BackPropagation::GradientSlice& slice : slices)
        parameters += double(slice.values.size());

    return per_parameter * parameters;
}

// The first-moment slot is FP32 or BF16, so a per-slice offset has to be taken in
// the element type the slot was allocated with rather than always in floats.
void* get_moment_slice(const TensorView& first_moment, const Index offset)
{
    return first_moment.is_bf16()
        ? static_cast<void*>(first_moment.as<bfloat16>() + offset)
        : static_cast<void*>(first_moment.as<float>() + offset);
}
#endif

}

AdaptiveMomentEstimation::AdaptiveMomentEstimation(Loss* new_loss)
    : Optimizer(new_loss)
{
    display_period = 100;
    name = "AdaptiveMomentEstimation";

    // Only the first moment. At beta_1 = 0.9 its per-step increment is 0.1
    // relative, well above BF16's 3.9e-3 half-ULP; at beta_2 = 0.999 the second
    // moment's is ~1e-3, below it, so a BF16 v would stop integrating under
    // round-to-nearest. Set OPENNN_ADAM_BF16_MOMENT=0 to get the FP32 slot back.
    bf16_first_moment = bf16_first_moment_default();
}

void AdaptiveMomentEstimation::set_beta_1(const float new_beta_1)
{
    throw_if(new_beta_1 < 0.0f || new_beta_1 >= 1.0f,
             "AdaptiveMomentEstimation::set_beta_1: beta_1 must be in [0, 1).");

    beta_1 = new_beta_1;
}

void AdaptiveMomentEstimation::set_beta_2(const float new_beta_2)
{
    throw_if(new_beta_2 < 0.0f || new_beta_2 >= 1.0f,
             "AdaptiveMomentEstimation::set_beta_2: beta_2 must be in [0, 1).");

    beta_2 = new_beta_2;
}

void AdaptiveMomentEstimation::configure_for_task(NetworkTask task)
{
    static constexpr float language_model_learning_rate = 0.0001f;

    Optimizer::configure_for_task(task);

    if (task == NetworkTask::LanguageModeling)
        learning_rate = language_model_learning_rate;
}

void AdaptiveMomentEstimation::setup_optimizer_data(OptimizerData& optimization_data,
                                                    Index parameters_number,
                                                    Device device)
{
    const bool use_graph = can_use_cuda_graph();

    // Host training keeps the FP32 slot whatever the knob says: the CPU update
    // reaches the moment through a VectorMap, which is FP32 by construction, and
    // the resident bytes this would save are not on any measured host cell.
    const Type first_moment_type = bf16_first_moment && device == Device::CUDA
                                 ? Type::BF16
                                 : Type::FP32;

    optimization_data.set({Shape{parameters_number},
                           Shape{parameters_number},
                           use_graph ? Shape{4} : Shape{}},
                          {first_moment_type, Type::FP32, Type::FP32},
                          device);
    update_step = 0;

#ifdef OPENNN_HAS_CUDA
    if (use_graph)
        set_scalar_device_cuda(optimization_data.views[GraphScalars].as<float>() + 3,
                               learning_rate,
                               device::get_compute_stream());
#endif
}

void AdaptiveMomentEstimation::on_epoch_begin(Index, OptimizerData& optimization_data)
{
#ifdef OPENNN_HAS_CUDA
    if (can_use_cuda_graph() && optimization_data.views[GraphScalars].size() >= 4)
        set_scalar_device_cuda(optimization_data.views[GraphScalars].as<float>() + 3,
                               learning_rate,
                               device::get_compute_stream());
#else
    (void)optimization_data;
#endif
}

void AdaptiveMomentEstimation::update_parameters(BackPropagation& back_propagation,
                                                 OptimizerData& optimization_data,
                                                 UpdateMode mode)
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    const bool has_graph_scalars =
        optimization_data.views.size() > size_t(GraphScalars)
        && optimization_data.views[GraphScalars].size() >= 4;

    const vector<BackPropagation::GradientSlice>& gradient_slices =
        back_propagation.get_gradient_slices();

    if (mode == UpdateMode::Capturable
        || (has_graph_scalars && neural_network->is_gpu() && can_use_cuda_graph()))
    {
#ifdef OPENNN_HAS_CUDA
        clip_gradient_norm(back_propagation, gradient_clip_norm);

        float* const graph_scalars = optimization_data.views[GraphScalars].as<float>();
        int* const graph_step = reinterpret_cast<int*>(graph_scalars);
        float* const graph_learning_rate = graph_scalars + 1;
        float* const graph_epsilon = graph_scalars + 2;
        const float* const graph_base_learning_rate = graph_scalars + 3;

        cudaStream_t stream = device::get_compute_stream();
        adam_prepare_capturable_cuda(
            beta_1, beta_2, graph_base_learning_rate, EPSILON,
            graph_step, graph_learning_rate, graph_epsilon, stream);

        float* const parameters = neural_network->get_parameters_data();
        const TensorView& first_moment = optimization_data.views[GradientMoment];
        const bool first_moment_bf16 = first_moment.is_bf16();
        float* const second_moment =
            optimization_data.views[SquareGradientMoment].as<float>();
        bfloat16* const mirror =
            neural_network->get_parameters_bf16_mirror_data();

        PROFILE_SCOPE_BYTES("optim:adam_update_capturable_cuda",
                            adam_bytes(gradient_slices, mirror != nullptr, first_moment_bf16));

        for(const BackPropagation::GradientSlice& slice : gradient_slices)
        {
            const Index offset = slice.parameter_offset;
            adam_update_prepared_cuda(
                slice.values.size(),
                parameters + offset,
                get_moment_slice(first_moment, offset),
                first_moment_bf16,
                second_moment + offset,
                slice.values.as<float>(),
                beta_1, beta_2,
                graph_learning_rate, graph_epsilon,
                mirror ? mirror + offset : nullptr,
                stream);
        }
        return;
#else
        throw runtime_error("Capturable Adam parameter updates require CUDA support.");
#endif
    }

    update_step++;

    {
        PROFILE_SCOPE("optim:clip_gradient_norm");
        clip_gradient_norm(back_propagation, gradient_clip_norm);
    }

    const float step = static_cast<float>(update_step);

    const float bias_correction_1 = 1.0f - pow(beta_1, step);
    const float bias_correction_2 = 1.0f - pow(beta_2, step);

    if (neural_network->is_gpu())
    {
#ifdef OPENNN_HAS_CUDA
        float* const parameters = neural_network->get_parameters_data();
        const TensorView& first_moment = optimization_data.views[GradientMoment];
        const bool first_moment_bf16 = first_moment.is_bf16();
        float* const second_moment =
            optimization_data.views[SquareGradientMoment].as<float>();
        bfloat16* const mirror =
            neural_network->get_parameters_bf16_mirror_data();

        PROFILE_SCOPE_BYTES("optim:adam_update_cuda",
                            adam_bytes(gradient_slices, mirror != nullptr, first_moment_bf16));

        for(const BackPropagation::GradientSlice& slice : gradient_slices)
        {
            const Index offset = slice.parameter_offset;
            adam_update_cuda(
                slice.values.size(),
                parameters + offset,
                get_moment_slice(first_moment, offset),
                first_moment_bf16,
                second_moment + offset,
                slice.values.as<float>(),
                beta_1, beta_2, learning_rate, EPSILON,
                bias_correction_1, bias_correction_2,
                mirror ? mirror + offset : nullptr);
        }
        return;
#else
        throw runtime_error("Adam parameter updates on GPU require CUDA support.");
#endif
    }

    VectorMap parameters = neural_network->get_parameters_map();

    VectorMap gradient_exponential_decay = optimization_data.views[GradientMoment].as_vector();
    VectorMap square_gradient_exponential_decay = optimization_data.views[SquareGradientMoment].as_vector();

    const float one_minus_beta_1 = 1.0f - beta_1;
    const float one_minus_beta_2 = 1.0f - beta_2;

    const float sqrt_bias_correction_2 = sqrt(bias_correction_2);
    const float effective_learning_rate = learning_rate * sqrt_bias_correction_2 / bias_correction_1;
    const float effective_epsilon = EPSILON * sqrt_bias_correction_2;

    const auto update_range = [&](const Index offset,
                                  const VectorMap& gradient)
    {
        PROFILE_SCOPE_HOST("optim:adam_update_cpu");

        const Index range_size = gradient.size();
        #pragma omp parallel for if(range_size > 65536)
        for (Index i = 0; i < range_size; ++i)
        {
            const float gradient_value = gradient(i);

            auto& first_moment = gradient_exponential_decay(offset + i);
            auto& second_moment = square_gradient_exponential_decay(offset + i);

            first_moment = beta_1 * first_moment + one_minus_beta_1 * gradient_value;
            second_moment = beta_2 * second_moment + one_minus_beta_2 * gradient_value * gradient_value;

            parameters(offset + i) -= effective_learning_rate * first_moment
                                    / (sqrt(second_moment) + effective_epsilon);
        }
    };

    for(const BackPropagation::GradientSlice& slice : gradient_slices)
    {
        update_range(slice.parameter_offset,
                     slice.values.as_vector());
    }
}

void AdaptiveMomentEstimation::to_JSON(JsonWriter& printer) const
{
    printer.open_element("AdaptiveMomentEstimation");

    add_json_field(printer, "BatchSize", batch_size);
    add_json_field(printer, "LearningRate", learning_rate);
    add_json_field(printer, "Beta1", beta_1);
    add_json_field(printer, "Beta2", beta_2);
    add_json_field(printer, "BF16FirstMoment", bf16_first_moment);
    write_common_json(printer);

    printer.close_element();
}

void AdaptiveMomentEstimation::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "AdaptiveMomentEstimation");

    set_batch_size(read_json_index(root_element, "BatchSize"));
    set_learning_rate(read_json_float(root_element, "LearningRate", learning_rate));
    set_beta_1(read_json_float(root_element, "Beta1", beta_1));
    set_beta_2(read_json_float(root_element, "Beta2", beta_2));
    set_bf16_first_moment(read_json_bool(root_element, "BF16FirstMoment", bf16_first_moment));
    read_common_json(root_element);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
