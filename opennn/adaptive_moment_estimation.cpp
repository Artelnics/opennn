//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "dataset.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "loss.h"
#include "profiler.h"
#include "batch.h"
#include "configuration.h"
#include "device_backend.h"
#include "adaptive_moment_estimation.h"

namespace opennn
{

#ifdef OPENNN_HAS_CUDA

static void update_parameters_cuda(NeuralNetwork* neural_network,
                                   BackPropagation& back_propagation,
                                   OptimizerData& optimization_data,
                                   float beta_1,
                                   float beta_2,
                                   float learning_rate,
                                   float bias_correction_1,
                                   float bias_correction_2)
{
    PROFILE_SCOPE("optim:adam_update_cuda");
    const Index parameters_number = neural_network->get_parameters_size();

    adam_update_cuda(
        parameters_number,
        neural_network->get_parameters_data(),
        optimization_data.views[AdaptiveMomentEstimation::GradientMoment].as<float>(),
        optimization_data.views[AdaptiveMomentEstimation::SquareGradientMoment].as<float>(),
        back_propagation.gradient.as<float>(),
        beta_1,
        beta_2,
        learning_rate,
        EPSILON,
        bias_correction_1,
        bias_correction_2,
        neural_network->get_parameters_bf16_mirror_data());
}

#else

static void update_parameters_cuda(NeuralNetwork*,
                                   BackPropagation&,
                                   OptimizerData&,
                                   float,
                                   float,
                                   float,
                                   float,
                                   float)
{
    throw runtime_error("update_parameters_cuda requires CUDA support.");
}

#endif

static void accumulate_scaled_gradient(Buffer& accumulator, Buffer& gradient, float alpha)
{
#ifdef OPENNN_HAS_CUDA
    if (accumulator.device_type == Device::CUDA)
    {
        CHECK_CUBLAS(cublasSaxpy(Backend::get_cublas_handle(),
                                 int(gradient.size_in_floats()), &alpha,
                                 gradient.as<float>(), 1,
                                 accumulator.as<float>(), 1));
        return;
    }
#endif
    VectorMap(accumulator.as<float>(), accumulator.size_in_floats()).noalias()
        += alpha * VectorMap(gradient.as<float>(), gradient.size_in_floats());
}

AdaptiveMomentEstimation::AdaptiveMomentEstimation(Loss* new_loss)
    : Optimizer(new_loss)
{
    set_default();
}

void AdaptiveMomentEstimation::set_batch_size(const Index new_batch_size)
{
    batch_size = new_batch_size;
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

void AdaptiveMomentEstimation::set_default()
{
    batch_size = 0;
    display_period = 100;
    name = "AdaptiveMomentEstimation";
}

void AdaptiveMomentEstimation::set_learning_rate(const float new_learning_rate)
{
    learning_rate = new_learning_rate;
}

void AdaptiveMomentEstimation::setup_optimizer_data(OptimizerData& optimization_data,
                                                    Index parameters_number,
                                                    Device device,
                                                    [[maybe_unused]] bool on_gpu)
{
    optimization_data.set({Shape{parameters_number}, Shape{parameters_number}}, device);

    optimization_data.iteration = 0;

    throw_if(update_period > 1 && use_cuda_graph,
             "gradient accumulation is not supported with the CUDA graph.");

    accumulated_batches = 0;
    if (update_period > 1)
    {
        gradient_accumulator.resize_bytes(parameters_number * Index(sizeof(float)), device);
        gradient_accumulator.setZero();
    }

#ifdef OPENNN_HAS_CUDA
    if (on_gpu)
    {
        optimization_data.graph_step.resize_bytes(Index(sizeof(int)), Device::CUDA);
        optimization_data.graph_step.setZero();
        optimization_data.graph_effective_lr.resize_bytes(Index(sizeof(float)), Device::CUDA);
        optimization_data.graph_effective_eps.resize_bytes(Index(sizeof(float)), Device::CUDA);

        graph_update = [this, &optimization_data](BackPropagation& back_propagation) {
            update_parameters_capturable(back_propagation, optimization_data);
        };
    }
#endif
}

void AdaptiveMomentEstimation::update_parameters(BackPropagation& back_propagation,
                                                 OptimizerData& optimization_data)
{
    const Index period = max(Index(1), update_period);

    if (period > 1)
    {
        accumulate_scaled_gradient(gradient_accumulator, back_propagation.gradient,
                                   1.0f / float(period));

        if (++accumulated_batches < period) return;

        device::copy_async(back_propagation.gradient.data, gradient_accumulator.data,
                           gradient_accumulator.bytes,
                           gradient_accumulator.device_type, gradient_accumulator.device_type,
                           gradient_accumulator.device_type == Device::CUDA ? device::get_compute_stream() : nullptr);
        gradient_accumulator.setZero();
        accumulated_batches = 0;
    }

    NeuralNetwork* neural_network = loss->get_neural_network();

    optimization_data.iteration++;

    {
        PROFILE_SCOPE("optim:clip_gradient_norm");
        clip_gradient_norm(back_propagation.gradient, gradient_clip_norm);
    }

    const float iteration = static_cast<float>(optimization_data.iteration);

    const float bias_correction_1 = 1.0f - pow(beta_1, iteration);
    const float bias_correction_2 = 1.0f - pow(beta_2, iteration);

    if (neural_network->is_gpu())
    {
        update_parameters_cuda(neural_network, back_propagation, optimization_data,
                               beta_1, beta_2, learning_rate,
                               bias_correction_1, bias_correction_2);
        return;
    }

    VectorMap parameters(neural_network->get_parameters_data(),
                         neural_network->get_parameters_size());

    VectorMap gradient_exponential_decay = optimization_data.views[GradientMoment].as_vector();
    VectorMap square_gradient_exponential_decay = optimization_data.views[SquareGradientMoment].as_vector();

    VectorMap gradient(back_propagation.gradient.as<float>(),
                       back_propagation.gradient.size_in_floats());

    const Index parameters_size = parameters.size();
    const float one_minus_beta_1 = 1.0f - beta_1;
    const float one_minus_beta_2 = 1.0f - beta_2;

    const float sqrt_bias_correction_2 = sqrt(bias_correction_2);
    const float effective_learning_rate = learning_rate * sqrt_bias_correction_2 / bias_correction_1;
    const float effective_epsilon = EPSILON * sqrt_bias_correction_2;

    #pragma omp parallel for if(parameters_size > 4096)
    for (Index i = 0; i < parameters_size; ++i)
    {
        const float gradient_value = gradient(i);

        auto& first_moment = gradient_exponential_decay(i);
        auto& second_moment = square_gradient_exponential_decay(i);

        first_moment = beta_1 * first_moment + one_minus_beta_1 * gradient_value;
        second_moment = beta_2 * second_moment + one_minus_beta_2 * gradient_value * gradient_value;

        parameters(i) -= effective_learning_rate * first_moment / (sqrt(second_moment) + effective_epsilon);
    }
}

#ifdef OPENNN_HAS_CUDA
void AdaptiveMomentEstimation::update_parameters_capturable(BackPropagation& back_propagation,
                                                            OptimizerData& optimization_data) const
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    clip_gradient_norm(back_propagation.gradient, gradient_clip_norm);

    adam_update_capturable_cuda(
        neural_network->get_parameters_size(),
        neural_network->get_parameters_data(),
        optimization_data.views[GradientMoment].as<float>(),
        optimization_data.views[SquareGradientMoment].as<float>(),
        back_propagation.gradient.as<float>(),
        beta_1, beta_2, learning_rate, EPSILON,
        optimization_data.graph_step.as<int>(),
        optimization_data.graph_effective_lr.as<float>(),
        optimization_data.graph_effective_eps.as<float>(),
        neural_network->get_parameters_bf16_mirror_data(),
        device::get_compute_stream());
}
#else
void AdaptiveMomentEstimation::update_parameters_capturable(BackPropagation&, OptimizerData&) const
{
    throw runtime_error("update_parameters_capturable requires CUDA support.");
}
#endif

void AdaptiveMomentEstimation::to_JSON(JsonWriter& printer) const
{
    printer.open_element("AdaptiveMomentEstimation");

    add_json_field(printer, "BatchSize", to_string(batch_size));
    add_json_field(printer, "LearningRate", to_string(learning_rate));
    add_json_field(printer, "Beta1", to_string(beta_1));
    add_json_field(printer, "Beta2", to_string(beta_2));
    write_common_json(printer);

    printer.close_element();
}

void AdaptiveMomentEstimation::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "AdaptiveMomentEstimation");

    set_batch_size(read_json_index(root_element, "BatchSize"));
    if (root_element->has("LearningRate"))     set_learning_rate(read_json_float(root_element, "LearningRate"));
    if (root_element->has("Beta1"))            set_beta_1(read_json_float(root_element, "Beta1"));
    if (root_element->has("Beta2"))            set_beta_2(read_json_float(root_element, "Beta2"));
    read_common_json(root_element);
}

REGISTER(Optimizer, AdaptiveMomentEstimation, "AdaptiveMomentEstimation");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
