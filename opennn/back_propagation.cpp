//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A C K   P R O P A G A T I O N   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "back_propagation.h"
#include "loss.h"
#include "neural_network.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

BackPropagation::BackPropagation(const Index new_batch_size, Loss* new_loss)
{
    set(new_batch_size, new_loss);
}

void BackPropagation::set(const Index new_batch_size, Loss* new_loss)
{
    batch_size = new_batch_size;
    loss = new_loss;

    if(!loss)
        throw runtime_error("BackPropagation error: loss is not set.");

    const NeuralNetwork* neural_network = loss->get_neural_network();

    if(!neural_network)
        throw runtime_error("BackPropagation error: neural network is not set.");

    const size_t layers_number = neural_network->get_layers_number();

    const vector<vector<Shape>> parameter_shapes = neural_network->get_parameter_shapes();

    Index total_parameters_size = 0;

    for(const auto& layer_shapes : parameter_shapes)
        for(const Shape& s : layer_shapes)
            total_parameters_size += get_aligned_size(s.size());

    gradient.resize(total_parameters_size);
    gradient.setZero();

    gradient_views.resize(layers_number);
    type* g_ptr = (total_parameters_size > 0) ? gradient.data() : nullptr;

    for(size_t i = 0; i < layers_number; ++i)
    {
        const vector<Shape>& layer_param_shapes = parameter_shapes[i];
        gradient_views[i].resize(layer_param_shapes.size());

        for(size_t j = 0; j < layer_param_shapes.size(); ++j)
        {
            const Shape& s = layer_param_shapes[j];
            if(s.size() > 0 && g_ptr)
            {
                gradient_views[i][j] = TensorView(g_ptr, s);
                g_ptr += get_aligned_size(s.size());
            }
        }
    }

    const vector<vector<Shape>> backward_shapes = neural_network->get_backward_shapes(batch_size);

    Index total_backward_size = 0;

    for(const auto& layer_shapes : backward_shapes)
        for(const Shape& s : layer_shapes)
            total_backward_size += get_aligned_size(s.size());

    backward.resize(total_backward_size);
    backward.setZero();

    backward_views.resize(layers_number);
    type* b_ptr = (total_backward_size > 0) ? backward.data() : nullptr;

    for(size_t i = 0; i < layers_number; ++i)
    {
        const vector<Shape>& shapes = backward_shapes[i];
        const size_t slots = shapes.size();

        backward_views[i].resize(slots + 1);
        backward_views[i][0].resize(1);

        for(size_t j = 0; j < slots; ++j)
        {
            const Shape& s = shapes[j];
            backward_views[i][j + 1].resize(1);

            if(s.size() > 0 && b_ptr)
            {
                backward_views[i][j + 1][0] = TensorView(b_ptr, s);
                b_ptr += get_aligned_size(s.size());
            }
        }
    }

    const Shape output_shape = neural_network->get_output_shape();
    const Index outputs_number = output_shape[0];

    loss_value = type(0);
    error = type(0);
    built_mask = false;
    accuracy.setZero();

    errors.resize(batch_size, outputs_number);

    output_gradient_dimensions = Shape({batch_size}).append(output_shape);

    const Index total_output_elements = output_shape.size() * batch_size;
    output_gradients.resize(total_output_elements);
    output_gradients.setZero();

    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();
    const auto& layer_output_indices = neural_network->get_layer_output_indices();
    const auto& layer_input_indices = neural_network->get_layer_input_indices();

    for(size_t i = 0; i < layers_number; ++i)
    {
        if(backward_views[i].empty()) continue;

        if(static_cast<Index>(i) == last_trainable_layer_index)
        {
            backward_views[i][0][0] = TensorView(output_gradients.data(), output_gradient_dimensions);
        }
        else
        {
            for(const Index consumer_idx : layer_output_indices[i])
            {
                if(consumer_idx >= 0 && consumer_idx < layers_number)
                {
                    const auto& consumer_inputs = layer_input_indices[consumer_idx];

                    Index port = 0;

                    for(size_t p = 0; p < consumer_inputs.size(); ++p)
                        if(consumer_inputs[p] == i)
                        {
                            port = static_cast<Index>(p);
                            break;
                        }

                    if(backward_views[consumer_idx].size() > 1 && !backward_views[consumer_idx][1].empty())
                        backward_views[i][0][0] = backward_views[consumer_idx][1][0];
                }
                break;
            }
        }
    }
}

void BackPropagation::allocate_device()
{
#ifdef CUDA
    if(!loss)
        throw runtime_error("BackPropagation error: loss is not set.");

    const NeuralNetwork* neural_network = loss->get_neural_network();
    if(!neural_network)
        throw runtime_error("BackPropagation error: neural network is not set.");

    const size_t layers_number = neural_network->get_layers_number();
    const Shape output_shape = neural_network->get_output_shape();
    const Index outputs_number = output_shape[0];

    gradient.resize_device(gradient.size());
    gradient.setZero_device();
    backward.resize_device(backward.size());
    backward.setZero_device();
    output_gradients.resize_device(output_gradients.size());
    output_gradients.setZero_device();

    const vector<vector<Shape>> parameter_shapes = neural_network->get_parameter_shapes();

    if(gradient.size() > 0)
    {
        type* dev_g_ptr = gradient.device();

        for(size_t i = 0; i < layers_number; ++i)
        {
            const vector<Shape>& layer_param_shapes = parameter_shapes[i];

            for(size_t j = 0; j < layer_param_shapes.size(); ++j)
            {
                const Shape& s = layer_param_shapes[j];
                if(s.size() > 0 && j < gradient_views[i].size())
                {
                    gradient_views[i][j].data = dev_g_ptr;
                    gradient_views[i][j].set_descriptor(s);
                    dev_g_ptr += get_aligned_size(s.size());
                }
            }
        }
    }

    const vector<vector<Shape>> backward_shapes = neural_network->get_backward_shapes(batch_size);
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();
    const auto& layer_output_indices = neural_network->get_layer_output_indices();
    const auto& layer_input_indices = neural_network->get_layer_input_indices();

    if(backward.size() > 0)
    {
        type* dev_b_ptr = backward.device();

        for(size_t i = 0; i < layers_number; ++i)
        {
            const vector<Shape>& shapes = backward_shapes[i];

            for(size_t j = 0; j < shapes.size(); ++j)
            {
                const Shape& s = shapes[j];
                if(s.size() > 0)
                {
                    backward_views[i][j + 1][0].data = dev_b_ptr;
                    backward_views[i][j + 1][0].set_descriptor(s);
                    dev_b_ptr += get_aligned_size(s.size());
                }
            }
        }

        for(size_t i = 0; i < layers_number; ++i)
        {
            if(backward_views[i].empty()) continue;

            if(static_cast<Index>(i) == last_trainable_layer_index)
            {
                TensorView og_view(output_gradients.device(), output_gradient_dimensions);
                og_view.set_descriptor(output_gradient_dimensions);
                backward_views[i][0][0] = og_view;
            }
            else
            {
                for(const Index consumer_idx : layer_output_indices[i])
                {
                    if(consumer_idx >= 0 && consumer_idx < layers_number)
                    {
                        if(backward_views[consumer_idx].size() > 1 && !backward_views[consumer_idx][1].empty())
                            backward_views[i][0][0] = backward_views[consumer_idx][1][0];
                    }
                    break;
                }
            }
        }
    }

    CHECK_CUDA(cudaMalloc(&errors_device, batch_size * outputs_number * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&error_device, 2 * sizeof(float)));
#endif
}


#ifdef CUDA

TensorView BackPropagation::get_output_gradients_device() const
{
    TensorView tv(const_cast<type*>(output_gradients.device()), output_gradient_dimensions);
    tv.set_descriptor(output_gradient_dimensions);
    return tv;
}

void BackPropagation::free_cuda()
{
    if(errors_device) { cudaFree(errors_device); errors_device = nullptr; }
    if(error_device) { cudaFree(error_device); error_device = nullptr; }
}

#endif


const NeuralNetwork* BackPropagation::get_neural_network() const
{
    return neural_network;
}

vector<vector<TensorView>> BackPropagation::get_layer_gradients() const
{
    const NeuralNetwork* neural_network_ptr = loss->get_neural_network();

    const size_t layers_number = neural_network_ptr->get_layers_number();

    vector<vector<TensorView>> layer_gradient_views(layers_number);

    return layer_gradient_views;
}

TensorView BackPropagation::get_output_gradients() const
{
    return {const_cast<type*>(output_gradients.data()), output_gradient_dimensions};
}

void BackPropagation::print() const
{
    cout << "Back-propagation" << "\n"
         << "Errors:" << "\n"
         << errors << "\n"
         << "Error:" << "\n"
         << error << "\n"
         << "Loss:" << "\n"
         << loss << "\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
