//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <vector>

#include "layer.h"
#include "random_utilities.h"

namespace opennn
{


void LayerForwardPropagation::set(const Index new_batch_size, Layer* new_layer)
{
    if(!new_layer) return;

    batch_size = new_batch_size;
    layer = new_layer;

    initialize();
}


void LayerBackPropagation::set(const Index new_batch_size, Layer* new_layer)
{
    if(!new_layer) return;

    batch_size = new_batch_size;
    layer = new_layer;

    initialize();
}


void LayerBackPropagationLM::set(const Index new_batch_size, Layer* new_layer)
{
    if(!new_layer) return;

    batch_size = new_batch_size;
    layer = new_layer;

    initialize();
}


TensorView LayerForwardPropagation::get_outputs() const
{
    return outputs;
}

vector<TensorView*> LayerForwardPropagation::get_workspace_views()
{
    return {&outputs};
}


vector<TensorView *> LayerBackPropagation::get_gradient_views()
{
    return vector<TensorView*>();
}


vector<TensorView> LayerBackPropagation::get_input_gradients() const
{
    return input_gradients;
}


vector<TensorView *> LayerBackPropagationLM::get_gradient_views()
{
    return vector<TensorView*>();
}


vector<TensorView> LayerBackPropagationLM::get_input_gradients() const
{
    return input_gradients;
}


#ifdef OPENNN_CUDA

void LayerForwardPropagationCuda::set(const Index new_batch_size, Layer* new_layer)
{
    if(!new_layer) return;

    batch_size = new_batch_size;
    layer = new_layer;

    initialize();
}


void LayerBackPropagationCuda::set(const Index new_batch_size, Layer* new_layer)
{
    if(!new_layer) return;

    batch_size = new_batch_size;
    layer = new_layer;

    initialize();
}


TensorViewCuda LayerForwardPropagationCuda::get_outputs() const
{
    return outputs;
}


vector<TensorViewCuda*> LayerForwardPropagationCuda::get_workspace_views()
{
    return { &outputs };
}


vector<TensorViewCuda> LayerBackPropagationCuda::get_input_gradient_views() const
{
    vector<TensorViewCuda> views;
    views.reserve(input_gradients.size());

    for (const TensorCuda& tensor : input_gradients)
        views.push_back(tensor.view());

    return views;
}

#endif


Layer::Layer()
{
}


Layer::~Layer() = default;


bool Layer::get_display() const
{
    return display;
}


const string& Layer::get_label() const
{
    return label;
}


const string& Layer::get_name() const
{
    return name;
}


void Layer::set_label(const string& new_label)
{
    label = new_label;
}


void Layer::set_display(bool new_display)
{
    display = new_display;
}


void Layer::set_parameters_random()
{
    const vector<TensorView*> parameter_views = get_parameter_views();

    for(const auto& view : parameter_views)
    {
        VectorMap this_parameters(view->data, view->size());

        set_random_uniform(this_parameters);
    }
}


void Layer::set_parameters_glorot()
{
    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    const type limit = sqrt(6.0 / (inputs_number + outputs_number));

    const vector<TensorView*> parameter_views = get_parameter_views();

    for(const TensorView* view : parameter_views)
    {
        VectorMap this_parameters(view->data, view->size());

        set_random_uniform(this_parameters, -limit, limit);
    }
}


Index Layer::get_parameters_number()
{
    vector<TensorView*> parameter_views = get_parameter_views();

    Index parameters_number = 0;

    for (const auto* view : parameter_views)
        parameters_number += view->size();

    return parameters_number;
}


string Layer::get_expression(const vector<string>&, const vector<string>&) const
{
    return string();
}


vector<string> Layer::get_default_feature_names() const
{
    const Index inputs_number = get_inputs_number();

    vector<string> input_names(inputs_number);

    for(Index i = 0; i < inputs_number; i++)
        input_names[i] = "input_" + to_string(i);

    return input_names;
}


vector<string> Layer::get_default_output_names() const
{
    const Index outputs_number = get_outputs_number();

    vector<string> output_names(outputs_number);

    for(Index i = 0; i < outputs_number; i++)
        output_names[i] = "output_" + to_string(i);

    return output_names;
}


bool Layer::get_is_trainable() const
{
    return is_trainable;
}


void Layer::add_gradients(const vector<TensorView>& output_gradient_views) const
{
    TensorMap3 output_gradients = tensor_map<3>(output_gradient_views[0]);

    for(Index i = 1; i < Index(output_gradient_views.size()); i++)
        output_gradients.device(get_device()) += tensor_map<3>(output_gradient_views[i]);
}


Index Layer::get_inputs_number() const
{
    const Shape input_shape = get_input_shape();

    return input_shape.count();
}


Index Layer::get_outputs_number() const
{
    const Shape output_shape = get_output_shape();

    return accumulate(output_shape.begin(), output_shape.end(), 1, multiplies<Index>());
}


void Layer::forward_propagate(const vector<TensorView>&,
                              unique_ptr<LayerForwardPropagation>&, bool)
{
    throw runtime_error("This method is not implemented in the layer type (" + name + ").\n");
}



void Layer::set_input_shape(const Shape&)
{
    throw runtime_error("This method is not implemented in the layer type (" + name + ").\n");
}


void Layer::set_output_shape(const Shape&)
{
    throw runtime_error("This method is not implemented in the layer type (" + name + ").\n");
}


void Layer::softmax(MatrixMap y) const
{
    y.colwise() -= y.rowwise().maxCoeff();

    y = y.array().exp();

    y.array().colwise() /= y.rowwise().sum().array();
}


void Layer::softmax(TensorMap3 y) const
{
    const Index rows_number = y.dimension(0);
    const Index columns_number = y.dimension(1);
    const Index channels = y.dimension(2);

    #pragma omp parallel for collapse(2)
    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j = 0; j < columns_number; j++)
        {
            type max_value = -numeric_limits<type>::infinity();

            for(Index k = 0; k < channels; k++)
                if(y(i, j, k) > max_value)
                    max_value = y(i, j, k);

            type sum = 0.0;
            for(Index k = 0; k < channels; k++)
            {
                y(i, j, k) = exp(y(i, j, k) - max_value);
                sum += y(i, j, k);
            }

            if(sum > 0.0)
            {
                const type inv_sum = type(1.0) / sum;

                for(Index k = 0; k < channels; k++)
                    y(i, j, k) *= inv_sum;
            }
        }
    }
}


void Layer::softmax(TensorMap4 y) const
{
    const Index rows_number = y.dimension(0);
    const Index columns_number = y.dimension(1);
    const Index channels = y.dimension(2);
    const Index blocks_number = y.dimension(3);

    #pragma omp parallel for collapse(3)
    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j = 0; j < columns_number; j++)
        {
            for(Index k = 0; k < channels; k++)
            {
                type max_value = -std::numeric_limits<type>::infinity();

                for(Index l = 0; l < blocks_number; l++)
                    if(y(i, j, k, l) > max_value)
                        max_value = y(i, j, k, l);

                type sum = 0.0;
                for(Index l = 0; l < blocks_number; l++)
                {
                    y(i, j, k, l) = exp(y(i, j, k, l) - max_value);
                    sum += y(i, j, k, l);
                }

                if(sum > 0.0)
                {
                    const type inv_sum = type(1.0) / sum;

                    for(Index l = 0; l < blocks_number; l++)
                        y(i, j, k, l) *= inv_sum;
                }
            }
        }
    }
}


void Layer::softmax_derivatives_times_tensor(const TensorMap3 softmax,
                                             TensorMap3 result,
                                             VectorMap aux_rows) const
{
    const Index rows = softmax.dimension(0);
    const Index columns = softmax.dimension(1);
    const Index depth = softmax.dimension(2);

    type* softmax_data = const_cast<type*>(softmax.data());
    type* result_data = result.data();

    for(Index i = 0; i < depth; i++)
    {
        for(Index j = 0; j < columns; j++)
        {
            const Index offset = rows * (i * columns + j);

            const VectorMap softmax_vector(softmax_data + offset, rows);
            const VectorMap tensor_vector(result_data + offset, rows);
            VectorMap result_vector(result_data + offset, rows);

            aux_rows.array() = softmax_vector.array() * tensor_vector.array();

            const type sum_val = aux_rows.sum();

            result_vector.array() = aux_rows.array() - (softmax_vector.array() * sum_val);
        }
    }
}

} 


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
