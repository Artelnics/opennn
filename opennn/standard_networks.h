//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   STANDARD   N E T W O R K   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef STANDARDNETWORKS_H
#define STANDARDNETWORKS_H

#include "multihead_attention_layer.h"
#include "scaling_layer_2d.h"
#include "scaling_layer_3d.h"
#include "scaling_layer_4d.h"
#include "unscaling_layer.h"
#include "dense_layer.h"
#include "bounding_layer.h"
#include "recurrent_layer.h"
#include "embedding_layer.h"
#include "convolutional_layer.h"
#include "pooling_layer.h"
#include "pooling_layer_3d.h"
#include "flatten_layer.h"
#include "addition_layer.h"
#include "neural_network.h"
#include "multihead_attention_layer.h"

namespace opennn
{

class ApproximationNetwork : public NeuralNetwork
{
public:

    ApproximationNetwork(const dimensions& input_dimensions,
                         const dimensions& complexity_dimensions,
                         const dimensions& output_dimensions) : NeuralNetwork()
    {
        const Index complexity_size = complexity_dimensions.size();

        add_layer(make_unique<Scaling2d>(input_dimensions));

        for (Index i = 0; i < complexity_size; i++)
            add_layer(make_unique<Dense2d>(get_output_dimensions(),
                                           dimensions{ complexity_dimensions[i] },
                                           "RectifiedLinear",
                                           false,
                                           "dense2d_layer_" + to_string(i + 1)));

        add_layer(make_unique<Dense2d>(get_output_dimensions(),
                                       output_dimensions,
                                       "Linear",
                                       false,
                                       "approximation_layer"));

        add_layer(make_unique<Unscaling>(output_dimensions));

        add_layer(make_unique<Bounding>(output_dimensions));

        const Index features_number = get_features_number();
        feature_names.resize(features_number);

        const Index outputs_number = get_outputs_number();
        output_names.resize(outputs_number);
    }
};


class ClassificationNetwork : public NeuralNetwork
{
public:

    ClassificationNetwork(const dimensions& input_dimensions,
                          const dimensions& complexity_dimensions,
                          const dimensions& output_dimensions) : NeuralNetwork()
    {
        const Index complexity_size = complexity_dimensions.size();

        add_layer(make_unique<Scaling2d>(input_dimensions));
        
        for (Index i = 0; i < complexity_size; i++)
            add_layer(make_unique<Dense2d>(get_output_dimensions(),
                                           dimensions{complexity_dimensions[i]},
                                           "HyperbolicTangent",
                                           false,
                                           "dense2d_layer_" + to_string(i + 1)));
                                           
        add_layer(make_unique<Dense2d>(get_output_dimensions(),
                                       output_dimensions,
                                       "Softmax",
                                       false,
                                       "classification_layer"));

        const Index features_number = get_features_number();
        feature_names.resize(features_number);

        const Index outputs_number = get_outputs_number();
        output_names.resize(outputs_number);
    }
};


class ForecastingNetwork : public NeuralNetwork
{
public:

    ForecastingNetwork(const dimensions& input_dimensions,
                       const dimensions& complexity_dimensions,
                       const dimensions& output_dimensions) : NeuralNetwork()
    {
        add_layer(make_unique<Scaling3d>(input_dimensions));

        add_layer(make_unique<Recurrent>(input_dimensions,
                                         complexity_dimensions));

        add_layer(make_unique<Dense2d>(complexity_dimensions,
                                       output_dimensions,
                                       "Linear",
                                       "dense_layer"));

        add_layer(make_unique<Unscaling>(output_dimensions));

        add_layer(make_unique<Bounding>(output_dimensions));

        const Index features_number = get_features_number();
        feature_names.resize(features_number);

        const Index outputs_number = get_outputs_number();
        output_names.resize(outputs_number);
    }
};


class AutoAssociationNetwork : public NeuralNetwork
{
public:

    AutoAssociationNetwork(const dimensions& input_dimensions,
                           const dimensions& complexity_dimensions,
                           const dimensions& output_dimensions) : NeuralNetwork()
    {
        add_layer(make_unique<Scaling2d>(input_dimensions));

        const Index mapping_neurons_number = 10;
        const Index bottle_neck_neurons_number = complexity_dimensions[0];

        add_layer(make_unique<Dense2d>(input_dimensions,
                                       dimensions{mapping_neurons_number},
                                       "HyperbolicTangent",
                                       false,
                                       "mapping_layer"));

        add_layer(make_unique<Dense2d>(dimensions{ mapping_neurons_number },
                                       dimensions{ bottle_neck_neurons_number },
                                       "Linear",
                                       false,
                                       "bottleneck_layer"));

        add_layer(make_unique<Dense2d>(dimensions{ bottle_neck_neurons_number },
                                       dimensions{ mapping_neurons_number },
                                       "HyperbolicTangent",
                                       false,
                                       "demapping_layer"));

        add_layer(make_unique<Dense2d>(dimensions{ mapping_neurons_number },
                                       dimensions{ output_dimensions },
                                       "Linear",
                                       false,
                                       "output_layer"));

        add_layer(make_unique<Unscaling>(output_dimensions));
    }
};


class ImageClassificationNetwork : public NeuralNetwork
{
public:

    ImageClassificationNetwork(const dimensions& input_dimensions,
                               const dimensions& complexity_dimensions,
                               const dimensions& output_dimensions) : NeuralNetwork()
    {
        if (input_dimensions.size() != 3)
            throw runtime_error("Input dimensions size is not 3.");

        reference_all_layers();

        add_layer(make_unique<Scaling4d>(input_dimensions));

        const Index complexity_size = complexity_dimensions.size();

        for (Index i = 0; i < complexity_size; i++)
        {
            const dimensions kernel_dimensions = { 3, 3, get_output_dimensions()[2], complexity_dimensions[i] };
            const dimensions stride_dimensions = { 1, 1 };

            add_layer(make_unique<Convolutional>(get_output_dimensions(),
                                                 kernel_dimensions,
                                                 "RectifiedLinear",
                                                 stride_dimensions,
                                                 "Same",
                                                 false, // Batch normalization
                                                 "convolutional_layer_" + to_string(i + 1)));
            
            const dimensions pool_dimensions = { 2, 2 };
            const dimensions pooling_stride_dimensions = { 2, 2 };
            const dimensions padding_dimensions = { 0, 0 };

            add_layer(make_unique<Pooling>(get_output_dimensions(),
                                           pool_dimensions,
                                           pooling_stride_dimensions,
                                           padding_dimensions,
                                           "MaxPooling",
                                           "pooling_layer_" + to_string(i + 1)));
        }

        add_layer(make_unique<Flatten<4>>(get_output_dimensions()));

        add_layer(make_unique<Dense2d>(get_output_dimensions(),
                                       output_dimensions,
                                       "Softmax",
                                       false, // Batch normalization
                                       "dense_2d_layer"));

        const Index features_number = get_features_number();
        feature_names.resize(features_number);

        const Index outputs_number = get_outputs_number();
        output_names.resize(outputs_number);
    }
};


class SimpleResNet : public NeuralNetwork
{
public:

    void print_dim(const dimensions& dims) const
    {
        cout << "{ ";
        for (size_t i = 0; i < dims.size(); ++i)
        {
            cout << dims[i] << (i == dims.size() - 1 ? " " : ", ");
        }
        cout << "}";
    }

    SimpleResNet(const dimensions& input_dimensions,
                 const vector<Index>& blocks_per_stage, // e.g., {2, 2, 2, 2} for a ResNet-18 like structure
                 const dimensions& initial_filters,    // e.g., {64, 128, 256, 512}
                 const dimensions& output_dimensions) : NeuralNetwork()
    {
        if (input_dimensions.size() != 3)
            throw runtime_error("Input dimensions size must be 3.");
        if (blocks_per_stage.size() != initial_filters.size())
            throw runtime_error("blocks_per_stage and initial_filters must have the same size.");

        reference_all_layers();

        add_layer(make_unique<Scaling4d>(input_dimensions));

        Index last_layer_index = 0;

        auto stem_conv = make_unique<Convolutional>(get_layer(last_layer_index)->get_output_dimensions(),
                                                    dimensions{ 7, 7, input_dimensions[2], initial_filters[0] }, 
                                                    "RectifiedLinear",
                                                    dimensions{ 2, 2 }, 
                                                    "Same",
                                                    false, 
                                                    "stem_conv_1");

        add_layer(std::move(stem_conv), { last_layer_index });

        last_layer_index = get_layers_number() - 1;

        auto stem_pool = make_unique<Pooling>(get_layer(last_layer_index)->get_output_dimensions(),
                                              dimensions{ 3, 3 }, 
                                              dimensions{ 2, 2 }, 
                                              dimensions{ 1, 1 },
                                              "MaxPooling",
                                              "stem_pool");

        add_layer(std::move(stem_pool), { last_layer_index });

        last_layer_index = get_layers_number() - 1;

        for (size_t stage = 0; stage < blocks_per_stage.size(); ++stage)
        {
            for (Index block = 0; block < blocks_per_stage[stage]; ++block)
            {
                const Index block_input_index = last_layer_index;

                dimensions current_input_dims = get_layer(block_input_index)->get_output_dimensions();

                const Index filters = initial_filters[stage];

                const Index stride = (stage > 0 && block == 0) ? 2 : 1;

                // Main
                auto conv1 = make_unique<Convolutional>(current_input_dims,
                                                        dimensions{ 3, 3, current_input_dims[2], filters }, 
                                                        "RectifiedLinear",
                                                        dimensions{ stride, stride }, 
                                                        "Same",
                                                        false,
                                                        "s" + to_string(stage) + "b" + to_string(block) + "_conv1");

                add_layer(std::move(conv1), { block_input_index });

                Index main_path_index = get_layers_number() - 1;

                auto conv2 = make_unique<Convolutional>(get_layer(main_path_index)->get_output_dimensions(),
                                                        dimensions{ 3, 3, filters, filters }, 
                                                        "Linear",
                                                        dimensions{ 1, 1 }, 
                                                        "Same",
                                                        false,
                                                        "s" + to_string(stage) + "b" + to_string(block) + "_conv2");

                add_layer(std::move(conv2), { main_path_index });

                main_path_index = get_layers_number() - 1;

                // Skip Connection
                Index skip_path_index = block_input_index;

                if (stride != 1 || current_input_dims[2] != filters)
                {
                    auto skip_conv = make_unique<Convolutional>(current_input_dims,
                                                                dimensions{ 1, 1, current_input_dims[2], filters }, 
                                                                "Linear",
                                                                dimensions{ stride, stride }, 
                                                                "Same",
                                                                false,
                                                                "s" + to_string(stage) + "b" + to_string(block) + "_skip");

                    add_layer(move(skip_conv), { block_input_index });

                    skip_path_index = get_layers_number() - 1;
                }

                const dimensions main_out_dims = get_layer(main_path_index)->get_output_dimensions();

                auto addition_layer = make_unique<Addition<4>>(main_out_dims, "s" + to_string(stage) + "b" + to_string(block) + "_add");

                add_layer(std::move(addition_layer), { main_path_index, skip_path_index });

                last_layer_index = get_layers_number() - 1;

                auto activation_layer = make_unique<Convolutional>(get_layer(last_layer_index)->get_output_dimensions(),
                                                                   dimensions{ 1, 1, filters, filters }, 
                                                                   "RectifiedLinear",
                                                                   dimensions{ 1, 1 }, 
                                                                   "Same",
                                                                   false,
                                                                   "s" + to_string(stage) + "b" + to_string(block) + "_relu");

                add_layer(std::move(activation_layer), { last_layer_index });

                last_layer_index = get_layers_number() - 1;
            }
        }

        const dimensions pre_pool_dims = get_layer(last_layer_index)->get_output_dimensions();

        auto global_pool = make_unique<Pooling>(pre_pool_dims,
                                                dimensions{ pre_pool_dims[0], pre_pool_dims[1] },
                                                dimensions{ 1, 1 }, 
                                                dimensions{ 0, 0 },
                                                "AveragePooling",
                                                "global_avg_pool");

        add_layer(std::move(global_pool), { last_layer_index });

        last_layer_index = get_layers_number() - 1;

        auto flatten_layer = make_unique<Flatten<2>>(get_layer(last_layer_index)->get_output_dimensions());

        add_layer(std::move(flatten_layer), { last_layer_index });

        last_layer_index = get_layers_number() - 1;

        auto dense_layer = make_unique<Dense2d>(get_layer(last_layer_index)->get_output_dimensions(),
                                                output_dimensions,
                                                "Softmax",
                                                false,
                                                "dense_classifier");

        add_layer(std::move(dense_layer), { last_layer_index });
    }
};


class TextClassificationNetwork : public NeuralNetwork
{
public:

    TextClassificationNetwork(const dimensions& input_dimensions,
                              const dimensions& complexity_dimensions,
                              const dimensions& output_dimensions,
                              const vector<string>& new_input_vocabulary) : NeuralNetwork()
    {
        layers.clear();

        reference_all_layers();

        const Index vocabulary_size = input_dimensions[0];
        const Index sequence_length = input_dimensions[1];
        const Index embedding_dimension = input_dimensions[2];

        const Index heads_number = complexity_dimensions[0];
        //const bool use_causal_mask = false;

        const string classification_layer_activation = output_dimensions[0] == 1 ? "Logistic" : "Softmax";

        add_layer(make_unique<Embedding>(dimensions({vocabulary_size, sequence_length}),
                                         embedding_dimension,
                                         "embedding_layer"));

        add_layer(make_unique<MultiHeadAttention>(
             dimensions({sequence_length, embedding_dimension}),
             heads_number,
             "multihead_attention_layer"));

        add_layer(make_unique<Pooling3d>(
            get_output_dimensions()));

        // add_layer(make_unique<Flatten<3>>(
        //     get_output_dimensions()
        //     ));

        add_layer(make_unique<Dense2d>(
            get_output_dimensions(),
            output_dimensions,
            classification_layer_activation,
            "classification_layer"));

        input_vocabulary = new_input_vocabulary;
    }

    Tensor<type, 2> calculate_outputs(const Tensor<string, 1>& input_documents) const
    {
        Tensor<type, 2> inputs;

        return Tensor<type, 2>();

        //return calculate_outputs<2,2>(inputs);
    }

private:

    vector<string> input_vocabulary;
};

} // namespace opennn

#endif // STANDARDNETWORKS_H

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software

// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
