//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   V I T  E M B E D D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "strings_utilities.h"
#include "vit_embedding_layer.h"

#include <iostream>


namespace opennn
{

    VitEmbeddingLayer::VitEmbeddingLayer(const Index& new_image_height,
        const Index& new_image_width,
        const Index& new_image_channels,
        const Index& new_patch_size,
        const Index& new_embedding_dimension,
        const bool& new_positional_encoding,
        const string& new_name) : Layer()
    {
        set(new_image_height, new_image_width, new_image_channels, new_patch_size, new_embedding_dimension, new_positional_encoding, new_name);

        layer_type = Type::VitEmbedding;

        name = new_name;
    }


    Index VitEmbeddingLayer::get_patch_number() const
    {
        return (image_height * image_width) / (patch_size * patch_size);
    }


    Index VitEmbeddingLayer::get_input_dimension() const
    {
        return patch_size * patch_size * image_channels;
    }


    Index VitEmbeddingLayer::get_image_height() const
    {
        return image_height;
    }


    Index VitEmbeddingLayer::get_image_width() const
    {
        return image_width;
    }


    Index VitEmbeddingLayer::get_image_channels() const
    {
        return image_channels;
    }


    Index VitEmbeddingLayer::get_patch_size() const
    {
        return patch_size;
    }


    Index VitEmbeddingLayer::get_embedding_dimension() const
    {
        return embedding_weights.dimension(1);
    }


    bool VitEmbeddingLayer::get_positional_encoding() const
    {
        return positional_encoding;
    }


    dimensions VitEmbeddingLayer::get_input_dimensions() const
    {
        const Index patch_number = get_patch_number();
        const Index input_dimension = get_input_dimension();

        return { patch_number, input_dimension };
    }


    dimensions VitEmbeddingLayer::get_output_dimensions() const
    {
        const Index embedding_dimension = get_embedding_dimension();
        const Index patch_number = get_patch_number();

        return { patch_number, embedding_dimension };
    }


    Index VitEmbeddingLayer::get_parameters_number() const
    {
        return embedding_weights.size();
    }


    Tensor<type, 1> VitEmbeddingLayer::get_parameters() const
    {
        Tensor<type, 1> parameters(get_parameters_number());
        
        memcpy(parameters.data(), embedding_weights.data(), embedding_weights.size() * sizeof(type));

        return parameters;
    }


    void VitEmbeddingLayer::set(const Index& new_image_height,
        const Index& new_image_width,
        const Index& new_image_channels,
        const Index& new_patch_size,
        const Index& new_embedding_dimension,
        const bool& new_positional_encoding,
        const string& new_name)
    {
        image_height = new_image_height;

        image_width = new_image_width;

        image_channels = new_image_channels;

        patch_size = new_patch_size;

        embedding_weights.resize(get_input_dimension(), new_embedding_dimension);
        
        set_parameters_glorot();

        positional_encoding = new_positional_encoding;

        name = "vit_embedding_layer";

        layer_type = Type::VitEmbedding;
    }


    void VitEmbeddingLayer::set_image_height(const Index& new_image_height)
    {
        image_height = new_image_height;
    }


    void VitEmbeddingLayer::set_image_width(const Index& new_image_width)
    {
        image_width = new_image_width;
    }


    void VitEmbeddingLayer::set_image_channels(const Index& new_image_channels)
    {
        image_channels = new_image_channels;

        const Index input_dimension = get_input_dimension();
        const Index embedding_dimension = get_embedding_dimension();

        embedding_weights.resize(input_dimension, embedding_dimension);

        set_parameters_glorot();
    }


    void VitEmbeddingLayer::set_patch_size(const Index& new_patch_size)
    {
        patch_size = new_patch_size;

        const Index input_dimension = get_input_dimension();
        const Index embedding_dimension = get_embedding_dimension();

        embedding_weights.resize(input_dimension, embedding_dimension);

        set_parameters_glorot();
    }


    void VitEmbeddingLayer::set_embedding_size(const Index& new_embedding_dimension)
    {
        const Index input_dimension = get_input_dimension();

        embedding_weights.resize(input_dimension, new_embedding_dimension);

        set_parameters_glorot();
    }


    void VitEmbeddingLayer::set_positional_encoding(const bool& new_positional_encoding)
    {
        positional_encoding = new_positional_encoding;
    }


    void VitEmbeddingLayer::set_dropout_rate(const type& new_dropout_rate)
    {
        dropout_rate = new_dropout_rate;
    }


    void VitEmbeddingLayer::set_embedding_weights()
    {
        const Index input_dimension = get_input_dimension();
        const Index embedding_dimension = get_embedding_dimension();

        embedding_weights.resize(input_dimension, embedding_dimension);

        set_parameters_glorot();
    }


    void VitEmbeddingLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
    {
        memcpy(embedding_weights.data(), new_parameters.data() + index, embedding_weights.size() * sizeof(type));
    }


    void VitEmbeddingLayer::set_parameters_random()
    {
        if (embedding_weights.size() == 0) return;

        const type minimum = type(-0.05);
        const type maximum = type(0.05);

#pragma omp parallel for

        for (Index i = 0; i < embedding_weights.dimension(0); i++)
            for (Index j = 0; j < embedding_weights.dimension(1); j++)
                embedding_weights(i, j) = get_random_type(minimum, maximum);
    }


    void VitEmbeddingLayer::set_parameters_glorot()
    {
        if (embedding_weights.size() == 0) return;

        const type fan_in = embedding_weights.dimension(1);
        const type fan_out = embedding_weights.dimension(0);

        //Glorot (Xavier)
        const type limit = sqrt(6.0 / (fan_in + fan_out));
        const type minimum = -limit;
        const type maximum = limit;

#pragma omp parallel for
        for (Index i = 0; i < embedding_weights.dimension(0); i++) {
            for (Index j = 0; j < embedding_weights.dimension(1); j++) {
                embedding_weights(i, j) = get_random_type(minimum, maximum);
            }
        }
    }


    void VitEmbeddingLayer::set_parameters_constant(const type& value)
    {
        embedding_weights.setConstant(value);
    }


    void VitEmbeddingLayer::dropout(Tensor<type, 3>& outputs, Tensor<type, 3>& dropout_mask)
    {
        const type scaling_factor = type(1) / (type(1) - dropout_rate);

#pragma omp parallel for
        for (Index i = 0; i < outputs.size(); i++)
        {
            if (get_random_type(type(0), type(1)) < dropout_rate)
            {
                outputs(i) = 0;
                dropout_mask(i) = 0;
            }
            else
            {
                outputs(i) *= scaling_factor;
                dropout_mask(i) = scaling_factor;
            }
        }
    }


    void VitEmbeddingLayer::lookup_embedding(const Tensor<type, 3>& inputs, Tensor<type, 3>& output_with_class_token)
    {
        const Index batch_size = inputs.dimension(0);
        const Index patch_number = get_patch_number();
        const Index embedding_dimension = get_embedding_dimension();

        Tensor<type, 3> outputs;
        outputs = inputs.contract(embedding_weights, product_dims);

        output_with_class_token.resize(batch_size, patch_number + 1, embedding_dimension);
        output_with_class_token.setZero();

        output_with_class_token.slice(Eigen::array<Index, 3>{0, 1, 0}, Eigen::array<Index, 3>{batch_size, patch_number, embedding_dimension}) = outputs;

    }


    void VitEmbeddingLayer::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
        unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
        const bool& is_training)
    {
        const TensorMap<Tensor<type, 4>> images_tensor = tensor_map_4(input_pairs[0]);

        const Index embedding_dimension = get_embedding_dimension();

        const Index samples_number = images_tensor.dimension(0);

        Tensor<type, 4> input_tensor = images_tensor.reverse(Eigen::array<ptrdiff_t, 4>{0, 1, 0, 0});

        //cout << "input tensor dimensions: " << input_tensor.dimensions() << endl;
        
        Tensor<type, 5> patches = input_tensor.extract_image_patches(patch_size, patch_size, patch_size, patch_size, PaddingType::PADDING_VALID);
        Tensor<type, 5> reordered_patches = patches.shuffle(Eigen::array<ptrdiff_t, 5>{0, 3, 1, 2, 4});
        //cout << "images with patches dimensions: " << reordered_patches.dimensions() << endl;

        const Index input_dimension = get_input_dimension();
        const Index patch_number = get_patch_number();
        TensorMap<Tensor<type, 3>> inputs(reordered_patches.data(), samples_number, patch_number, input_dimension);
        inputs = inputs / type(255);
        //cout << "flatten patches: " << inputs.dimensions() << endl;

        VitEmbeddingLayerForwardPropagation* vit_embedding_layer_forward_propagation =
            static_cast<VitEmbeddingLayerForwardPropagation*>(layer_forward_propagation.get());

        Tensor<type, 3>& outputs = vit_embedding_layer_forward_propagation->outputs;
        
        lookup_embedding(inputs, outputs);
        
        
        if (positional_encoding)
        {
            outputs.device(*thread_pool_device) = outputs * sqrt(type(embedding_dimension));

            const Tensor<type, 2>& positional_encoding = vit_embedding_layer_forward_propagation->positional_encoding;

            for (Index sample_index = 0; sample_index < samples_number; sample_index++)
                outputs.chip(sample_index, 0).device(*thread_pool_device) += positional_encoding;
        }

        Tensor<type, 3>& dropout_mask = vit_embedding_layer_forward_propagation->dropout_mask;
        dropout_mask.setConstant(type(1));
        
        /*
        if (dropout_rate > 0 && is_training) {
            dropout(outputs, dropout_mask);
        }
        */
        //cout << "Vit Embedding outputs dimensions: " << outputs.dimensions() << endl;
        //cout << "Vit Embedding outputs: " << outputs << endl;
    }


    void VitEmbeddingLayer::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
        const vector<pair<type*, dimensions>>& delta_pairs,
        unique_ptr<LayerForwardPropagation>& forward_propagation,
        unique_ptr<LayerBackPropagation>& back_propagation) const
    {        
        const Index embedding_dimension = get_embedding_dimension();

        const Index batch_size = input_pairs[0].second[0];
        const Index inputs_number = input_pairs[0].second[1];
        
        const TensorMap<Tensor<type, 4>> images_tensor = tensor_map_4(input_pairs[0]);
        
        Tensor<type, 4> input_tensor = images_tensor.reverse(Eigen::array<ptrdiff_t, 4>{0, 1, 0, 0});

        //cout << "input tensor dimensions: " << input_tensor.dimensions() << endl;

        Tensor<type, 5> patches = input_tensor.extract_image_patches(patch_size, patch_size, patch_size, patch_size, PaddingType::PADDING_VALID);
        Tensor<type, 5> reordered_patches = patches.shuffle(Eigen::array<ptrdiff_t, 5>{0, 3, 1, 2, 4});
        //cout << "images with patches dimensions: " << reordered_patches.dimensions() << endl;

        const Index input_dimension = get_input_dimension();
        const Index patch_number = get_patch_number();
        TensorMap<Tensor<type, 3>> inputs(reordered_patches.data(), batch_size, patch_number, input_dimension);
        inputs = inputs / type(255);
        //cout << "flatten patches: " << inputs.dimensions() << endl;
        
        if (delta_pairs.size() > 1)
            add_deltas(delta_pairs);

        const TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);

        //cout << "Deltas embedding layer dimensions: " << deltas.dimensions() << endl;
        //cout << "Deltas embedding layer:" << endl;
        //cout << deltas << endl;
        
        // Back propagation

        VitEmbeddingLayerForwardPropagation* embedding_layer_forward_propagation =
            static_cast<VitEmbeddingLayerForwardPropagation*>(forward_propagation.get());

        Tensor<type, 3>& dropout_mask = embedding_layer_forward_propagation->dropout_mask;

        VitEmbeddingLayerBackPropagation* embedding_layer_back_propagation =
            static_cast<VitEmbeddingLayerBackPropagation*>(back_propagation.get());

        Tensor<type, 2>& embedding_weights_derivatives = embedding_layer_back_propagation->embedding_weights_derivatives;

        Tensor <type, 3> dropout_derivatives (deltas.dimension(0), deltas.dimension(1), deltas.dimension(2));
        dropout_derivatives = deltas;

        dropout_derivatives = dropout_derivatives * dropout_mask;

        Tensor<type, 3> deltas_no_cls(batch_size, patch_number, embedding_dimension);
        deltas_no_cls = dropout_derivatives.slice(Eigen::array<Index, 3>{0, 1, 0}, Eigen::array<Index, 3>{batch_size, patch_number, embedding_dimension});

        deltas_no_cls.device(*thread_pool_device) = deltas_no_cls * sqrt(type(embedding_dimension));

        embedding_weights_derivatives.device(*thread_pool_device)
            = inputs.contract(deltas_no_cls, embedding_weights_derivatives_contraction_indices);
    }


    void VitEmbeddingLayer::add_deltas(const vector<pair<type*, dimensions>>& delta_pairs) const
    {
        TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);

        for (Index i = 1; i < Index(delta_pairs.size()); i++)
            deltas.device(*thread_pool_device) += tensor_map_3(delta_pairs[i]);
    }


    void VitEmbeddingLayer::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
        const Index& index,
        Tensor<type, 1>& gradient) const
    {
        const Index embedding_weights_number = get_parameters_number();

        const VitEmbeddingLayerBackPropagation* embedding_layer_back_propagation =
            static_cast<VitEmbeddingLayerBackPropagation*>(back_propagation.get());

        const type* embedding_weights_derivatives_data = embedding_layer_back_propagation->embedding_weights_derivatives.data();

        type* gradient_data = gradient.data();

        memcpy(gradient_data + index, embedding_weights_derivatives_data, embedding_weights_number * sizeof(type));

    }

    
    void VitEmbeddingLayer::from_XML(const XMLDocument& document)
    {
        const XMLElement* vit_embedding_layer_element = document.FirstChildElement("VitEmbedding");

        if (!vit_embedding_layer_element)
            throw runtime_error("VitEmbedding element is nullptr.\n");

        set_name(read_xml_string(vit_embedding_layer_element, "Name"));
        set_embedding_size(read_xml_index(vit_embedding_layer_element, "EmbeddingSize"));
        set_positional_encoding(read_xml_bool(vit_embedding_layer_element, "PositionalEncoding"));
        set_parameters(to_type_vector(read_xml_string(vit_embedding_layer_element, "Parameters"), " "));
    }
    

    void VitEmbeddingLayer::to_XML(XMLPrinter& printer) const
    {
        printer.OpenElement("VitEmbedding");

        add_xml_element(printer, "Name", name);
        add_xml_element(printer, "EmbeddingSize", to_string(get_embedding_dimension()));
        add_xml_element(printer, "PositionalEncoding", to_string(positional_encoding ? 1 : 0));
        add_xml_element(printer, "Parameters", tensor_to_string(get_parameters()));

        printer.CloseElement();
    }
    

    VitEmbeddingLayerForwardPropagation::VitEmbeddingLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : LayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }


    pair<type*, dimensions> VitEmbeddingLayerForwardPropagation::get_outputs_pair() const
    {
        const VitEmbeddingLayer* vit_embedding_layer = static_cast<VitEmbeddingLayer*>(layer);

        const Index tokens_number = vit_embedding_layer->get_patch_number();

        const Index embedding_dimension = vit_embedding_layer->get_embedding_dimension();

        return { (type*)outputs.data(), {batch_samples_number, tokens_number + 1, embedding_dimension} };
    }


    void VitEmbeddingLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
    {
        layer = new_layer;

        const VitEmbeddingLayer* vit_embedding_layer = static_cast<VitEmbeddingLayer*>(new_layer);

        batch_samples_number = new_batch_samples_number;

        const Index tokens_number = vit_embedding_layer->get_patch_number();

        const Index embedding_dimension = vit_embedding_layer->get_embedding_dimension();

        // Outputs

        outputs.resize(batch_samples_number, tokens_number + 1, embedding_dimension);

        dropout_mask.resize(batch_samples_number, tokens_number + 1, embedding_dimension);

        if (vit_embedding_layer->get_positional_encoding())
            build_positional_encoding_matrix();
    }


    void VitEmbeddingLayerForwardPropagation::print() const
    {
        cout << "Attention scores:" << endl;
        //       cout << attention_scores.dimensions() << endl;
        cout << "Outputs dimensions:" << endl;
        //       cout << output_dimensions << endl;
        cout << "Outputs:" << endl;
        //       cout << TensorMap<Tensor<type,3>>(outputs_data, output_dimensions(0), output_dimensions(1), output_dimensions(2)) << endl;
        cout << "Attention scores:" << endl;
        //       cout << attention_scores << endl;
    }


    void VitEmbeddingLayerForwardPropagation::build_positional_encoding_matrix()
    {
        const VitEmbeddingLayer* vit_embedding_layer = static_cast<VitEmbeddingLayer*>(layer);

        const Index patch_number = vit_embedding_layer->get_patch_number();
        const Index embedding_dimension = vit_embedding_layer->get_embedding_dimension();

        positional_encoding.resize(patch_number + 1, embedding_dimension);

        positional_encoding.setZero();

#pragma omp parallel for
        for (Index i = 0; i < patch_number + 1; i++)
            for (Index j = 0; j < Index(embedding_dimension) / 2; j++) {
                positional_encoding(i, 2 * j) = sin(i / pow(10000, (2.0 * j) / embedding_dimension));
                positional_encoding(i, 2 * j + 1) = cos(i / pow(10000, (2.0 * j) / embedding_dimension));
            }

        built_positional_encoding_matrix = true;
    }


    VitEmbeddingLayerBackPropagation::VitEmbeddingLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : LayerBackPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }


    vector<pair<type*, dimensions>> VitEmbeddingLayerBackPropagation::get_input_derivative_pairs() const
    {
        return vector<pair<type*, dimensions>>();
    }


    void VitEmbeddingLayerBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
    {
        layer = new_layer;

        const VitEmbeddingLayer* embedding_layer = static_cast<VitEmbeddingLayer*>(new_layer);

        batch_samples_number = new_batch_samples_number;

        const Index input_dimension= embedding_layer->get_input_dimension();
        const Index embedding_dimension = embedding_layer->get_embedding_dimension();

        embedding_weights_derivatives.resize(input_dimension, embedding_dimension);
    }


    void VitEmbeddingLayerBackPropagation::print() const
    {
    }
    
}

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
