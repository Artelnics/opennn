//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   V I S I O N  T R A N S F O R M E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "vision_transformer.h"
#include "tensors.h"
#include "vit_embedding_layer.h"
#include "normalization_layer_3d.h"
#include "vit_multihead_attention_layer.h"
#include "addition_layer_3d.h"
#include "vit_feed_forward_network_layer_3d.h"
#include "probabilistic_layer.h"
#include "forward_propagation.h"
#include "images.h"


namespace opennn
{

    VisionTransformer::VisionTransformer(const Index& image_height,
        const Index& image_width,
        const Index& image_channels,
        const Index& patch_size,
        const Index& target_number,
        const Index& embedding_dimension,
        const Index& hidden_dimension,
        const Index& heads_number,
        const Index& layers_number)
    {
        set(image_height,
            image_width,
            image_channels,
            patch_size,
            target_number,
            embedding_dimension,
            hidden_dimension,
            heads_number,
            layers_number);
    
    }


    void VisionTransformer::set(const Index& new_image_height,
        const Index& new_image_width,
        const Index& new_image_channels,
        const Index& new_patch_size,
        const Index& new_target_number,
        const Index& new_embedding_dimension,
        const Index& new_hidden_dimension,
        const Index& new_heads_number,
        const Index& new_blocks_number)
    {
        name = "vision_transformer";
        const Index patch_number = (new_patch_size == 0)
            ? 0
            : (new_image_height * new_image_width) / (new_patch_size * new_patch_size);

        const Index input_dimension = new_patch_size * new_patch_size * new_image_channels;

        layers.clear();

        input_names.resize(patch_number * input_dimension);

        // Embedding Layer

        add_layer(make_unique<VitEmbeddingLayer>(new_image_height,
            new_image_width,
            new_image_channels,
            new_patch_size,
            new_embedding_dimension,
            true,
            "vit_embedding"));

        set_layer_inputs_indices("vit_embedding", "input");
        //input_embedding_layer->set_dropout_rate(dropout_rate);


        // Encoder

        for (Index i = 0; i < new_blocks_number; i++)
        {
            // Normalization

            add_layer(make_unique<NormalizationLayer3D>(patch_number + 1, // patches + cls token
                new_embedding_dimension,
                "normalization_" + to_string(i + 1)));

            //set_layer_inputs_indices("normalization_" + to_string(i + 1), "vit_embedding");
            i == 0
                ? set_layer_inputs_indices("normalization_" + to_string(i + 1), "vit_embedding")
                : set_layer_inputs_indices("normalization_" + to_string(i + 1), "feed_forward_network_addition_" + to_string(i));

            //Multihead Attention

            add_layer(make_unique<VitMultiheadAttentionLayer>(patch_number + 1,
                new_embedding_dimension,
                new_heads_number,
                "self_attention_" + to_string(i + 1)));

            set_layer_inputs_indices("self_attention_" + to_string(i + 1), "normalization_" + to_string(i + 1));

            // Addition

            add_layer(make_unique<AdditionLayer3D>(patch_number + 1,
                new_embedding_dimension,
                "self_attention_addition_" + to_string(i + 1)));

            set_layer_inputs_indices("self_attention_addition_" + to_string(i + 1), { "vit_embedding", "self_attention_" + to_string(i + 1) });

            // Normalization

            add_layer(make_unique<NormalizationLayer3D>(patch_number + 1,
                new_embedding_dimension,
                "self_attention_normalization_" + to_string(i + 1)));

            set_layer_inputs_indices("self_attention_normalization_" + to_string(i + 1), "self_attention_addition_" + to_string(i + 1));

            // Feed Forward Network

            add_layer(make_unique<VitFeedForwardNetworkLayer3D>(patch_number + 1,
                new_embedding_dimension,
                new_hidden_dimension,
                VitFeedForwardNetworkLayer3D::ActivationFunction::RectifiedLinear,
                "feed_forward_network_" + to_string(i + 1)));

            set_layer_inputs_indices("feed_forward_network_" + to_string(i + 1), "self_attention_normalization_" + to_string(i + 1));

            // Addition

            add_layer(make_unique<AdditionLayer3D>(patch_number + 1,
                new_embedding_dimension,
                "feed_forward_network_addition_" + to_string(i + 1)));

            set_layer_inputs_indices("feed_forward_network_addition_" + to_string(i + 1), { "self_attention_addition_" + to_string(i + 1), "feed_forward_network_" + to_string(i + 1) });

        }

        const vector <Index> new_input_dimensions = { new_embedding_dimension };
        const vector <Index> new_output_dimensions = { new_target_number };

        add_layer(make_unique<ProbabilisticLayer>(new_input_dimensions,
            new_output_dimensions,
            "probabilistic_layer"));

        set_layer_inputs_indices("probabilistic_layer", "feed_forward_network_addition_" + to_string(new_blocks_number));
    }


    Index VisionTransformer::calculate_image_output(const filesystem::path& image_path)
    {
        Tensor<type, 3> image = read_bmp_image(image_path);

        VitEmbeddingLayer* vit_embedding_layer = static_cast<VitEmbeddingLayer*>(get_first(Layer::Type::VitEmbedding));

        const Index height = vit_embedding_layer->get_image_height();
        const Index width = vit_embedding_layer->get_image_width();
        const Index channels = vit_embedding_layer->get_image_channels();

        const Index current_height = image.dimension(0);
        const Index current_width = image.dimension(1);
        const Index current_channels = image.dimension(2);

        if (current_channels != channels)
            throw runtime_error("Error: Different channels number " + image_path.string() + "\n");

        if (current_height != height || current_width != width)
            image = resize_image(image, height, width);

        Tensor<type, 4> input_data(1, height, width, channels);

        const Index pixels_number = height * width * channels;

#pragma omp parallel for
        for (Index j = 0; j < pixels_number; j++)
            input_data(j) = image(j);
        
        const Tensor<type, 2> outputs = calculate_outputs(input_data);
        
        Index predicted_index = 0;
        
        if (outputs.size() > 1)
        {
            type max_value = outputs(0);
            
            for (Index i = 1; i < outputs.dimension(1); ++i)
            {
                if (outputs(i) > max_value)
                {
                    max_value = outputs(i);
                    predicted_index = i;
                }
            }
        }
        else
            predicted_index = outputs(0);

        return predicted_index;
    }


    void VisionTransformer::save(const filesystem::path& file_name) const
    {
        ofstream file(file_name);

        if (!file.is_open())
            return;

        XMLPrinter printer;
        to_XML(printer);
        file << printer.CStr();
    }


    void VisionTransformer::load(const filesystem::path& file_name)
    {
        set_default();

        XMLDocument document;

        if (document.LoadFile(file_name.string().c_str()))
            throw runtime_error("Cannot load XML file " + file_name.string() + ".\n");

        from_XML(document);
    }


    void VisionTransformer::set_dropout_rate(const type& new_dropout_rate)
    {
        dropout_rate = new_dropout_rate;
    }
};

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
