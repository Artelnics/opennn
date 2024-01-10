//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef EMBEDDING_LAYER_H
#define EMBEDDING_LAYER_H

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "config.h"
#include "layer.h"
#include "perceptron_layer.h"

#ifdef OPENNN_MKL
#include "../mkl/mkl.h"
#endif

namespace opennn
{

struct EmbeddingLayerForwardPropagation;
struct EmbeddingLayerBackPropagation;
struct EmbeddingLayerBackPropagationLM;

#ifdef OPENNN_CUDA
#include "../../opennn-cuda/opennn-cuda/struct_perceptron_layer_cuda.h"
#endif


/// This class represents an Embedding layer.

/// EmbeddingLayer has inputs of a fixed length (input_length) and within a fixed set of possible integer values (input_dim).
/// The layer will assign to each possible value a dense vector of fixed length (depth).
/// The vectors are learnable parameters, which is implemented by passing a one-hot encoding of each value through a PerceptronLayer.


class EmbeddingLayer : public Layer

{

public:

    // Constructors

    explicit EmbeddingLayer();

    explicit EmbeddingLayer(const Index&, /// Input dim
                            const Index&, /// Input length
                            const Index&, /// Embedding depth
                            const bool& = false); /// Add positional encoding or not

    // Get methods

    bool is_empty() const;

    Index get_input_dim() const;
    Index get_input_length() const;
    Index get_depth() const;
    Tensor<type, 2> get_lookup_table() const;

    Index get_parameters_number() const final;

    // Display messages

    const bool& get_display() const;

    // Set methods

    void set();
    void set(const Index&, const Index&, const Index&, const bool& = false);

    void set_default();
    void set_name(const string&);

    // Architecture

    void set_input_dim(const Index&);
    void set_input_length(const Index&);
    void set_depth(const Index&);

    void set_lookup_table();
    void set_parameters_random() final;

    // Display messages

    void set_display(const bool&);

    // Embedding lookup

//    Tensor<type, 2> one_hot_encode_row(const Tensor<type, 1>&);

    void lookup_embedding(const Tensor<type, 2>&, EmbeddingLayerForwardPropagation*);

    // Positional encoding

    const Tensor<type, 2> build_positional_encoding_matrix();

    // Embedding layer outputs

    void forward_propagate(const pair<type*, dimensions>&, LayerForwardPropagation*, const bool&) final;

    // Expression methods

    string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const final;

    // Serialization methods

    /// @todo

    void from_XML(const tinyxml2::XMLDocument&) final;
    void write_XML(tinyxml2::XMLPrinter&) const final;

protected:

    // MEMBERS

    /// Input dimension (i.e. number of values input can take or vocabulary size)

    Index input_dim;

    /// Length of each input entry (assuming equal length)

    Index input_length;

    /// Embedding depth

    Index depth;

    /// Lookup table

    Tensor<type, 2> lookup_table;

    /// Whether the layer has to add positional encoding or not

    bool positional_encoding;

    /// Display messages to screen.

    bool display = true;

#ifdef OPENNN_CUDA
#include "../../opennn-cuda/opennn-cuda/perceptron_layer_cuda.h"
#else
    };
#endif

    struct EmbeddingLayerForwardPropagation : LayerForwardPropagation
    {
        // Default constructor

        explicit EmbeddingLayerForwardPropagation() : LayerForwardPropagation()
        {
        }

        virtual ~EmbeddingLayerForwardPropagation()
        {
        }


        explicit EmbeddingLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
            : LayerForwardPropagation()
        {
            set(new_batch_samples_number, new_layer_pointer);
        }

        void set(const Index& new_batch_samples_number, Layer* new_layer_pointer) final
        {
            EmbeddingLayer* layer_pointer = static_cast<EmbeddingLayer*>(new_layer_pointer);

            batch_samples_number = new_batch_samples_number;

            const Index input_length = layer_pointer->get_input_length();

            // Outputs

//            outputs.resize(1);
//            Tensor<Index, 1> output_dimensions(3);
//            output_dimensions.setValues({batch_samples_number, input_length, depth});
//            outputs(0).set_dimensions(output_dimensions);

            // Rest of quantities

        }

        void print() const
        {
            //            cout << "Attention scores:" << endl;
            //            cout << attention_scores.dimensions() << endl;


            //            cout << "Outputs dimensions:" << endl;
            //            cout << outputs_dimensions << endl;

            //            cout << "Outputs:" << endl;
            //            cout << TensorMap<Tensor<type,3>>(outputs_data, outputs_dimensions(0), outputs_dimensions(1), outputs_dimensions(2)) << endl;

            //            cout << "Attention scores:" << endl;
            //            cout << attention_scores << endl;
        }

        Tensor<type, 3> outputs;

    };


    struct EmbeddingLayerBackPropagationLM : LayerBackPropagationLM
    {
        // Default constructor

        explicit EmbeddingLayerBackPropagationLM() : LayerBackPropagationLM()
        {

        }


        explicit EmbeddingLayerBackPropagationLM(const Index& new_batch_samples_number, Layer* new_layer_pointer)
            : LayerBackPropagationLM()
        {
            set(new_batch_samples_number, new_layer_pointer);
        }


        virtual ~EmbeddingLayerBackPropagationLM()
        {

        }

        /// @todo
        /*
        void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        {
            layer_pointer = new_layer_pointer;

            batch_samples_number = new_batch_samples_number;

            const Index neurons_number = layer_pointer->get_neurons_number();
            const Index parameters_number = layer_pointer->get_parameters_number();

            deltas.resize(batch_samples_number, neurons_number);

            squared_errors_Jacobian.resize(batch_samples_number, parameters_number);
        }

        void print() const
        {
            cout << "Deltas:" << endl;
            cout << deltas << endl;

            cout << "Squared errors Jacobian: " << endl;
            cout << squared_errors_Jacobian << endl;
>>>>>>> f437e115fe9e567c3475cda88f60e74912a668c2

        PerceptronLayerForwardPropagation perceptron_forward_propagation;
    };


    struct EmbeddingLayerBackPropagation : LayerBackPropagation
    {
        // Default constructor

        explicit EmbeddingLayerBackPropagation() : LayerBackPropagation()
        {

        }

        virtual ~EmbeddingLayerBackPropagation()
        {
        }


        explicit EmbeddingLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
            : LayerBackPropagation()
        {
            set(new_batch_samples_number, new_layer_pointer);
        }

        /// @todo

        void set(const Index& new_batch_samples_number, Layer* new_layer_pointer) final
        {
            layer_pointer = new_layer_pointer;

            batch_samples_number = new_batch_samples_number;

            const Index neurons_number = layer_pointer->get_neurons_number();
            const Index inputs_number = layer_pointer->get_inputs_number();

            deltas.resize(batch_samples_number, neurons_number);

            biases_derivatives.resize(neurons_number);

            synaptic_weights_derivatives.resize(inputs_number, neurons_number);

            deltas_times_activations_derivatives.resize(batch_samples_number, neurons_number);
        }


        void print() const
        {
            cout << "Deltas:" << endl;
            cout << deltas << endl;

            cout << "Biases derivatives:" << endl;
            cout << biases_derivatives << endl;

            cout << "Synaptic weights derivatives:" << endl;
            cout << synaptic_weights_derivatives << endl;
        }

        Tensor<type, 2> deltas;

        Tensor<type, 1> biases_derivatives;
        Tensor<type, 2> synaptic_weights_derivatives;

        Tensor<type, 2> deltas_times_activations_derivatives;
    };

}

#endif // EMBEDDING_LAYER_H


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2023 Artificial Intelligence Techniques, SL.
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
