//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S   H E A D E R                 
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <errno.h>

// OpenNN includes

#include "config.h"
#include "data_set.h"
#include "layer.h"
#include "perceptron_layer.h"
#include "scaling_layer.h"
#include "unscaling_layer.h"
#include "bounding_layer.h"
#include "probabilistic_layer.h"
#include "convolutional_layer.h"
#include "pooling_layer.h"
#include "long_short_term_memory_layer.h"
#include "recurrent_layer.h"

namespace OpenNN
{
    struct NeuralNetworkForwardPropagation;
    struct NeuralNetworkBackPropagation;

/// This class represents the concept of neural network in the OpenNN library
///
/// This neural network is used to span a function space for the problem at hand.

class NeuralNetwork
{

public:

   enum ProjectType{Approximation, Classification, Forecasting, ImageApproximation, ImageClassification};

   // Constructors

   explicit NeuralNetwork();

   explicit NeuralNetwork(const NeuralNetwork::ProjectType&, const Tensor<Index, 1>&);
   explicit NeuralNetwork(const NeuralNetwork::ProjectType&, const initializer_list<Index>&);

   explicit NeuralNetwork(const Tensor<Index, 1>&, const Index&, const Tensor<Index, 1>&, const Index&);

   explicit NeuralNetwork(const string&);

   explicit NeuralNetwork(const tinyxml2::XMLDocument&);

   explicit NeuralNetwork(const Tensor<Layer*, 1>&);

   // Destructor

   virtual ~NeuralNetwork();

   // APPENDING LAYERS

   void delete_layers();

   void add_layer(Layer*);

   bool check_layer_type(const Layer::Type);

   // Get methods

   bool has_scaling_layer() const;
   bool has_long_short_term_memory_layer() const;
   bool has_recurrent_layer() const;
   bool has_unscaling_layer() const;
   bool has_bounding_layer() const;
   bool has_probabilistic_layer() const;
   bool has_convolutional_layer() const;
   bool is_empty() const;  

   const Tensor<string, 1>& get_inputs_names() const;
   string get_input_name(const Index&) const;
   Index get_input_index(const string&) const;

   const Tensor<string, 1>& get_outputs_names() const;
   string get_output_name(const Index&) const;
   Index get_output_index(const string&) const;

   Tensor<Layer*, 1> get_layers_pointers() const;
   Layer* get_layer_pointer(const Index&) const;
   Tensor<Layer*, 1> get_trainable_layers_pointers() const;
   Tensor<Index, 1> get_trainable_layers_indices() const;

   ScalingLayer* get_scaling_layer_pointer() const;
   UnscalingLayer* get_unscaling_layer_pointer() const;
   BoundingLayer* get_bounding_layer_pointer() const;
   ProbabilisticLayer* get_probabilistic_layer_pointer() const;
   LongShortTermMemoryLayer* get_long_short_term_memory_layer_pointer() const;
   RecurrentLayer* get_recurrent_layer_pointer() const;

   Layer* get_last_trainable_layer_pointer() const;
   PerceptronLayer* get_first_perceptron_layer_pointer() const;

   const bool& get_display() const;

   // Set methods

   void set();

   void set(const NeuralNetwork::ProjectType&, const Tensor<Index, 1>&);
   void set(const NeuralNetwork::ProjectType&, const initializer_list<Index>&);
   void set(const Tensor<Index, 1>&, const Index&, const Tensor<Index, 1>&, const Index&);

   void set(const string&);

   void set_layers_pointers(Tensor<Layer*, 1>&);

   void set_inputs_names(const Tensor<string, 1>&);
   void set_outputs_names(const Tensor<string, 1>&);

   void set_inputs_number(const Index&);
   void set_inputs_number(const Tensor<bool, 1>&);

   virtual void set_default();

   void set_threads_number(const int&);


   void set_scaling_layer(ScalingLayer&);

   void set_display(const bool&);

   // Layers 

   Index get_layers_number() const;
   Tensor<Index, 1> get_layers_neurons_numbers() const;

   Index get_trainable_layers_number() const;

   Index get_perceptron_layers_number() const;
   Index get_probabilistic_layers_number() const;
   Index get_long_short_term_memory_layers_number() const;
   Index get_recurrent_layers_number() const;

   // Architecture

   Index get_inputs_number() const;
   Index get_outputs_number() const;

   Tensor<Index, 1> get_trainable_layers_neurons_numbers() const;
   Tensor<Index, 1> get_trainable_layers_inputs_numbers() const;
   Tensor<Index, 1> get_trainable_layers_synaptic_weight_numbers() const;

   Tensor<Index, 1> get_architecture() const;

   // Parameters

   Index get_parameters_number() const;
   Tensor<type, 1> get_parameters() const;

   Tensor<Index, 1> get_trainable_layers_parameters_numbers() const;
   Tensor<Tensor<type, 1>, 1> get_trainable_layers_parameters(const Tensor<type, 1>&) const;

   void set_parameters(Tensor<type, 1>&);

   // Parameters initialization methods

   void set_parameters_constant(const type&);

   void set_parameters_random();

   // Parameters

   type calculate_parameters_norm() const;

   void perturbate_parameters(const type&);

   // Output 

   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);

   Tensor<type, 2> calculate_outputs(const Tensor<type, 4>&);

   Tensor<type, 2> calculate_directional_inputs(const Index&, const Tensor<type, 1>&, const type&, const type&, const Index& = 101) const;

   // Serialization methods

   Tensor<string, 2> get_information() const;
   Tensor<string, 2> get_perceptron_layers_information() const;
   Tensor<string, 2> get_probabilistic_layer_information() const;

   virtual void from_XML(const tinyxml2::XMLDocument&);
   void inputs_from_XML(const tinyxml2::XMLDocument&);
   void layers_from_XML(const tinyxml2::XMLDocument&);
   void outputs_from_XML(const tinyxml2::XMLDocument&);

   virtual void write_XML(tinyxml2::XMLPrinter&) const;
   // virtual void read_XML( );

   void print() const;
   void save(const string&) const;
   void save_parameters(const string&) const;

   virtual void load(const string&);
   void load_parameters_binary(const string&);

   Tensor<string, 1> get_layers_names() const;

   // Expression methods

   string write_expression() const;
   string write_expression_python() const;
   string write_expression_c() const;

   void save_expression_c(const string&);
   void save_expression_python(const string&);

   void save_outputs(const Tensor<type, 2>&, const string&);

   /// Calculate forward propagation in neural network

   void forward_propagate(const DataSetBatch&, NeuralNetworkForwardPropagation&) const;
   void forward_propagate(const DataSetBatch&, Tensor<type, 1>&, NeuralNetworkForwardPropagation&) const;

protected:

   string name = "neural_network";

   /// Names of inputs

   Tensor<string, 1> inputs_names;

   /// Names of ouputs

   Tensor<string, 1> outputs_names;

   /// Layers

   Tensor<Layer*, 1> layers_pointers;

   /// Display messages to screen.

   bool display = true;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn_cuda/neural_network_cuda.h"
#endif

};

struct NeuralNetworkForwardPropagation
{
    /// Default constructor.

    NeuralNetworkForwardPropagation() {}

    NeuralNetworkForwardPropagation(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network_pointer)
    {
        set(new_batch_samples_number, new_neural_network_pointer);
    }

    /// Destructor.

    virtual ~NeuralNetworkForwardPropagation() {}

    void set(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network_pointer)
    {
        if(new_batch_samples_number == 0) return;

        batch_samples_number = new_batch_samples_number;

        neural_network_pointer = new_neural_network_pointer;

        const Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

        const Index trainable_layers_number = trainable_layers_pointers.size();

        layers.resize(trainable_layers_number);

        for(Index i = 0; i < trainable_layers_number; i++)
        {
            switch (trainable_layers_pointers(i)->get_type())
            {
            case Layer::Perceptron:
            {
                layers(i) = new PerceptronLayerForwardPropagation(new_batch_samples_number, trainable_layers_pointers(i));
            }
            break;

            case Layer::Probabilistic:
            {
                layers(i) = new ProbabilisticLayerForwardPropagation(new_batch_samples_number, trainable_layers_pointers(i));
            }
            break;

            case Layer::Recurrent:
            {
                layers(i) = new RecurrentLayerForwardPropagation(new_batch_samples_number, trainable_layers_pointers(i));
            }
            break;

            case Layer::LongShortTermMemory:
            {
                layers(i) = new LongShortTermMemoryLayerForwardPropagation(new_batch_samples_number, trainable_layers_pointers(i));
            }
            break;

            case Layer::Convolutional:
            {
                layers(i) = new ConvolutionalLayerForwardPropagation(new_batch_samples_number, trainable_layers_pointers(i));
            }
            break;

            default: break;
            }
        }
    }

    void print() const
    {
        const Index layers_number = layers.size();

        cout << "Layers number: " << layers_number << endl;

        for(Index i = 0; i < layers_number; i++)
        {
            cout << "Layer " << i + 1 << endl;

            layers(i)->print();
        }
    }

    Index batch_samples_number = 0;

    NeuralNetwork* neural_network_pointer = nullptr;

    Tensor<LayerForwardPropagation*, 1> layers;
};


struct NeuralNetworkBackPropagation
{
    NeuralNetworkBackPropagation() {}

    NeuralNetworkBackPropagation(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network_pointer)
    {
        batch_samples_number = new_batch_samples_number;

        neural_network_pointer = new_neural_network_pointer;
    }

    void set(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network_pointer)
    {
        batch_samples_number = new_batch_samples_number;

        neural_network_pointer = new_neural_network_pointer;

        const Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

        const Index trainable_layers_number = trainable_layers_pointers.size();

        layers.resize(trainable_layers_number);

        for(Index i = 0; i < trainable_layers_number; i++)
        {
            switch (trainable_layers_pointers(i)->get_type())
            {
            case Layer::Perceptron:
            {
                layers(i) = new PerceptronLayerBackPropagation(new_batch_samples_number, trainable_layers_pointers(i));
            }
            break;

            case Layer::Probabilistic:
            {
                layers(i) = new ProbabilisticLayerBackPropagation(new_batch_samples_number, trainable_layers_pointers(i));
            }
            break;

            case Layer::Recurrent:
            {
                layers(i) = new RecurrentLayerBackPropagation(new_batch_samples_number, trainable_layers_pointers(i));
            }
            break;

            case Layer::LongShortTermMemory:
            {
                layers(i) = new LongShortTermMemoryLayerBackPropagation(new_batch_samples_number, trainable_layers_pointers(i));
            }
            break;

            case Layer::Convolutional:
            {
                layers(i) = new ConvolutionalLayerBackPropagation(new_batch_samples_number, trainable_layers_pointers(i));
            }
            break;

            default: break;
            }
        }
    }

    void print() const
    {
        cout << "Neural network back-propagation" << endl;

        const Index layers_number = layers.size();

        cout << "Layers number: " << layers_number << endl;

        for(Index i = 0; i < layers_number; i++)
        {
            cout << "Layer " << i + 1 << endl;

            layers(i)->print();
        }
    }

    Index batch_samples_number = 0;

    NeuralNetwork* neural_network_pointer = nullptr;

    Tensor<LayerBackPropagation*, 1> layers;
};


struct NeuralNetworkBackPropagationLM
{
    NeuralNetworkBackPropagationLM() {}

    NeuralNetworkBackPropagationLM(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network_pointer)
    {
        batch_samples_number = new_batch_samples_number;

        neural_network_pointer = new_neural_network_pointer;
    }

    void set(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network_pointer)
    {
        batch_samples_number = new_batch_samples_number;

        neural_network_pointer = new_neural_network_pointer;

        const Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

        const Index trainable_layers_number = trainable_layers_pointers.size();

        layers.resize(trainable_layers_number);

        for(Index i = 0; i < trainable_layers_number; i++)
        {
            switch (trainable_layers_pointers(i)->get_type())
            {
            case Layer::Perceptron:

            layers(i) = new PerceptronLayerBackPropagationLM(new_batch_samples_number, trainable_layers_pointers(i));

            break;

            case Layer::Probabilistic:

            layers(i) = new ProbabilisticLayerBackPropagationLM(new_batch_samples_number, trainable_layers_pointers(i));

            break;

            default:
            {
                ostringstream buffer;

                buffer << "OpenNN Exception: NeuralNetwork class.\n"
                       << "Levenberg-Marquardt can only be used with Perceptron and Probabilistic layers.\n";

                throw logic_error(buffer.str());
            }
            }
        }
    }

    void print()
    {
        const Index layers_number = layers.size();

        cout << "Layers number: " << layers_number << endl;

        for(Index i = 0; i < layers_number; i++)
        {
            cout << "Layer " << i + 1 << endl;

            layers(i)->print();
        }
    }

    Index batch_samples_number = 0;

    NeuralNetwork* neural_network_pointer = nullptr;

    Tensor<LayerBackPropagationLM*, 1> layers;
};

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2021 Artificial Intelligence Techniques, SL.
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
