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
#include "principal_components_layer.h"
#include "unscaling_layer.h"
#include "bounding_layer.h"
#include "probabilistic_layer.h"
#include "convolutional_layer.h"
#include "pooling_layer.h"
#include "long_short_term_memory_layer.h"
#include "recurrent_layer.h"
#include "tinyxml2.h"

#ifdef __OPENNN_CUDA__
    #include "../../artelnics/opennn_cuda/opennn_cuda/kernels.h"
#endif


namespace OpenNN
{

/// This class represents the concept of neural network in the OpenNN library.

///
/// This neural network is used to span a function space for the problem at hand.

class NeuralNetwork
{

public:

    enum ProjectType{Approximation, Classification, Forecasting, ImageApproximation, ImageClassification};

   // Constructors

   explicit NeuralNetwork();

   explicit NeuralNetwork(const NeuralNetwork::ProjectType&, const Tensor<int, 1>&);

   explicit NeuralNetwork(const Tensor<int, 1>&, const int&, const Tensor<int, 1>&, const int&);

   explicit NeuralNetwork(const string&);

   explicit NeuralNetwork(const tinyxml2::XMLDocument&);

   explicit NeuralNetwork(const vector<Layer*>&);

   NeuralNetwork(const NeuralNetwork&);

   // Destructor

   virtual ~NeuralNetwork();

   struct ForwardPropagation
   {
       /// Default constructor.

       ForwardPropagation() {}

       ForwardPropagation(const Index& new_batch_instances_number, NeuralNetwork* new_neural_network_pointer)
       {
           batch_instances_number = new_batch_instances_number;
           neural_network_pointer = new_neural_network_pointer;

           allocate();
       }

       /// Destructor.

       virtual ~ForwardPropagation() {}

       void allocate()
       {
           const int trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

           const vector<Layer*> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

           layers.resize(trainable_layers_number);

           for(int i = 0; i < trainable_layers_number; i++)
           {
               if(trainable_layers_pointers[i]->get_type() == Layer::Convolutional)
               {
                   const ConvolutionalLayer* convolutional_layer = dynamic_cast<ConvolutionalLayer*>(trainable_layers_pointers[i]);

                   const int outputs_channels_number = convolutional_layer->get_filters_number();
                   const int outputs_rows_number = convolutional_layer->get_outputs_rows_number();
                   const int outputs_columns_number = convolutional_layer->get_outputs_columns_number();
/*
                   layers[i].combinations.resize(Tensor<int, 1>({batch_instances_number, outputs_channels_number, outputs_rows_number, outputs_columns_number}));
                   layers[i].activations.resize(Tensor<int, 1>({batch_instances_number, outputs_channels_number, outputs_rows_number, outputs_columns_number}));
                   layers[i].activations_derivatives.resize(Tensor<int, 1>({batch_instances_number, outputs_channels_number, outputs_rows_number, outputs_columns_number}));
*/
               }
               else if(trainable_layers_pointers[i]->get_type() == Layer::Pooling)
               {
                   const PoolingLayer* pooling_layer = dynamic_cast<PoolingLayer*>(trainable_layers_pointers[i]);

                   const int outputs_channels_number = pooling_layer->get_inputs_channels_number();
                   const int outputs_rows_number = pooling_layer->get_outputs_rows_number();
                   const int outputs_columns_number = pooling_layer->get_outputs_columns_number();
/*
                   layers[i].combinations.resize(Tensor<int, 1>({batch_instances_number, outputs_channels_number, outputs_rows_number, outputs_columns_number}));
                   layers[i].activations.resize(Tensor<int, 1>({batch_instances_number, outputs_channels_number, outputs_rows_number, outputs_columns_number}));
                   layers[i].activations_derivatives.resize(Tensor<int, 1>({batch_instances_number, outputs_channels_number, outputs_rows_number, outputs_columns_number}));
*/
               }
               else if(trainable_layers_pointers[i]->get_type() == Layer::Recurrent)
               {
                   const RecurrentLayer* recurrent_layer = dynamic_cast<RecurrentLayer*>(trainable_layers_pointers[i]);

                   const int neurons_number = recurrent_layer->get_neurons_number();

                   layers[i].combinations = Tensor<type, 2>(batch_instances_number, neurons_number);
                   layers[i].activations = Tensor<type, 2>(batch_instances_number, neurons_number);
                   layers[i].activations_derivatives = Tensor<type, 2>(batch_instances_number, neurons_number);

               }
               else if(trainable_layers_pointers[i]->get_type() == Layer::LongShortTermMemory)
               {
                   const LongShortTermMemoryLayer* long_short_term_memory_layer = dynamic_cast<LongShortTermMemoryLayer*>(trainable_layers_pointers[i]);

                   const int neurons_number = long_short_term_memory_layer->get_neurons_number();

                   layers[i].combinations = Tensor<type, 2>(batch_instances_number, neurons_number);
                   layers[i].activations = Tensor<type, 2>(batch_instances_number, neurons_number);
                   layers[i].activations_derivatives = Tensor<type, 2>(batch_instances_number, neurons_number);
               }
               else if(trainable_layers_pointers[i]->get_type() == Layer::Perceptron)
               {
                   const PerceptronLayer* perceptron_layer = dynamic_cast<PerceptronLayer*>(trainable_layers_pointers[i]);

                   const int neurons_number = perceptron_layer->get_neurons_number();

                   layers[i].combinations = Tensor<type, 2>(batch_instances_number, neurons_number);
                   layers[i].activations = Tensor<type, 2>(batch_instances_number, neurons_number);
                   layers[i].activations_derivatives = Tensor<type, 2>(batch_instances_number, neurons_number);

                   layers[i].combinations.setRandom();
                   layers[i].activations.setRandom();
                   layers[i].activations_derivatives.setRandom();
               }
               else if(trainable_layers_pointers[i]->get_type() == Layer::Probabilistic)
               {
                   const ProbabilisticLayer* probabilistic_layer = dynamic_cast<ProbabilisticLayer*>(trainable_layers_pointers[i]);

                   const int neurons_number = probabilistic_layer->get_neurons_number();

                   layers[i].combinations = Tensor<type, 2>(batch_instances_number, neurons_number);
                   layers[i].activations = Tensor<type, 2>(batch_instances_number, neurons_number);
                   /*
                   layers[i].activations_derivatives_3d = Tensor<type, 3>(batch_instances_number, neurons_number, neurons_number);
                   */
               }
               else
               {
                   /// @todo throw exception.
               }
           }
       }

       void print()
       {
           const int layers_number = layers.size();

           cout << "Layers number: " << layers_number << endl;

           for(int i = 0; i < layers_number; i++)
           {
               cout << "Layer " << i+1 << endl;

               layers[i].print();
           }
       }

       int batch_instances_number = 0;
       NeuralNetwork* neural_network_pointer = nullptr;

       vector<Layer::ForwardPropagation> layers;
   };


   // APPENDING LAYERS

   void add_layer(Layer*);

   bool check_layer_type(const Layer::Type);

   // Get methods

   bool has_scaling_layer() const;
   bool has_principal_components_layer() const;
   bool has_long_short_term_memory_layer() const;
   bool has_recurrent_layer() const;
   bool has_unscaling_layer() const;
   bool has_bounding_layer() const;
   bool has_probabilistic_layer() const;

   bool is_empty() const;  

   Tensor<string, 1> get_inputs_names() const;
   string get_input_name(const int&) const;
   int get_input_index(const string&) const;

   Tensor<string, 1> get_outputs_names() const;
   string get_output_name(const int&) const;
   int get_output_index(const string&) const;

   vector<Layer*> get_layers_pointers() const;
   vector<Layer*> get_trainable_layers_pointers() const;
   Tensor<int, 1> get_trainable_layers_indices() const;

   ScalingLayer* get_scaling_layer_pointer() const;
   UnscalingLayer* get_unscaling_layer_pointer() const;
   BoundingLayer* get_bounding_layer_pointer() const;
   ProbabilisticLayer* get_probabilistic_layer_pointer() const;
   PrincipalComponentsLayer* get_principal_components_layer_pointer() const;
   LongShortTermMemoryLayer* get_long_short_term_memory_layer_pointer() const;
   RecurrentLayer* get_recurrent_layer_pointer() const;

   Layer* get_output_layer_pointer() const;
   Layer* get_layer_pointer(const int&) const;
   PerceptronLayer* get_first_perceptron_layer_pointer() const;

   const bool& get_display() const;

   // Set methods

   void set();

   void set(const NeuralNetwork::ProjectType&, const Tensor<int, 1>&);
   void set(const Tensor<int, 1>&, const int&, const Tensor<int, 1>&, const int&);

   void set(const string&);
   void set(const NeuralNetwork&);

   void set_inputs_names(const Tensor<string, 1>&);
   void set_outputs_names(const Tensor<string, 1>&);

   void set_inputs_number(const int&);
   void set_inputs_number(const Tensor<bool, 1>&);

   virtual void set_default();

   void set_layers_pointers(vector<Layer*>&);

   void set_scaling_layer(ScalingLayer&);

   void set_display(const bool&);

   // Layers 

   int get_layers_number() const;
   Tensor<int, 1> get_layers_neurons_numbers() const;

   int get_trainable_layers_number() const;

   // Architecture

   int get_inputs_number() const;
   int get_outputs_number() const;

   Tensor<int, 1> get_architecture() const;

   // Parameters

   int get_parameters_number() const;
   int get_trainable_parameters_number() const;
   Tensor<type, 1> get_parameters() const;

   Tensor<int, 1> get_trainable_layers_parameters_numbers() const;

   vector<Tensor<type, 1>> get_trainable_layers_parameters(const Tensor<type, 1>&) const;

   void set_parameters(const Tensor<type, 1>&);

   // Parameters initialization methods

   void initialize_parameters(const double&);

   void randomize_parameters_uniform(const double& = -1.0, const double& = 1.0);  

   void randomize_parameters_normal(const double& = 0.0, const double& = 1.0);  

   // Parameters

   double calculate_parameters_norm() const;
   Descriptives calculate_parameters_descriptives() const;
   Histogram calculate_parameters_histogram(const int& = 10) const;

   void perturbate_parameters(const double&);

   // Output 

   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);

   Tensor<type, 2> calculate_trainable_outputs(const Tensor<type, 2>&) const;

   Tensor<type, 2> calculate_trainable_outputs(const Tensor<type, 2>&, const Tensor<type, 1>&) const;

   Tensor<type, 2> calculate_directional_inputs(const int&, const Tensor<type, 1>&, const double&, const double&, const int& = 101) const;

   vector<Histogram> calculate_outputs_histograms(const int& = 1000, const int& = 10);
   vector<Histogram> calculate_outputs_histograms(const Tensor<type, 2>&, const int& = 10);

   Tensor<type, 1> calculate_outputs_std(const Tensor<type, 1>&);

   // Serialization methods

   string object_to_string() const;
 
   Tensor<string, 2> get_information() const;

   virtual tinyxml2::XMLDocument* to_XML() const;
   virtual void from_XML(const tinyxml2::XMLDocument&);

   virtual void write_XML(tinyxml2::XMLPrinter&) const;
   // virtual void read_XML( );

   void print() const;
   void print_summary() const;
   void save(const string&) const;
   void save_parameters(const string&) const;

   virtual void load(const string&);
   void load_parameters(const string&);

   void save_data(const string&) const;

   // Expression methods

   string write_expression() const;
   string write_mathematical_expression_php() const;
   string write_expression_python() const;
   string write_expression_php() const;
   string write_expression_R() const;

   void save_expression(const string&);
   void save_expression_python(const string&);
   void save_expression_R(const string&);

   /// Calculate de forward propagation in the neural network

   vector<Layer::ForwardPropagation> calculate_forward_propagation(const Tensor<type, 2>&) const;

   void calculate_forward_propagation(const ThreadPoolDevice& thread_pool_device,
                                      const DataSet::Batch& batch,
                                      ForwardPropagation& forward_propagation) const
   {
       const int trainable_layers_number = get_trainable_layers_number();

       const vector<Layer*> trainable_layers_pointers = get_trainable_layers_pointers();

       // First layer

       trainable_layers_pointers[0]->calculate_forward_propagation(thread_pool_device, batch.inputs_2d, forward_propagation.layers[0]);

       // Rest of layers

       for(int i = 1; i < trainable_layers_number; i++)
       {
            trainable_layers_pointers[i]->calculate_forward_propagation(thread_pool_device,
                                                                        forward_propagation.layers[i-1].activations,
                                                                        forward_propagation.layers[i]);
       }

   }

protected:

   /// Names of inputs

   Tensor<string, 1> inputs_names;

   /// Names of ouputs

   Tensor<string, 1> outputs_names;

   /// Layers

   vector<Layer*> layers_pointers;

   /// Display messages to screen.

   bool display = true;

#ifdef __OPENNN_CUDA__
    #include "../../artelnics/opennn_cuda/opennn_cuda/neural_network_cuda.h"
#endif

};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
