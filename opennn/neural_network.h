//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <string>
#include <memory>

#include "config.h"
#include "layer.h"
#include "perceptron_layer.h"
//#include "perceptron_layer_3d.h"
//#include "addition_layer_3d.h"
//#include "normalization_layer_3d.h"
#include "scaling_layer_2d.h"
#include "scaling_layer_4d.h"
#include "unscaling_layer.h"
#include "bounding_layer.h"
#include "probabilistic_layer.h"
//#include "probabilistic_layer_3d.h"
//#include "flatten_layer.h"
//#include "pooling_layer.h"
//#include "convolutional_layer.h"
#include "long_short_term_memory_layer.h"
//#include "multihead_attention_layer.h"
#include "embedding_layer.h"
#include "recurrent_layer.h"

namespace opennn
{

struct ForwardPropagation;
struct NeuralNetworkBackPropagation;
struct NeuralNetworkBackPropagationLM;

#ifdef OPENNN_CUDA
struct ForwardPropagationCuda;
struct NeuralNetworkBackPropagationCuda;
#endif

class NeuralNetwork
{

public:

   enum class ModelType{AutoAssociation,
                        Approximation,
                        Classification,
                        Forecasting,
                        ImageClassification,
                        TextClassification};

   // Constructors

   explicit NeuralNetwork();

   explicit NeuralNetwork(const NeuralNetwork::ModelType&, const dimensions&, const dimensions&, const dimensions&);

   explicit NeuralNetwork(const string&);

   // APPENDING LAYERS


   void add_layer(unique_ptr<Layer>, 
                  const vector<Index>& = vector<Index>());

   bool validate_layer_type(const Layer::Type&) const;

   // Get

   bool has(const Layer::Type&) const;

   bool is_empty() const;

   const vector<string>& get_input_names() const;
   string get_input_name(const Index&) const;
   Index get_input_index(const string&) const;

   ModelType get_model_type() const;
   string get_model_type_string() const;

   const vector<string>& get_output_names() const;
   string get_output_name(const Index&) const;
   Index get_output_index(const string&) const;

   const vector<unique_ptr<Layer>>& get_layers() const;
   const unique_ptr<Layer>& get_layer(const Index&) const;
   const unique_ptr<Layer>& get_layer(const string&) const;

   Index get_layer_index(const string&) const;

   const vector<vector<Index>>& get_layer_input_indices() const;
   vector<vector<Index>> get_layer_output_indices() const;

   Index find_input_index(const vector<Index>&, const Index&) const;

   ScalingLayer2D* get_scaling_layer_2d() const;
   ScalingLayer4D* get_scaling_layer_4d() const;
   UnscalingLayer* get_unscaling_layer() const;
   BoundingLayer* get_bounding_layer() const;
   //FlattenLayer* get_flatten_layer() const;
   //ConvolutionalLayer* get_convolutional_layer() const;
   //PoolingLayer* get_pooling_layer() const;
   ProbabilisticLayer* get_probabilistic_layer() const;
   LongShortTermMemoryLayer* get_long_short_term_memory_layer() const;
   RecurrentLayer* get_recurrent_layer() const;

   PerceptronLayer* get_first_perceptron_layer() const;

   const bool& get_display() const;

   // Set

   void set();

   void set(const NeuralNetwork::ModelType&, 
            const dimensions& = {}, 
            const dimensions& = {},
            const dimensions& = {});

   void set_approximation(const dimensions&, const dimensions&, const dimensions&);
   void set_classification(const dimensions&, const dimensions&, const dimensions&);
   void set_forecasting(const dimensions&, const dimensions&, const dimensions&);
   void set_auto_association(const dimensions&, const dimensions&, const dimensions&);
   void set_image_classification(const dimensions&, const dimensions&, const dimensions&);

   void set(const string&);

   void set_layers_number(const Index&);

   void set_layer_input_indices(const vector<vector<Index>>&);
   void set_layer_inputs_indices(const Index&, const vector<Index>&);

   void set_layer_inputs_indices(const string&, const vector<string>&);
   void set_layer_inputs_indices(const string&, const initializer_list<string>&);
   void set_layer_inputs_indices(const string&, const string&);

   void set_model_type(const ModelType&);
   void set_model_type_string(const string&);
   void set_input_names(const vector<string>&);
   void set_output_namess(const vector<string>&);

   void set_input_dimensions(const dimensions&);

   void set_default();

   void set_threads_number(const int&);

   void set_display(const bool&);

   // Layers

   static bool is_trainable(const Layer::Type&);

   Index get_layers_number() const;
   Index get_first_trainable_layer_index() const;
   Index get_last_trainable_layer_index() const;

   Index get_perceptron_layers_number() const;
   Index get_probabilistic_layers_number() const;
   Index get_flatten_layers_number() const;
   Index get_convolutional_layers_number() const;
   Index get_pooling_layers_number() const;
   Index get_long_short_term_memory_layers_number() const;
   Index get_recurrent_layers_number() const;

   bool is_input_layer(const vector<Index>&) const;
   bool is_context_layer(const vector<Index>&) const;

   // Architecture

   Index get_inputs_number() const;
   Index get_outputs_number() const;
   dimensions get_output_dimensions() const;

   // Parameters

   Index get_parameters_number() const;
   Tensor<type, 1> get_parameters() const;

   vector<Index> get_layer_parameter_numbers() const;

   void set_parameters(const Tensor<type, 1>&) const;

   // Parameters initialization

   void set_parameters_constant(const type&) const;

   void set_parameters_random() const;

   // Output

   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);

   Tensor<type, 2> calculate_outputs(const Tensor<type, 4>&);

   Tensor<type, 2> calculate_scaled_outputs(const Tensor<type, 2>&)
   {
       return Tensor<type, 2>();
   }

   Tensor<type, 2> calculate_directional_inputs(const Index&, const Tensor<type, 1>&, const type&, const type&, const Index& = 101) const;

   Index calculate_image_output(const string&);

   // Serialization

   Tensor<string, 2> get_perceptron_layers_information() const;
   Tensor<string, 2> get_probabilistic_layer_information() const;

   void from_XML(const XMLDocument&);
   void inputs_from_XML(const XMLDocument&);
   void layers_from_XML(const XMLDocument&);
   void outputs_from_XML(const XMLDocument&);

   void to_XML(XMLPrinter&) const;

   void print() const;
   void save(const string&) const;
   void save_parameters(const string&) const;

   void load(const string&);
   void load_parameters_binary(const string&);

   vector<string> get_layer_names() const;
   vector<string> get_layer_types_string() const;

   void save_outputs(Tensor<type, 2>&, const string&);

   void forward_propagate(const vector<pair<type*, dimensions>>&,
                          ForwardPropagation&,
                          const bool& = false) const;

   void forward_propagate(const vector<pair<type*, dimensions>>&,
                          const Tensor<type, 1>&, 
                          ForwardPropagation&) const;

   string get_expression() const;

   void save_expression_c(const string&) const;
   void save_expression_python(const string&) const;
   void save_expression_api(const string&) const;
   void save_expression_javascript(const string&) const;


#ifdef OPENNN_CUDA
#include "../../opennn_cuda/opennn_cuda/neural_network_cuda.h"
#endif

protected:

   string name = "neural_network";

   NeuralNetwork::ModelType model_type;

   vector<string> input_names;

   vector<string> output_names;

   vector<unique_ptr<Layer>> layers;

   vector<vector<Index>> layer_input_indices;

   ThreadPool* thread_pool;
   ThreadPoolDevice* thread_pool_device;

   bool display = true;

};

#ifdef OPENNN_CUDA
#include "../../opennn_cuda/opennn_cuda/forward_propagation_cuda.h"
#include "../../opennn_cuda/opennn_cuda/neural_network_back_propagation_cuda.h"
#endif

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
