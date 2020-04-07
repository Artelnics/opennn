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

#include "vector.h"
#include "matrix.h"

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

   explicit NeuralNetwork(const NeuralNetwork::ProjectType&, const Vector<size_t>&);

   explicit NeuralNetwork(const Vector<size_t>&, const size_t&, const Vector<size_t>&, const size_t&);

   explicit NeuralNetwork(const string&);

   explicit NeuralNetwork(const tinyxml2::XMLDocument&);

   explicit NeuralNetwork(const Vector<Layer*>&);

   NeuralNetwork(const NeuralNetwork&);

   // Destructor

   virtual ~NeuralNetwork();

   // APPENDING LAYERS

   void add_layer(Layer*);

   bool check_layer_type(const Layer::LayerType);

   // Get methods

   bool has_scaling_layer() const;
   bool has_principal_components_layer() const;
   bool has_long_short_term_memory_layer() const;
   bool has_recurrent_layer() const;
   bool has_unscaling_layer() const;
   bool has_bounding_layer() const;
   bool has_probabilistic_layer() const;

   bool is_empty() const;  

   Vector<string> get_inputs_names() const;
   string get_input_name(const size_t&) const;
   size_t get_input_index(const string&) const;

   Vector<string> get_outputs_names() const;
   string get_output_name(const size_t&) const;
   size_t get_output_index(const string&) const;

   Vector<Layer*> get_layers_pointers() const;
   Vector<Layer*> get_trainable_layers_pointers() const;
   Vector<size_t> get_trainable_layers_indices() const;

   ScalingLayer* get_scaling_layer_pointer() const;
   UnscalingLayer* get_unscaling_layer_pointer() const;
   BoundingLayer* get_bounding_layer_pointer() const;
   ProbabilisticLayer* get_probabilistic_layer_pointer() const;
   PrincipalComponentsLayer* get_principal_components_layer_pointer() const;
   LongShortTermMemoryLayer* get_long_short_term_memory_layer_pointer() const;
   RecurrentLayer* get_recurrent_layer_pointer() const;

   Layer* get_output_layer_pointer() const;
   Layer* get_layer_pointer(const size_t&) const;
   PerceptronLayer* get_first_perceptron_layer_pointer() const;

   const bool& get_display() const;

   // Set methods

   void set();

   void set(const NeuralNetwork::ProjectType&, const Vector<size_t>&);
   void set(const Vector<size_t>&, const size_t&, const Vector<size_t>&, const size_t&);

   void set(const string&);
   void set(const NeuralNetwork&);

   void set_inputs_names(const Vector<string>&);
   void set_outputs_names(const Vector<string>&);

   void set_inputs_number(const size_t&);
   void set_inputs_number(const Vector<bool>&);

   virtual void set_default();

   void set_layers_pointers(Vector<Layer*>&);

   void set_scaling_layer(ScalingLayer&);

   void set_display(const bool&);

   // Layers 

   size_t get_layers_number() const;
   Vector<size_t> get_layers_neurons_numbers() const;

   size_t get_trainable_layers_number() const;

   // Architecture

   size_t get_inputs_number() const;
   size_t get_outputs_number() const;

   Vector<size_t> get_architecture() const;

   // Parameters

   size_t get_parameters_number() const;
   size_t get_trainable_parameters_number() const;
   Vector<double> get_parameters() const;

   Vector<size_t> get_trainable_layers_parameters_numbers() const;

   Vector<Vector<double>> get_trainable_layers_parameters(const Vector<double>&) const;

   void set_parameters(const Vector<double>&);

   // Parameters initialization methods

   void initialize_parameters(const double&);

   void randomize_parameters_uniform(const double& = -1.0, const double& = 1.0);  

   void randomize_parameters_normal(const double& = 0.0, const double& = 1.0);  

   // Parameters

   double calculate_parameters_norm() const;
   Descriptives calculate_parameters_descriptives() const;
   Histogram calculate_parameters_histogram(const size_t& = 10) const;

   void perturbate_parameters(const double&);

   // Output 

   Tensor<double> calculate_outputs(const Tensor<double>&);
   Eigen::MatrixXd calculate_outputs_eigen(const Eigen::MatrixXd&);

   Tensor<double> calculate_trainable_outputs(const Tensor<double>&) const;

   Tensor<double> calculate_trainable_outputs(const Tensor<double>&, const Vector<double>&) const;

   Matrix<double> calculate_directional_inputs(const size_t&, const Vector<double>&, const double&, const double&, const size_t& = 101) const;

   Vector<Histogram> calculate_outputs_histograms(const size_t& = 1000, const size_t& = 10);
   Vector<Histogram> calculate_outputs_histograms(const Tensor<double>&, const size_t& = 10);

   vector<double> calculate_outputs_std(const vector<double>&);

   // Serialization methods

   string object_to_string() const;
 
   Matrix<string> get_information() const;

   virtual tinyxml2::XMLDocument* to_XML() const;
   virtual void from_XML(const tinyxml2::XMLDocument&);
   void inputs_from_XML(const tinyxml2::XMLDocument&);
   void layers_from_XML(const tinyxml2::XMLDocument&);
   void outputs_from_XML(const tinyxml2::XMLDocument&);

   virtual void write_XML(tinyxml2::XMLPrinter&) const;
   // virtual void read_XML(  );

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

   /// Calculate de Forward Propagation in Neural Network   

   Vector<Layer::FirstOrderActivations> calculate_trainable_forward_propagation(const Tensor<double>&) const;

protected:

   /// Names of inputs

   Vector<string> inputs_names;

   /// Names of ouputs

   Vector<string> outputs_names;

   /// Layers

   Vector<Layer*> layers_pointers;

   /// Display messages to screen.

   bool display = true;
};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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
