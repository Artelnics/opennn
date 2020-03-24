//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef PROBABILISTICLAYER_H
#define PROBABILISTICLAYER_H

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

#include "tinyxml2.h"

#include "opennn_strings.h"

namespace OpenNN
{

/// This class represents a layer of probabilistic neurons.

///
/// The neural network defined in OpenNN includes a probabilistic layer for those problems
/// when the outptus are to be interpreted as probabilities.
/// It does not has Synaptic weights or Biases

class ProbabilisticLayer : public Layer
{

public:

   // Constructors

   explicit ProbabilisticLayer();

   explicit ProbabilisticLayer(const Index&, const Index&);

   ProbabilisticLayer(const ProbabilisticLayer&);

   // Destructor

   virtual ~ProbabilisticLayer();

   // Enumerations

   /// Enumeration of available methods for interpreting variables as probabilities.

   enum ActivationFunction{Binary, Logistic, Competitive, Softmax};

   // Get methods

   Tensor<Index, 1> get_input_variables_dimensions() const;

   Index get_inputs_number() const;
   Index get_neurons_number() const;

   Index get_biases_number() const;
   Index get_synaptic_weights_number() const;

   const type& get_decision_threshold() const;

   const ActivationFunction& get_activation_function() const;
   string write_activation_function() const;
   string write_activation_function_text() const;

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const Index&, const Index&);
   void set(const ProbabilisticLayer&);

   void set_inputs_number(const Index&);
   void set_neurons_number(const Index&);

   void set_biases(const Tensor<type, 2>&);
   void set_synaptic_weights(const Tensor<type, 2>&);

   void set_parameters(const Tensor<type, 1>&, const Index& index=0);

   void set_decision_threshold(const type&);

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);

   virtual void set_default();

   // Parameters

   const Tensor<type, 2>& get_biases() const;
   const Tensor<type, 2>& get_synaptic_weights() const;

   Tensor<type, 2> get_biases(Tensor<type, 1>&) const;
   Tensor<type, 2> get_synaptic_weights(Tensor<type, 1>&) const;

   Index get_parameters_number() const;
   Tensor<type, 1> get_parameters() const;

   // Display messages

   void set_display(const bool&);

   // Parameters initialization methods

   void set_biases_constant(const type&);
   void set_synaptic_weights_constant(const type&);
   void set_synaptic_weights_constant_Glorot(const type&,const type&);

   void set_parameters_constant(const type&);

   void set_parameters_random();

   // Combinations

   void calculate_combinations(const Tensor<type, 2>& inputs,
                               const Tensor<type, 2>& biases,
                               const Tensor<type, 2>& synaptic_weights,
                               Tensor<type, 2>& combinations_2d) const
   {
       const Index batch_instances_number = inputs.dimension(0);
       const Index biases_number = get_neurons_number();

       for(Index i = 0; i < biases_number; i++)
       {
           fill_n(combinations_2d.data()+i*batch_instances_number, batch_instances_number, biases(i));
       }

       switch(device_pointer->get_type())
       {
            case Device::EigenDefault:
            {
                DefaultDevice* default_device = device_pointer->get_eigen_default_device();

                combinations_2d.device(*default_device) += inputs.contract(synaptic_weights, A_B);

                break;
            }

            case Device::EigenSimpleThreadPool:
            {
               ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

               combinations_2d.device(*thread_pool_device) += inputs.contract(synaptic_weights, A_B);

                break;
            }


           case Device::EigenGpu:
           {
#ifdef EIGEN_USE_GPU

                GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

                //combinations_2d.device(*gpu_device) = inputs.contract(synaptic_weights, product_dimensions);
#endif

                break;
           }
       }
   }

   // Outputs

   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);
   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 1>&);

   void forward_propagate(const Tensor<type, 2>& inputs,
                          ForwardPropagation& forward_propagation) const
   {

       calculate_combinations(inputs, biases, synaptic_weights, forward_propagation.combinations_2d);

       calculate_activations(forward_propagation.combinations_2d, forward_propagation.activations_2d);

       calculate_activations_derivatives(forward_propagation.combinations_2d, forward_propagation.activations_2d, forward_propagation.activations_derivatives_3d);
   }

   // Activations

   void calculate_activations(const Tensor<type, 2>& combinations_2d, Tensor<type, 2>& activations_2d) const
   {
        #ifdef __OPENNN_DEBUG__

        const Index dimensions_number = combinations_2d.rank();

        if(dimensions_number != 2)
        {
           ostringstream buffer;

           buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                  << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                  << "Dimensions of combinations_2d (" << dimensions_number << ") must be 2.\n";

           throw logic_error(buffer.str());
        }

        const Index neurons_number = get_neurons_number();

        const Index combinations_columns_number = combinations_2d.dimension(1);

        if(combinations_columns_number != neurons_number)
        {
           ostringstream buffer;

           buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                  << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                  << "Number of combinations_2d columns (" << combinations_columns_number << ") must be equal to number of neurons (" << neurons_number << ").\n";

           throw logic_error(buffer.str());
        }

        #endif

        switch(activation_function)
        {
            case Binary: binary(combinations_2d, activations_2d); return;

            case Logistic: logistic(combinations_2d, activations_2d); return;

            case Competitive: competitive(combinations_2d, activations_2d); return;

            case Softmax: softmax(combinations_2d, activations_2d); return;
        }

        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
               << "Unknown probabilistic method.\n";

        throw logic_error(buffer.str());
   }

   void calculate_activations_derivatives(const Tensor<type, 2>& combinations_2d,
                                          Tensor<type, 2>& activations,
                                          Tensor<type, 3>& activations_derivatives) const
   {
        #ifdef __OPENNN_DEBUG__

        const Index neurons_number = get_neurons_number();

        const Index combinations_columns_number = combinations_2d.dimension(1);

        if(combinations_columns_number != neurons_number)
        {
           ostringstream buffer;

           buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                  << "void calculate_activations_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                  << "Number of combinations_2d columns (" << combinations_columns_number
                  << ") must be equal to number of neurons (" << neurons_number << ").\n";

           throw logic_error(buffer.str());
        }

        #endif

        switch(activation_function)
        {

            case Logistic: logistic_derivatives(combinations_2d, activations, activations_derivatives); return;

            case Softmax: softmax_derivatives(combinations_2d, activations, activations_derivatives); return;

       }
   }


   void calculate_output_delta(ForwardPropagation& forward_propagation,
                                  const Tensor<type, 2>& output_gradient,
                                  Tensor<type, 2>& output_delta) const
   {
       const Index neurons_number = get_neurons_number();
       const Index batch_instances_number = forward_propagation.activations_derivatives_3d.dimension(0);

       if(neurons_number == 1)
       {
           TensorMap< Tensor<type, 2> > activations_derivatives(forward_propagation.activations_derivatives_3d.data(), batch_instances_number, neurons_number);

           switch(device_pointer->get_type())
           {
           case Device::EigenDefault:
           {
               DefaultDevice* default_device = device_pointer->get_eigen_default_device();

               output_delta.device(*default_device) = activations_derivatives*output_gradient;

               return;
           }

           case Device::EigenSimpleThreadPool:
           {
               ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

               output_delta.device(*thread_pool_device) = activations_derivatives*output_gradient;

               return;
           }

           case Device::EigenGpu:
           {
               //                 GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

               break;
           }
           }
       }
       else
       {

           Tensor<type, 3> activations_derivatives = forward_propagation.activations_derivatives_3d;

           cout << "Activations derivatives: " << activations_derivatives << endl;

           cout << "output_gradient: " << output_gradient << endl;

           const Index items_number = output_gradient.dimension(1);

           const Index rows_number = activations_derivatives.dimension(0);
           const Index columns_number = activations_derivatives.dimension(1);
           const Index matrix_number = activations_derivatives.dimension(2);
           /*
              if(output_gradient.dimension(0) != matrix_number)
              {
                  ostringstream buffer;

                  buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                         << "void calculate_output_delta(ForwardPropagation& ,const Tensor<type, 2>& ,Tensor<type, 2>& ) const.\n"
                         << "Pointer to neural network is nullptr.\n";

                  throw logic_error(buffer.str());
              }
              if(items_number != rows_number)
              {
                  ostringstream buffer;

                  buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                         << "void calculate_output_delta(ForwardPropagation& ,const Tensor<type, 2>& ,Tensor<type, 2>& ) const.\n"
                         << "Pointer to neural network is nullptr.\n";

                  throw logic_error(buffer.str());
              }
*/

           Tensor<type, 1> row_values(items_number);
           Tensor<type, 1> output_delta_row(neurons_number);

           Index index = 0;
           Index step = rows_number*columns_number;

           switch(device_pointer->get_type())
           {
           case Device::EigenDefault:
           {
               DefaultDevice* default_device = device_pointer->get_eigen_default_device();

               for(Index i = 0; i < matrix_number; i++)
               {
                   row_values = output_gradient.chip(i,0);

                   TensorMap< Tensor<type, 2> > activations_derivatives_matrix(activations_derivatives.data()+index,
                                                                               batch_instances_number, neurons_number);

                   output_delta_row.device(*default_device) = row_values.contract(activations_derivatives_matrix, A_B);

                   for(Index j = 0; j < neurons_number; j++)
                   {
                       output_delta(i,j) = output_delta_row(j);
                   }

                   index += step;
               }

               return;
           }

           case Device::EigenSimpleThreadPool:
           {
               ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

               for(Index i = 0; i < matrix_number; i++)
               {
                   row_values = output_gradient.chip(i,0);

                   TensorMap< Tensor<type, 2> > activations_derivatives_matrix(activations_derivatives.data()+index,
                                                                               batch_instances_number, neurons_number);

                   output_delta_row.device(*thread_pool_device) = row_values.contract(activations_derivatives_matrix, AT_B);

                   for(Index j = 0; j < items_number; j++)
                   {
                       output_delta(i,j) = output_delta_row(j);
                   }

                   index += step;
               }

               return;
           }

           case Device::EigenGpu:
           {
               //                 GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

               break;
           }
           }
       }
   }

   void calculate_error_gradient(const Tensor<type, 2>& inputs,
                                 const Layer::ForwardPropagation&,
                                 Layer::BackPropagation& back_propagation) const
   {
       switch(device_pointer->get_type())
       {
            case Device::EigenDefault:
            {
                DefaultDevice* default_device = device_pointer->get_eigen_default_device();

                back_propagation.biases_derivatives.device(*default_device) = back_propagation.delta.sum(Eigen::array<Index, 1>({0}));

                back_propagation.synaptic_weights_derivatives.device(*default_device) = inputs.contract(back_propagation.delta, AT_B);

                return;
            }

            case Device::EigenSimpleThreadPool:
            {
                ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

                back_propagation.biases_derivatives.device(*thread_pool_device) = back_propagation.delta.sum(Eigen::array<Index, 1>({0}));

                back_propagation.synaptic_weights_derivatives.device(*thread_pool_device) = inputs.contract(back_propagation.delta, AT_B);

                return;
            }

           case Device::EigenGpu:
           {
//                GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

                break;
           }
       }
   }


   void insert_parameters(const Tensor<type, 1>& parameters, const Index& )
   {
       const Index biases_number = get_biases_number();
       const Index synaptic_weights_number = get_synaptic_weights_number();

       memcpy(biases.data() , parameters.data(), static_cast<size_t>(biases_number)*sizeof(type));
       memcpy(synaptic_weights.data(), parameters.data() + biases_number, static_cast<size_t>(synaptic_weights_number)*sizeof(type));
   }

   // calculate error gradient

   void insert_gradient(const BackPropagation& back_propagation, const Index& index, Tensor<type, 1>& gradient) const
   {
       const Index biases_number = get_biases_number();
       const Index synaptic_weights_number = get_synaptic_weights_number();

       memcpy(gradient.data() + index,
              back_propagation.biases_derivatives.data(),
              static_cast<size_t>(biases_number)*sizeof(type));

       memcpy(gradient.data() + index + biases_number,
              back_propagation.synaptic_weights_derivatives.data(),
              static_cast<size_t>(synaptic_weights_number)*sizeof(type));
   }

   // Expression methods

   string write_binary_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
   string write_probability_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
   string write_competitive_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
   string write_softmax_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
   string write_no_probabilistic_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   // Serialization methods

   string object_to_string() const;

   virtual tinyxml2::XMLDocument* to_XML() const;

   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

protected:

   /// Bias is a neuron parameter that is summed with the neuron's weighted inputs
   /// and passed through the neuron's trabsfer function to generate the neuron's output.

   Tensor<type, 2> biases;

   /// This matrix containing conection strengths from a layer's inputs to its neurons.

   Tensor<type, 2> synaptic_weights;

   /// Activation function variable.

   ActivationFunction activation_function = Logistic;

   type decision_threshold;

   /// Display messages to screen.

   bool display;
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
