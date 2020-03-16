//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   C L A S S   H E A D E R             
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef PERCEPTRONLAYER_H
#define PERCEPTRONLAYER_H

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
#include "probabilistic_layer.h"
#include "opennn_strings.h"

#ifdef __OPENNN_CUDA__
    #include "../../artelnics/opennn_cuda/opennn_cuda/kernels.h"
    #include "cuda_runtime_api.h"
#endif

namespace OpenNN
{

/// This class represents a layer of perceptrons.

/// PerceptronLayer is a single-layer network with a hard-limit trabsfer function.
/// This network is often trained with the perceptron learning rule.
///
/// Layers of perceptrons will be used to construct multilayer perceptrons, such as an approximation problems .

class PerceptronLayer : public Layer
{

public:

    /// Enumeration of available activation functions for the perceptron neuron model.

    enum ActivationFunction{Threshold, SymmetricThreshold, Logistic, HyperbolicTangent, Linear, RectifiedLinear,
                            ExponentialLinear, ScaledExponentialLinear, SoftPlus, SoftSign, HardSigmoid};

   // Constructors

   explicit PerceptronLayer();

   explicit PerceptronLayer(const Index&, const Index&, const ActivationFunction& = PerceptronLayer::HyperbolicTangent);

   PerceptronLayer(const PerceptronLayer&);

   // Destructor
   
   virtual ~PerceptronLayer();

   // Get methods

   bool is_empty() const;

   Tensor<Index, 1> get_input_variables_dimensions() const;

   Index get_inputs_number() const;
   Index get_neurons_number() const;

   // Parameters

   const Tensor<type, 2>& get_biases() const;
   const Tensor<type, 2>& get_synaptic_weights() const;

   Tensor<type, 2> get_biases(const Tensor<type, 1>&) const;
   Tensor<type, 2> get_synaptic_weights(const Tensor<type, 1>&) const;

   Index get_biases_number() const;
   Index get_synaptic_weights_number() const;
   Index get_parameters_number() const;
   Tensor<type, 1> get_parameters() const;

   // Activation functions

   const PerceptronLayer::ActivationFunction& get_activation_function() const;

   string write_activation_function() const;

   // Display messages

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const Index&, const Index&, const PerceptronLayer::ActivationFunction& = PerceptronLayer::HyperbolicTangent);
   void set(const PerceptronLayer&);

   void set_default();

   // Architecture

   void set_inputs_number(const Index&);
   void set_neurons_number(const Index&);

   // Parameters

   void set_biases(const Tensor<type, 2>&);
   void set_synaptic_weights(const Tensor<type, 2>&);

   void set_parameters(const Tensor<type, 1>&, const Index& index=0);

   // Activation functions

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);

   // Display messages

   void set_display(const bool&);

   // Parameters initialization methods
   void set_biases_constant(const type&);
   void set_synaptic_weights_constant(const type&);
   void set_synaptic_weights_constant_glorot_uniform();

   void set_parameters_constant(const type&);

   void set_parameters_random();

   // Perceptron layer combinations_2d

   void calculate_combinations(const Tensor<type, 2>& inputs,
                               const Tensor<type, 2>& biases,
                               const Tensor<type, 2>& synaptic_weights,
                               Tensor<type, 2>& combinations_2d) const
   {
       const Index batch_instances_number = inputs.dimension(0);
       const Index biases_number = get_biases_number();

       for(Index i = 0; i < biases_number; i++)
       {
           fill_n(combinations_2d.data() + i*batch_instances_number, batch_instances_number, biases(i));
       }

       switch(device_pointer->get_type())
       {
            case Device::EigenDefault:
            {
                DefaultDevice* default_device = device_pointer->get_eigen_default_device();

                combinations_2d.device(*default_device) += inputs.contract(synaptic_weights, A_B);

                return;
            }

            case Device::EigenSimpleThreadPool:
            {
               ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

               combinations_2d.device(*thread_pool_device) += inputs.contract(synaptic_weights, A_B);

                return;
            }

           case Device::EigenGpu:
           {
#ifdef EIGEN_USE_GPU
                GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

                //combinations_2d.device(*gpu_device) = inputs.contract(synaptic_weights, product_dimensions);
#endif

                return;
           }
       }
   }

   // Perceptron layer activations_2d

   void calculate_activations(const Tensor<type, 2>& combinations_2d, Tensor<type, 2>& activations_2d) const
   {
        #ifdef __OPENNN_DEBUG__

        const Index neurons_number = get_neurons_number();

        const Index combinations_columns_number = combinations_2d.dimension(1);

        if(combinations_columns_number != neurons_number)
        {
           ostringstream buffer;

           buffer << "OpenNN Exception: PerceptronLayer class.\n"
                  << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                  << "Number of combinations_2d columns (" << combinations_columns_number
                  << ") must be equal to number of neurons (" << neurons_number << ").\n";

           throw logic_error(buffer.str());
        }

        #endif

        switch(activation_function)
        {
            case Linear: linear(combinations_2d, activations_2d); return;

            case Logistic: logistic(combinations_2d, activations_2d); return;

            case HyperbolicTangent: hyperbolic_tangent(combinations_2d, activations_2d); return;

            case Threshold: threshold(combinations_2d, activations_2d); return;

            case SymmetricThreshold: symmetric_threshold(combinations_2d, activations_2d); return;

            case RectifiedLinear: rectified_linear(combinations_2d, activations_2d); return;

            case ScaledExponentialLinear: scaled_exponential_linear(combinations_2d, activations_2d); return;

            case SoftPlus: soft_plus(combinations_2d, activations_2d); return;

            case SoftSign: soft_sign(combinations_2d, activations_2d); return;

            case HardSigmoid: hard_sigmoid(combinations_2d, activations_2d); return;

            case ExponentialLinear: exponential_linear(combinations_2d, activations_2d); return;
        }
   }


   void calculate_activations_derivatives(const Tensor<type, 2>& combinations_2d, Tensor<type, 2>& activations_derivatives) const
   {
        #ifdef __OPENNN_DEBUG__

        const Index neurons_number = get_neurons_number();

        const Index combinations_columns_number = combinations_2d.dimension(1);

        if(combinations_columns_number != neurons_number)
        {
           ostringstream buffer;

           buffer << "OpenNN Exception: PerceptronLayer class.\n"
                  << "void calculate_activations_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                  << "Number of combinations_2d columns (" << combinations_columns_number
                  << ") must be equal to number of neurons (" << neurons_number << ").\n";

           throw logic_error(buffer.str());
        }

        #endif

        switch(activation_function)
        {
            case Linear: linear_derivatives(combinations_2d, activations_derivatives); return;

            case Logistic: logistic_derivatives(combinations_2d, activations_derivatives); return;

            case HyperbolicTangent: hyperbolic_tangent_derivatives(combinations_2d, activations_derivatives); return;

            case Threshold: threshold_derivatives(combinations_2d, activations_derivatives); return;

            case SymmetricThreshold: symmetric_threshold_derivatives(combinations_2d, activations_derivatives); return;

            case RectifiedLinear: rectified_linear_derivatives(combinations_2d, activations_derivatives); return;

            case ScaledExponentialLinear: scaled_exponential_linear_derivatives(combinations_2d, activations_derivatives); return;

            case SoftPlus: soft_plus_derivatives(combinations_2d, activations_derivatives); return;

            case SoftSign: soft_sign_derivatives(combinations_2d, activations_derivatives); return;

            case HardSigmoid: hard_sigmoid_derivatives(combinations_2d, activations_derivatives); return;

            case ExponentialLinear: exponential_linear_derivatives(combinations_2d, activations_derivatives); return;
        }
   }


   void calculate_derivatives_activations(const Tensor<type, 2>& combinations_2d,
                                          Tensor<type, 2>& activations,
                                          Tensor<type, 2>& activations_derivatives) const
   {
        #ifdef __OPENNN_DEBUG__

        const Index neurons_number = get_neurons_number();

        const Index combinations_columns_number = combinations_2d.dimension(1);

        if(combinations_columns_number != neurons_number)
        {
           ostringstream buffer;

           buffer << "OpenNN Exception: PerceptronLayer class.\n"
                  << "void calculate_activations_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                  << "Number of combinations_2d columns (" << combinations_columns_number
                  << ") must be equal to number of neurons (" << neurons_number << ").\n";

           throw logic_error(buffer.str());
        }

        #endif

        switch(activation_function)
        {
            case Linear: derivatives_linear(combinations_2d, activations, activations_derivatives); return;

            case Logistic: derivatives_logistic(combinations_2d, activations, activations_derivatives); return;

            case HyperbolicTangent: derivatives_hyperbolic_tangent(combinations_2d, activations, activations_derivatives); return;

            case Threshold: derivatives_threshold(combinations_2d, activations, activations_derivatives); return;

            case SymmetricThreshold: derivatives_symmetric_threshold(combinations_2d, activations, activations_derivatives); return;

            case RectifiedLinear: derivatives_rectified_linear(combinations_2d, activations, activations_derivatives); return;

            case ScaledExponentialLinear: derivatives_scaled_exponential_linear(combinations_2d, activations, activations_derivatives); return;

            case SoftPlus: derivatives_soft_plus(combinations_2d, activations, activations_derivatives); return;

            case SoftSign: derivatives_soft_sign(combinations_2d, activations, activations_derivatives); return;

            case HardSigmoid: derivatives_hard_sigmoid(combinations_2d, activations, activations_derivatives); return;

            case ExponentialLinear: derivatives_exponential_linear(combinations_2d, activations, activations_derivatives); return;
        }
   }


   // Perceptron layer outputs

   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);
   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 1>&);

   void forward_propagate(const Tensor<type, 2>& inputs,
                                      ForwardPropagation& forward_propagation) const
    {
#ifdef __OPENNN_DEBUG__

       const Index inputs_number = get_inputs_number();

       if(inputs_number != inputs.dimension(1))
       {
           ostringstream buffer;

           buffer << "OpenNN Exception: PerceptronLayer class.\n"
                  << "void forward_propagate(const Tensor<type, 2>&,, ForwardPropagation&) method.\n"
                  << "Number of inputs columns (" << inputs.dimension(1) << ") must be equal to number of inputs ("
                  << inputs_number << ").\n";

           throw logic_error(buffer.str());
       }

#endif

       calculate_combinations(inputs,
                              biases,
                              synaptic_weights,
                              forward_propagation.combinations_2d);

       calculate_derivatives_activations(forward_propagation.combinations_2d,
                                         forward_propagation.activations_2d,
                                         forward_propagation.activations_derivatives_2d);
   }


   void forward_propagate(const Tensor<type, 2>& inputs,
                                      Tensor<type, 1> potential_parameters,
                                      ForwardPropagation& forward_propagation) const
      {
       const Index neurons_number = get_neurons_number();
       const Index inputs_number = get_inputs_number();

#ifdef __OPENNN_DEBUG__

       if(inputs_number != inputs.dimension(1))
       {
           ostringstream buffer;

           buffer << "OpenNN Exception: PerceptronLayer class.\n"
                  << "void forward_propagate(const Tensor<type, 2>&, Tensor<type, 1>&, ForwardPropagation&) method.\n"
                  << "Number of inputs columns (" << inputs.dimension(1) << ") must be equal to number of inputs ("
                  << inputs_number << ").\n";

           throw logic_error(buffer.str());
       }

#endif

       const TensorMap<Tensor<type, 2>> potential_biases(potential_parameters.data(), neurons_number, 1);

       const TensorMap<Tensor<type, 2>> potential_synaptic_weights(potential_parameters.data()+neurons_number,
                                                                   inputs_number, neurons_number);

       calculate_combinations(inputs, potential_biases, potential_synaptic_weights, forward_propagation.combinations_2d);

       calculate_derivatives_activations(forward_propagation.combinations_2d,
                                         forward_propagation.activations_2d,
                                         forward_propagation.activations_derivatives_2d);
   }


   // Delta methods

   void calculate_output_delta(ForwardPropagation& forward_propagation,
                                  const Tensor<type, 2>& output_gradient,
                                  Tensor<type, 2>& output_delta) const
      {
          switch(device_pointer->get_type())
          {
               case Device::EigenDefault:
               {
                   DefaultDevice* default_device = device_pointer->get_eigen_default_device();

                   output_delta.device(*default_device) = forward_propagation.activations_derivatives_2d*output_gradient;

                   return;
               }

               case Device::EigenSimpleThreadPool:
               {
                  ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

                  output_delta.device(*thread_pool_device) = forward_propagation.activations_derivatives_2d*output_gradient;

                  return;
               }

              case Device::EigenGpu:
              {
                   break;
              }
          }
      }


   void calculate_hidden_delta(Layer* next_layer_pointer,
                               const Tensor<type, 2>&,
                               const Tensor<type, 2>& activations_derivatives,
                               const Tensor<type, 2>& next_layer_delta,
                               Tensor<type, 2>& hidden_delta) const
   {
       const Type next_layer_type = next_layer_pointer->get_type();

       switch(next_layer_type)
       {
            case Perceptron:

            calculate_hidden_delta_perceptron(next_layer_pointer, activations_derivatives, next_layer_delta, hidden_delta);

            return;

            case Probabilistic:

            calculate_hidden_delta_probabilistic(next_layer_pointer, activations_derivatives, next_layer_delta, hidden_delta);

            return;

            default:

            return;
       }
   }

   void calculate_hidden_delta_perceptron(Layer* next_layer_pointer,
                                          const Tensor<type, 2>& activations_derivatives,
                                          const Tensor<type, 2>& next_layer_delta,
                                          Tensor<type, 2>& hidden_delta) const
   {
       const PerceptronLayer* next_perceptron_layer = dynamic_cast<PerceptronLayer*>(next_layer_pointer);

       const Tensor<type, 2>& next_synaptic_weights = next_perceptron_layer->get_synaptic_weights();

       switch(device_pointer->get_type())
       {
            case Device::EigenDefault:
            {
                DefaultDevice* default_device = device_pointer->get_eigen_default_device();

                hidden_delta.device(*default_device) = next_layer_delta.contract(next_synaptic_weights, A_BT);

                hidden_delta.device(*default_device) = hidden_delta*activations_derivatives;

                return;
            }

            case Device::EigenSimpleThreadPool:
            {
               ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

               hidden_delta.device(*thread_pool_device) = next_layer_delta.contract(next_synaptic_weights, A_BT);

               hidden_delta.device(*thread_pool_device) = hidden_delta*activations_derivatives;

               return;
            }

           case Device::EigenGpu:
           {
//                GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

                break;
           }
       }
   }


   void calculate_hidden_delta_probabilistic(Layer* next_layer_pointer,
                                             const Tensor<type, 2>& activations_derivatives,
                                             const Tensor<type, 2>& next_layer_delta,
                                             Tensor<type, 2>& hidden_delta) const
   {
       const ProbabilisticLayer* next_probabilistic_layer = dynamic_cast<ProbabilisticLayer*>(next_layer_pointer);

       const Tensor<type, 2>& next_synaptic_weights = next_probabilistic_layer->get_synaptic_weights();

       switch(device_pointer->get_type())
       {
            case Device::EigenDefault:
            {
                DefaultDevice* default_device = device_pointer->get_eigen_default_device();

                hidden_delta.device(*default_device) = next_layer_delta.contract(next_synaptic_weights, A_BT);

                hidden_delta.device(*default_device) = hidden_delta*activations_derivatives;

                return;
            }

            case Device::EigenSimpleThreadPool:
            {
               ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

               hidden_delta.device(*thread_pool_device) = next_layer_delta.contract(next_synaptic_weights, A_BT);

               hidden_delta.device(*thread_pool_device) = hidden_delta*activations_derivatives;

               return;
            }

           case Device::EigenGpu:
           {
//                GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

                break;
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

                back_propagation.biases_derivatives.device(*default_device)
                        = back_propagation.delta.sum(Eigen::array<Index, 1>({0}));

                back_propagation.synaptic_weights_derivatives.device(*default_device)
                        = inputs.contract(back_propagation.delta, AT_B);

                return;
            }

            case Device::EigenSimpleThreadPool:
            {
                ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

                back_propagation.biases_derivatives.device(*thread_pool_device)
                        = back_propagation.delta.sum(Eigen::array<Index, 1>({0}));

                back_propagation.synaptic_weights_derivatives.device(*thread_pool_device)
                        = inputs.contract(back_propagation.delta, AT_B);

                return;
            }

           case Device::EigenGpu:
           {
//                GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

                return;
           }
       }
   }


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

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
   string write_activation_function_expression() const;

   string object_to_string() const;

   // Serialization methods

   void from_XML(const tinyxml2::XMLDocument&);
   void write_XML(tinyxml2::XMLPrinter&) const;

protected:

   // MEMBERS

   /// Bias is a neuron parameter that is summed with the neuron's weighted inputs
   /// and passed through the neuron's trabsfer function to generate the neuron's output.

   Tensor<type, 2> biases;

   /// This matrix containing conection strengths from a layer's inputs to its neurons.

   Tensor<type, 2> synaptic_weights;

   /// Activation function variable.

   ActivationFunction activation_function;

   /// Display messages to screen. 

   bool display;

#ifdef __OPENNN_CUDA__
    #include "../../artelnics/opennn_cuda/opennn_cuda/perceptron_layer_cuda.h"
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
