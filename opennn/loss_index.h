//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   I N D E X   C L A S S   H E A D E R                         
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LOSSINDEX_H
#define LOSSINDEX_H

// System includes

#include <string>
#include <sstream>
#include <fstream>
#include <ostream>
#include <iostream>
#include <cmath>

// OpenNN includes

#include "config.h"
#include "device.h"
#include "data_set.h"
#include "neural_network.h"
#include "numerical_differentiation.h"
#include "tinyxml2.h"

namespace OpenNN
{

/// This abstract class represents the concept of loss index composed of an error term and a regularization term.

///
/// The error terms could be:
/// <ul>
/// <li> Cross Entropy Error.
/// <li> Mean Squared Error.
/// <li> Minkowski Error.
/// <li> Normalized Squared Error.
/// <li> Sum Squared Error.
/// <li> Weighted Squared Error.
/// </ul>

class LossIndex
{

public:

   // Constructors

   explicit LossIndex();

   explicit LossIndex(NeuralNetwork*);

   explicit LossIndex(DataSet*);

   explicit LossIndex(NeuralNetwork*, DataSet*);

   explicit LossIndex(const tinyxml2::XMLDocument&);

   LossIndex(const LossIndex&);

   // Destructor

   virtual ~LossIndex();

   // Methods

   /// Enumeration of available regularization methods.

   enum RegularizationMethod{L1, L2, NoRegularization};


   /// A loss index composed of several terms, this structure represent the First Order for this function.

   ///
   /// Set of loss value and gradient vector of the loss index.
   /// A method returning this structure might be implemented more efficiently than the loss and gradient methods separately.

   struct BackPropagation
   {
       /// Default constructor.

       explicit BackPropagation() {}

       explicit BackPropagation(const Index& new_batch_instances_number, LossIndex* new_loss_index_pointer)
       {
           set(new_batch_instances_number, new_loss_index_pointer);
       }

       virtual ~BackPropagation();

       void set(const Index& new_batch_instances_number, LossIndex* new_loss_index_pointer)
       {                      
           batch_instances_number = new_batch_instances_number;

           loss_index_pointer = new_loss_index_pointer;

           // Neural network

           NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

           const Index parameters_number = neural_network_pointer->get_parameters_number();

           const Index outputs_number = neural_network_pointer->get_outputs_number();

           // First order loss

           errors.resize(batch_instances_number, 1);

           loss = 0;

           output_gradient.resize(batch_instances_number, outputs_number);

           neural_network.set(batch_instances_number, neural_network_pointer);

           gradient.resize(parameters_number);
       }


       void print()
       {
           cout << "Errors:" << endl;
           cout << errors << endl;

           cout << "Loss:" << endl;
           cout << loss << endl;

           cout << "Output gradient:" << endl;
           cout << output_gradient << endl;

           cout << "Gradient:" << endl;
           cout << gradient << endl; 
       }

       LossIndex* loss_index_pointer = nullptr;

       Index batch_instances_number = 0;

       NeuralNetwork::BackPropagation neural_network;

       Tensor<type, 2> output_gradient;

       type loss;

       Tensor<type, 2> errors;

       Tensor<type, 1> gradient;
   };


   /// This structure contains second order information about the loss function (loss, gradient and Hessian).

   ///
   /// Set of loss value, gradient vector and <i>Hessian</i> matrix of the loss index.
   /// A method returning this structure might be implemented more efficiently than the loss,
   /// gradient and <i>Hessian</i> methods separately.

   struct SecondOrderLoss
   {
       /// Default constructor.

       SecondOrderLoss() {}

       SecondOrderLoss(const Index& parameters_number)
       {
           loss = 0;
           gradient = Tensor<type, 1>(parameters_number);
           hessian = Tensor<type, 2>(parameters_number, parameters_number);
       }

       type loss;
       Tensor<type, 1> gradient;
       Tensor<type, 2> hessian;
   };


   /// Returns a pointer to the neural network object associated to the error term.

   inline NeuralNetwork* get_neural_network_pointer() const 
   {
        #ifdef __OPENNN_DEBUG__

        if(!neural_network_pointer)
        {
             ostringstream buffer;

             buffer << "OpenNN Exception: LossIndex class.\n"
                    << "NeuralNetwork* get_neural_network_pointer() const method.\n"
                    << "Neural network pointer is nullptr.\n";

             throw logic_error(buffer.str());
        }

        #endif

      return neural_network_pointer;
   }

   /// Returns a pointer to the data set object associated to the error term.

   inline DataSet* get_data_set_pointer() const 
   {
        #ifdef __OPENNN_DEBUG__

        if(!data_set_pointer)
        {
             ostringstream buffer;

             buffer << "OpenNN Exception: LossIndex class.\n"
                    << "DataSet* get_data_set_pointer() const method.\n"
                    << "DataSet pointer is nullptr.\n";

             throw logic_error(buffer.str());
        }

        #endif

      return data_set_pointer;
   }

   const type& get_regularization_weight() const;

   const bool& get_display() const;

   bool has_neural_network() const;

   bool has_data_set() const;

   // Get methods

   RegularizationMethod get_regularization_method() const;

   // Set methods

   void set();
   void set(NeuralNetwork*);
   void set(DataSet*);
   void set(NeuralNetwork*, DataSet*);

   void set(const LossIndex&);

   void set_device_pointer(Device*);

   void set_neural_network_pointer(NeuralNetwork*);

   void set_data_set_pointer(DataSet*);

   void set_default();

   void set_regularization_method(const RegularizationMethod&);
   void set_regularization_method(const string&);
   void set_regularization_weight(const type&);

   void set_display(const bool&);

   bool has_selection() const;

   // GRADIENT METHODS

   virtual void calculate_output_gradient(const DataSet::Batch&,
                                          const NeuralNetwork::ForwardPropagation&,
                                          BackPropagation&) const = 0;

   Tensor<type, 1> calculate_training_error_gradient_numerical_differentiation() const;

   // ERROR TERMS METHODS

   virtual Tensor<type, 1> calculate_batch_error_terms(const Tensor<Index, 1>&) const {return Tensor<type, 1>();}
   virtual Tensor<type, 2> calculate_batch_error_terms_Jacobian(const Tensor<Index, 1>&) const {return Tensor<type, 2>();}

   virtual type calculate_error(const DataSet::Batch&, const NeuralNetwork::ForwardPropagation&) const = 0;

   virtual void calculate_error(BackPropagation&) const {}

   void back_propagate(const DataSet::Batch& batch,
                                   const NeuralNetwork::ForwardPropagation& forward_propagation,
                                   BackPropagation& back_propagation) const
   {
       // Loss index

       calculate_errors(batch, forward_propagation, back_propagation);

       calculate_error(back_propagation);

       calculate_output_gradient(batch, forward_propagation, back_propagation);

       calculate_layers_delta(forward_propagation, back_propagation);

       calculate_error_gradient(batch, forward_propagation, back_propagation);

       // Regularization

       if(regularization_method != RegularizationMethod::NoRegularization)
       {      
           const Tensor<type, 1> parameters = neural_network_pointer->get_parameters();

           back_propagation.loss += regularization_weight*calculate_regularization(parameters);

           back_propagation.gradient += regularization_weight*calculate_regularization_gradient(parameters);
       }
   }

   virtual SecondOrderLoss calculate_terms_second_order_loss() const {return SecondOrderLoss();}

   // Regularization methods

   type calculate_regularization(const Tensor<type, 1>&) const;
   Tensor<type, 1> calculate_regularization_gradient(const Tensor<type, 1>&) const;
   Tensor<type, 2> calculate_regularization_hessian(const Tensor<type, 1>&) const;

   // Delta methods

   void calculate_layers_delta(const NeuralNetwork::ForwardPropagation& forward_propagation,
                               BackPropagation& back_propagation) const
   {
        const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

        if(trainable_layers_number == 0) return;

        const Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

        // Output layer

        trainable_layers_pointers[trainable_layers_number-1]
        ->calculate_output_delta(forward_propagation.layers[trainable_layers_number-1].activations_derivatives_2d,
                                 back_propagation.output_gradient,
                                 back_propagation.neural_network.layers[trainable_layers_number-1].delta);

      // Hidden layers

      for(Index i = static_cast<Index>(trainable_layers_number)-2; i >= 0; i--)
      {
          Layer* previous_layer_pointer = trainable_layers_pointers[static_cast<Index>(i+1)];

          trainable_layers_pointers[i]
          ->calculate_hidden_delta(previous_layer_pointer,
                                   forward_propagation.layers[i].activations_2d,
                                   forward_propagation.layers[i].activations_derivatives_2d,
                                   back_propagation.neural_network.layers[i+1].delta,
                                   back_propagation.neural_network.layers[i].delta);
      }
   }

   void calculate_errors(const DataSet::Batch& batch,
                         const NeuralNetwork::ForwardPropagation& forward_propagation,
                         BackPropagation& back_propagation) const
   {
        #ifdef __OPENNN_DEBUG__

        check();

        #endif

        const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

        switch(device_pointer->get_type())
        {
             case Device::EigenDefault:
             {
                 DefaultDevice* default_device = device_pointer->get_eigen_default_device();

                 back_propagation.errors.device(*default_device)
                         = forward_propagation.layers[trainable_layers_number-1].activations_2d - batch.targets_2d;

                 return;
             }

             case Device::EigenSimpleThreadPool:
             {
                ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

                back_propagation.errors.device(*thread_pool_device)
                        = forward_propagation.layers[trainable_layers_number-1].activations_2d - batch.targets_2d;

                return;
             }

            case Device::EigenGpu:
            {
//                 GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

                 return;
            }
        }
   }

   /// @todo Guillermo Change insert_gradient with TensorMap

   void calculate_error_gradient(const DataSet::Batch& batch,
                                 const NeuralNetwork::ForwardPropagation& forward_propagation,
                                 BackPropagation& back_propagation) const
   {
       const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

       #ifdef __OPENNN_DEBUG__

       check();

       #endif

       const Tensor<Index, 1> trainable_layers_parameters_number
               = neural_network_pointer->get_trainable_layers_parameters_numbers();

       const Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

       trainable_layers_pointers[0]->calculate_error_gradient(batch.inputs_2d,
                                                              forward_propagation.layers[0],
                                                              back_propagation.neural_network.layers[0]);

       Index index = 0;

       trainable_layers_pointers[0]->insert_gradient(back_propagation.neural_network.layers[0],
               index, back_propagation.gradient);

       index += trainable_layers_parameters_number[0];

       for(Index i = 1; i < trainable_layers_number; i++)
       {
           trainable_layers_pointers[i]->calculate_error_gradient(
                   forward_propagation.layers[i-1].activations_2d,
                   forward_propagation.layers[i-1],
                   back_propagation.neural_network.layers[i]);

           trainable_layers_pointers[i]->insert_gradient(back_propagation.neural_network.layers[i],
                                                         index,
                                                         back_propagation.gradient);

           index += trainable_layers_parameters_number[i];
       }
   }

   Tensor<type, 2> calculate_layer_error_terms_Jacobian(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   Tensor<type, 2> calculate_error_terms_Jacobian(const Tensor<type, 2>&,
                                                  const Tensor<Layer::ForwardPropagation, 1>&,
                                                  const Tensor<Tensor<type, 2>, 1>&) const;

   // Serialization methods

   tinyxml2::XMLDocument* to_XML() const;

   void from_XML(const tinyxml2::XMLDocument&);

   virtual void write_XML(tinyxml2::XMLPrinter&) const;

   void regularization_from_XML(const tinyxml2::XMLDocument&);
   void write_regularization_XML(tinyxml2::XMLPrinter&) const;

   string get_error_type() const;
   virtual string get_error_type_text() const;

   string write_information() const;

   string write_regularization_method() const;

   // Checking methods

   void check() const;

   // Metrics

   Tensor<type, 2> kronecker_product(const Tensor<type, 1>&, const Tensor<type, 1>&) const;

   type l2_norm(const Tensor<type, 1>& parameters) const
   {
       Tensor<type, 0> norm;

       switch(device_pointer->get_type())
       {
            case Device::EigenDefault:
            {
                DefaultDevice* default_device = device_pointer->get_eigen_default_device();

                norm.device(*default_device) = parameters.square().sum().sqrt();

                break;
            }

            case Device::EigenSimpleThreadPool:
            {
               ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

               norm.device(*thread_pool_device) = parameters.square().sum().sqrt();

                break;
            }

           case Device::EigenGpu:
           {
//                GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

                break;
           }
       }

       return norm(0);
   }

   type l1_norm(const Tensor<type, 1>& parameters) const
   {
       Tensor<type, 0> norm;

       switch(device_pointer->get_type())
       {
            case Device::EigenDefault:
            {
                DefaultDevice* default_device = device_pointer->get_eigen_default_device();

                norm.device(*default_device) = parameters.abs().sum();

                break;
            }

            case Device::EigenSimpleThreadPool:
            {
               ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

               norm.device(*thread_pool_device) = parameters.abs().sum();

                break;
            }

           case Device::EigenGpu:
           {
//                GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

                break;
           }
       }

       return norm(0);
   }

   Tensor<type, 1> l1_norm_gradient(const Tensor<type, 1>& parameters) const
   {
       const Index parameters_number = parameters.size();

       Tensor<type, 1> gradient(parameters_number);

       switch(device_pointer->get_type())
       {
            case Device::EigenDefault:
            {
                DefaultDevice* default_device = device_pointer->get_eigen_default_device();

                gradient.device(*default_device) = parameters.sign();

                return gradient;
            }

            case Device::EigenSimpleThreadPool:
            {
               ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

               gradient.device(*thread_pool_device) = parameters.sign();

               return gradient;
            }

           case Device::EigenGpu:
           {
//                GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

                break;
           }
       }

       return Tensor<type, 1>();
   }


   Tensor<type, 2> l1_norm_hessian(const Tensor<type, 1>& parameters) const
   {
       const Index parameters_number = parameters.size();

       Tensor<type, 2> hessian(parameters_number, parameters_number);

           switch(device_pointer->get_type())
           {
                case Device::EigenDefault:
                {
                    DefaultDevice* default_device = device_pointer->get_eigen_default_device();

                    hessian.device(*default_device) = hessian.setZero();  //<---

                    return hessian;

                }

                case Device::EigenSimpleThreadPool:
                {
                   ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

                   hessian.device(*thread_pool_device) =  hessian.setZero();  //<---

                   return hessian;
                }

               case Device::EigenGpu:
               {
    //                GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

                    break;
               }
           }

           return Tensor<type, 2>();
   }


   Tensor<type, 1> l2_norm_gradient(const Tensor<type, 1>& parameters) const
   {
       const Index parameters_number = parameters.size();

       Tensor<type, 1> gradient(parameters_number);

       const type norm = l2_norm(parameters);

       if(static_cast<Index>(norm) ==  0)
       {
           gradient.setZero();

           return gradient;
       }

       switch(device_pointer->get_type())
       {
            case Device::EigenDefault:
            {
                DefaultDevice* default_device = device_pointer->get_eigen_default_device();

                gradient.device(*default_device) = parameters/norm;

                return gradient;
            }

            case Device::EigenSimpleThreadPool:
            {
               ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

               gradient.device(*thread_pool_device) = parameters/norm;

               return gradient;
            }

           case Device::EigenGpu:
           {
//                GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

                return Tensor<type, 1>();
           }
       }
   }


   Tensor<type, 2> l2_norm_hessian(const Tensor<type, 1>& parameters) const
   {
       const Index parameters_number = parameters.size();

       Tensor<type, 2> hessian(parameters_number, parameters_number);

       const type norm = l2_norm(parameters);

       if(static_cast<Index>(norm) == 0.0)
       {
           hessian.setZero();

           return hessian;
       }

       switch(device_pointer->get_type())
       {
            case Device::EigenDefault:
            {
                DefaultDevice* default_device = device_pointer->get_eigen_default_device();

                hessian.device(*default_device) = kronecker_product(parameters, parameters)/(norm*norm*norm);

                return hessian;
            }

            case Device::EigenSimpleThreadPool:
            {
               ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

               hessian.device(*thread_pool_device) = kronecker_product(parameters, parameters)/(norm*norm*norm);

               return hessian;

            }

           case Device::EigenGpu:
           {
//                GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

               return hessian;
           }
       }

       return Tensor<type, 2>();
   }

protected:

   Device* device_pointer = nullptr;

   /// Pointer to a neural network object.

   NeuralNetwork* neural_network_pointer = nullptr;

   /// Pointer to a data set object.

   DataSet* data_set_pointer = nullptr;

   /// Pointer to a regularization method object.

   RegularizationMethod regularization_method = L2;

   /// Regularization weight value.

   type regularization_weight;

   /// Display messages to screen. 

   bool display = true;

   const Eigen::array<IndexPair<Index>, 1> AT_B = {IndexPair<Index>(0, 0)};

   const Eigen::array<IndexPair<Index>, 2> SSE = {IndexPair<Index>(0, 0), IndexPair<Index>(1, 1)};
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
