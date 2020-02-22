//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M I N K O W S K I   E R R O R   C L A S S   H E A D E R               
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef MINKOWSKIERROR_H
#define MINKOWSKIERROR_H

// System includes

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cmath>

// OpenNN includes

#include "config.h"
#include "loss_index.h"
#include "data_set.h"
#include "tinyxml2.h"

namespace OpenNN
{

/// This class represents the Minkowski error term. 

///
/// The Minkowski error measures the difference between the outputs of a neural network and the targets in a data set. 
/// This error term is used in data modeling problems.
/// It can be more useful when the data set presents outliers. 

class MinkowskiError : public LossIndex
{

public:

   // Constructors

   explicit MinkowskiError();

   explicit MinkowskiError(NeuralNetwork*);

   explicit MinkowskiError(DataSet*);

   explicit MinkowskiError(NeuralNetwork*, DataSet*);

   explicit MinkowskiError(const tinyxml2::XMLDocument&);

   // Destructor

   virtual ~MinkowskiError();

   // Get methods

   type get_Minkowski_parameter() const;

   // Set methods

   void set_default();

   void set_Minkowski_parameter(const type&);

   // loss methods

   /// @todo Check formula

   type calculate_error(const DataSet::Batch& batch, const NeuralNetwork::ForwardPropagation& forward_propagation) const
   {
       Tensor<type, 0> minkowski_error;

       const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

       const Tensor<type, 2>& outputs = forward_propagation.layers[trainable_layers_number-1].activations_2d;

       const Tensor<type, 2>& targets = batch.targets_2d;

       switch(device_pointer->get_type())
       {
            case Device::EigenDefault:
            {
                DefaultDevice* default_device = device_pointer->get_eigen_default_device();

                minkowski_error.device(*default_device) = (outputs - targets).abs().pow(minkowski_parameter).sum()
                                                            .pow(static_cast<type>(1.0)/minkowski_parameter);

                break;
            }

            case Device::EigenSimpleThreadPool:
            {
               ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

               minkowski_error.device(*thread_pool_device) = (outputs - targets).abs().pow(minkowski_parameter).sum()
                                                                .pow(static_cast<type>(1.0)/minkowski_parameter);

                break;
            }

           case Device::EigenGpu:
           {
//                GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

                break;
           }
       }

       return minkowski_error(0);
   }

   void calculate_error(BackPropagation& back_propagation) const
   {
       Tensor<type, 0> minkowski_error;

       switch(device_pointer->get_type())
       {
            case Device::EigenDefault:
            {
                DefaultDevice* default_device = device_pointer->get_eigen_default_device();

//                minkowski_error.device(*default_device) = ((back_propagation.errors.abs().pow(minkowski_parameter)).sum())
//                                                            .pow(static_cast<type>(1.0)/minkowski_parameter);

                break;
            }

            case Device::EigenSimpleThreadPool:
            {
               ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

//               minkowski_error.device(*thread_pool_device) = ((back_propagation.errors.abs().pow(minkowski_parameter)).sum())
//                                                               .pow(static_cast<type>(1.0)/minkowski_parameter);

                break;
            }

           case Device::EigenGpu:
           {
//                GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

                break;
           }
       }

       back_propagation.loss = minkowski_error(0);
   }

   /// @todo Virtual method not implemented.

   void calculate_output_gradient(const DataSet::Batch& batch,
                                  const NeuralNetwork::ForwardPropagation& forward_propagation,
                                  BackPropagation& back_propagation) const
   {
        #ifdef __OPENNN_DEBUG__

        check();

        #endif

        const Index training_instances_number = data_set_pointer->get_training_instances_number();

        const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

//        back_propagation.output_gradient = lp_norm_gradient(forward_propagation.layers[trainable_layers_number].activations_2d
//                                           - batch.targets_2d, minkowski_parameter)/static_cast<type>(training_instances_number);

        switch(device_pointer->get_type())
        {
             case Device::EigenDefault:
             {
                 DefaultDevice* default_device = device_pointer->get_eigen_default_device();

                 Tensor<type, 0> p_norm = back_propagation.errors.abs().pow(minkowski_parameter).sum().pow(1.0/minkowski_parameter);

                 back_propagation.output_gradient.device(*default_device)
                         = back_propagation.errors.abs().pow(minkowski_parameter-2.0)/p_norm;

                 break;
             }

             case Device::EigenSimpleThreadPool:
             {
                ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

                 break;
             }

            case Device::EigenGpu:
            {
 //                GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

                 break;
            }
        }
   }

   string get_error_type() const;
   string get_error_type_text() const;

   // Serialization methods

   tinyxml2::XMLDocument* to_XML() const;   
   void from_XML(const tinyxml2::XMLDocument&);   

   void write_XML(tinyxml2::XMLPrinter&) const;

private:

   /// Minkowski exponent value.

   type minkowski_parameter;

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
