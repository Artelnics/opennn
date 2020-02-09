//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E A N   S Q U A R E D   E R R O R    C L A S S   H E A D E R        
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef MEANSQUAREDERROR_H
#define MEANSQUAREDERROR_H

// System includes

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <limits>
#include <math.h>

// OpenNN includes

#include "config.h"
#include "loss_index.h"
#include "data_set.h"
#include "tinyxml2.h"

namespace OpenNN
{

/// This class represents the mean squared error term.

///
/// The mean squared error measures the difference between the outputs from a neural network and the targets in a data set. 
/// This functional is used in data modeling problems, such as function regression, 
/// classification and time series prediction.

class MeanSquaredError : public LossIndex
{

public:

   // DEFAULT CONSTRUCTOR

   explicit MeanSquaredError();

   // NEURAL NETWORK CONSTRUCTOR

   explicit MeanSquaredError(NeuralNetwork*);

   // DATA SET CONSTRUCTOR

   explicit MeanSquaredError(DataSet*);
   
   explicit MeanSquaredError(NeuralNetwork*, DataSet*);

   explicit MeanSquaredError(const tinyxml2::XMLDocument&);

   // COPY CONSTRUCTOR

   MeanSquaredError(const MeanSquaredError&);

   // Destructor

   virtual ~MeanSquaredError(); 

   // Error methods

   type calculate_training_error() const;
   type calculate_training_error(const Tensor<type, 1>&) const;

   type calculate_selection_error() const;

   type calculate_batch_error(const Tensor<Index, 1>&) const;
   type calculate_batch_error(const Tensor<Index, 1>&, const Tensor<type, 1>&) const;

   // Gradient methods

   BackPropagation calculate_back_propagation() const;



   // Error terms methods

   Tensor<type, 1> calculate_training_error_terms(const Tensor<type, 2>&, const Tensor<type, 2>&) const;
   Tensor<type, 1> calculate_training_error_terms(const Tensor<type, 1>&) const;

   string get_error_type() const;
   string get_error_type_text() const;


   void calculate_error(BackPropagation& back_propagation) const
   {
       Tensor<type, 0> sum_squared_error;

       switch(device_pointer->get_type())
       {
            case Device::EigenDefault:
            {
                DefaultDevice* default_device = device_pointer->get_eigen_default_device();

                sum_squared_error.device(*default_device) = back_propagation.errors.square().sum();

                break;
            }

            case Device::EigenSimpleThreadPool:
            {
               ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

               sum_squared_error.device(*thread_pool_device) = back_propagation.errors.square().sum();

                break;
            }

           case Device::EigenGpu:
           {
//                GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

                break;
           }

            default:
            {
               ostringstream buffer;

               buffer << "OpenNN Exception: Layer class.\n"
                      << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                      << "Unknown device.\n";

               throw logic_error(buffer.str());
           }
       }

       const Index batch_instances_number = back_propagation.errors.dimension(0);

       back_propagation.loss = sum_squared_error(0)/static_cast<type>(batch_instances_number);
   }


   void calculate_output_gradient(const NeuralNetwork::ForwardPropagation& forward_propagation,
                                  BackPropagation& back_propagation) const
   {
        #ifdef __OPENNN_DEBUG__

        check();

        #endif

        const Index instances_number = data_set_pointer->get_training_instances_number();

        const type coefficient = static_cast<type>(2.0)/static_cast<type>(instances_number);

        switch(device_pointer->get_type())
        {
             case Device::EigenDefault:
             {
                 DefaultDevice* default_device = device_pointer->get_eigen_default_device();

                 back_propagation.output_gradient.device(*default_device) = coefficient*back_propagation.errors;

                 return;
             }

             case Device::EigenSimpleThreadPool:
             {
                ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

                back_propagation.output_gradient.device(*thread_pool_device) = coefficient*back_propagation.errors;

                return;
             }

            case Device::EigenGpu:
            {
//                 GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

                 break;
            }

             default:
             {
                ostringstream buffer;

                buffer << "OpenNN Exception: Layer class.\n"
                       << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                       << "Unknown device.\n";

                throw logic_error(buffer.str());
            }
        }
   }

   LossIndex::SecondOrderLoss calculate_terms_second_order_loss() const;

   type sum_squared_error(const Tensor<type, 2>& ,const Tensor<type, 2>&) const;

   // Serialization methods

   tinyxml2::XMLDocument* to_XML() const;   

   void write_XML(tinyxml2::XMLPrinter &) const;
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
