//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef CROSSENTROPYERROR_H
#define CROSSENTROPYERROR_H

// System includes

#include <iostream>
#include <fstream>
#include <math.h>

// OpenNN includes

#include "loss_index.h"
#include "data_set.h"
#include "config.h"

#include "tinyxml2.h"

namespace OpenNN
{

/// This class represents the cross entropy error term, used for predicting probabilities.

///
/// This functional is used in classification problems.

class CrossEntropyError : public LossIndex
{

public:

   // Constructors

   explicit CrossEntropyError();

   explicit CrossEntropyError(NeuralNetwork*);

   explicit CrossEntropyError(DataSet*);

   explicit CrossEntropyError(NeuralNetwork*, DataSet*);

   explicit CrossEntropyError(const tinyxml2::XMLDocument&);

   CrossEntropyError(const CrossEntropyError&);

   // Destructor

   virtual ~CrossEntropyError();

   // Error methods

   void calculate_error(const DataSet::Batch& batch,
                        const NeuralNetwork::ForwardPropagation& forward_propagation,
                        LossIndex::BackPropagation& back_propagation) const
   {
       const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

       const Tensor<type, 2>& outputs = forward_propagation.layers[trainable_layers_number-1].activations_2d;
       const Tensor<type, 2>& targets = batch.targets_2d;

       const Index rows_number = outputs.dimension(0);
       const Index columns_number = outputs.dimension(1);

       type cross_entropy_error = 0.0;

       for(Index i = 0; i < rows_number; i++)
       {
           for(Index j = 0; j < columns_number; j++)
           {
               const type target = targets(i,j);
               const type output = outputs(i,j);

               if(abs(target) < numeric_limits<type>::min() && abs(output) < numeric_limits<type>::min())
               {
                   // Do nothing
               }
               else if(abs(target - 1) < numeric_limits<type>::min()
                    && abs(output - 1) < numeric_limits<type>::min())
               {
                   // Do nothing
               }
               else if(abs(output) < numeric_limits<type>::min())
               {
                   cross_entropy_error -= (1 - target)*log(1-output) + target*log(static_cast<type>(0.000000001));
               }
               else if(abs(output - 1) < numeric_limits<type>::min())
               {
                   cross_entropy_error -= (1 - target)*log(1-output) + target*log(static_cast<type>(0.999999999));
               }
               else
               {
                   cross_entropy_error -= (1 - target)*log(1-output) + target*log(output);
               }
           }
       }

       back_propagation.error = cross_entropy_error;

       return;
  }

   // Gradient methods

   void calculate_output_gradient(const DataSet::Batch& batch,
                                  const NeuralNetwork::ForwardPropagation& forward_propagation,
                                  BackPropagation& back_propagation) const
   {
        #ifdef __OPENNN_DEBUG__

        check();

        #endif

        const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

        const Tensor<type, 2>& targets = batch.targets_2d;
        const Tensor<type, 2>& outputs = forward_propagation.layers[trainable_layers_number-1].activations_2d;

        switch(device_pointer->get_type())
        {
             case Device::EigenDefault:
             {
                 DefaultDevice* default_device = device_pointer->get_eigen_default_device();

                 back_propagation.output_gradient.device(*default_device) =
                         -1.0*(targets/outputs) + (1.0 - targets)/(1.0 - outputs);

                 return;
             }

             case Device::EigenThreadPool:
             {
                ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

                back_propagation.output_gradient.device(*thread_pool_device) =
                        -1.0*(targets/outputs) + (1.0 - targets)/(1.0 - outputs);

                return;
             }
        }
   }


   string get_error_type() const;
   string get_error_type_text() const;

   // Serialization methods

   tinyxml2::XMLDocument* to_XML() const;   
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

#ifdef OPENNN_CUDA
    #include "../../artelnics/opennn_cuda/opennn_cuda/cross_entropy_error_cuda.h"
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
