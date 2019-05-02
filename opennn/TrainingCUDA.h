/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   T R A I N I N G   C U D A    C L A S S   H E A D E R                                                       */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __TRAININGCUDA_H__
#define __TRAININGCUDA_H__

// System includes

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <ctime>

//#ifdef __OPENNN_CUDA__
//#include <cublas_v2.h>
//#endif

// OpenNN includes

#include "vector.h"

#include "data_set.h"
#include "multilayer_perceptron.h"

// TinyXml includes

#include "tinyxml2.h"

namespace OpenNN
{

///
/// This concrete class represents a quasi-Newton training algorithm for a loss index of a neural network.
///

class TrainingCUDA
{

public:

    // DEFAULT CONSTRUCTOR

    explicit TrainingCUDA();

    // LOSS INDEX CONSTRUCTOR

    explicit TrainingCUDA(DataSet*);

    // DESTRUCTOR

    virtual ~TrainingCUDA();

    // Getters

    Matrix<double> get_weights_host(const size_t&) const;
    Vector<double> get_biases_host(const size_t&) const;

    Vector<double> get_parameters_host() const;

    // Setters

    void set_neural_network_architecture(const Vector<size_t>&);

    // CUDA initialization methods

    void initialize_CUDA(void);

    void randomize_parameters(void);

    // Operation methods

    Matrix<double> calculate_outputs(const Matrix<double>&);

    MultilayerPerceptron::FirstOrderForwardPropagation calculate_first_order_forward_propagation(const Matrix<double>&);

    Vector<double> calculate_batch_error_gradient(const Vector<size_t>&);

    void update_parameters(const Vector<double>&);

private: 

    bool CUDA_initialized = false;

    DataSet* data_set_pointer;

    Vector<size_t> neural_network_architecture;

    Vector<string> layer_activations;

    Vector<double*> weights_gpu;
    Vector<double*> biases_gpu;

    string loss_method;
};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
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
