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

#include "data_set.h"
#include "neural_network.h"
#include "numerical_differentiation.h"

namespace OpenNN
{

struct LossIndexBackPropagation;
struct LossIndexBackPropagationLM;

/// This abstract class represents the concept of loss index composed of an error term and a regularization term.

///
/// The error term could be:
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

   explicit LossIndex(NeuralNetwork*, DataSet*);

   // Destructor

   virtual ~LossIndex();

   // Methods

   /// Enumeration of available regularization methods.

   enum RegularizationMethod{L1, L2, NoRegularization};

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

   void set_threads_number(const int&);

   void set_neural_network_pointer(NeuralNetwork*);

   virtual void set_data_set_pointer(DataSet*);

   void set_default();

   void set_regularization_method(const RegularizationMethod&);
   void set_regularization_method(const string&);
   void set_regularization_weight(const type&);

   void set_display(const bool&);

   bool has_selection() const;

   // GRADIENT METHODS

   virtual void calculate_output_delta(const DataSetBatch&,
                                       NeuralNetworkForwardPropagation&,
                                       LossIndexBackPropagation&) const = 0;

   // Numerical differentiation

   type calculate_eta() const;
   type calculate_h(const type&) const;

   Tensor<type, 1> calculate_gradient_numerical_differentiation(LossIndex*) const;

   Tensor<type, 2> calculate_Jacobian_numerical_differentiation(LossIndex*) const;


   void calculate_errors(const DataSetBatch&,
                         const NeuralNetworkForwardPropagation&,
                         LossIndexBackPropagation&) const;

   void calculate_squared_errors(const DataSetBatch&,
                                 const NeuralNetworkForwardPropagation&,
                                 LossIndexBackPropagationLM&) const;

   virtual void calculate_error(const DataSetBatch&,
                                const NeuralNetworkForwardPropagation&,
                                LossIndexBackPropagation&) const = 0;

   virtual void calculate_error(const DataSetBatch&,
                                const NeuralNetworkForwardPropagation&,
                                LossIndexBackPropagationLM&) const {}

   void back_propagate(const DataSetBatch&,
                       NeuralNetworkForwardPropagation&,
                       LossIndexBackPropagation&) const;

   void back_propagate(const DataSetBatch&,
                       NeuralNetworkForwardPropagation&,
                       LossIndexBackPropagationLM&) const;

   void calculate_error_terms_jacobian(const DataSetBatch&,
                                       NeuralNetworkForwardPropagation&,
                                       LossIndexBackPropagation&,
                                       LossIndexBackPropagationLM&) const;

   virtual void calculate_gradient(const DataSetBatch&,
                                   LossIndexBackPropagationLM&) const {}

   virtual void calculate_hessian_approximation(const DataSetBatch&,
                                                LossIndexBackPropagationLM&) const {}

   // Regularization methods

   type calculate_regularization(const Tensor<type, 1>&) const;

   void calculate_regularization_gradient(const Tensor<type, 1>&, Tensor<type, 1>&) const;
   void calculate_regularization_hessian(const Tensor<type, 1>&, Tensor<type, 2>&) const;

   // Delta methods

   void calculate_layers_delta(const DataSetBatch&,
                               NeuralNetworkForwardPropagation&,
                               LossIndexBackPropagation&) const;

   void calculate_error_gradient(const DataSetBatch&,
                                 const NeuralNetworkForwardPropagation&,
                                 LossIndexBackPropagation&) const;

   // Serialization methods

   void from_XML(const tinyxml2::XMLDocument&);

   virtual void write_XML(tinyxml2::XMLPrinter&) const;

   void regularization_from_XML(const tinyxml2::XMLDocument&);
   void write_regularization_XML(tinyxml2::XMLPrinter&) const;

   string get_error_type() const;
   virtual string get_error_type_text() const;

   string write_regularization_method() const;

   // Checking methods

   void check() const;

   // Metrics

   Tensor<type, 2> kronecker_product(const Tensor<type, 1>&, const Tensor<type, 1>&) const;

   type l1_norm(const Tensor<type, 1>& parameters) const;
   void l1_norm_gradient(const Tensor<type, 1>&, Tensor<type, 1>&) const;
   void l1_norm_hessian(const Tensor<type, 1>&, Tensor<type, 2>&) const;

   type l2_norm(const Tensor<type, 1>& parameters) const;
   void l2_norm_gradient(const Tensor<type, 1>&, Tensor<type, 1>&) const;
   void l2_norm_hessian(const Tensor<type, 1>&, Tensor<type, 2>&) const;

protected:

   NonBlockingThreadPool* non_blocking_thread_pool = nullptr;
   ThreadPoolDevice* thread_pool_device = nullptr;

   /// Pointer to a neural network object.

   NeuralNetwork* neural_network_pointer = nullptr;

   /// Pointer to a data set object.

   DataSet* data_set_pointer = nullptr;

   /// Pointer to a regularization method object.

   RegularizationMethod regularization_method = L2;

   /// Regularization weight value.

   type regularization_weight = static_cast<type>(0.01);

   /// Display messages to screen. 

   bool display = true;

   const Eigen::array<IndexPair<Index>, 1> AT_B = {IndexPair<Index>(0, 0)};
   const Eigen::array<IndexPair<Index>, 1> A_B = {IndexPair<Index>(1, 0)};

   const Eigen::array<IndexPair<Index>, 2> SSE = {IndexPair<Index>(0, 0), IndexPair<Index>(1, 1)};

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn_cuda/loss_index_cuda.h"
#endif

#ifdef OPENNN_MKL
    #include "../../opennn-mkl/opennn_mkl/loss_index_mkl.h"
#endif

};


/// Set of loss value and gradient vector of the loss index.
/// A method returning this structure might be implemented more efficiently than the loss and gradient methods separately.

struct LossIndexBackPropagation
{
    /// Default constructor.

    explicit LossIndexBackPropagation() {}

    explicit LossIndexBackPropagation(const Index& new_batch_samples_number, LossIndex* new_loss_index_pointer)
    {
        if (new_batch_samples_number == 0) return;

        set(new_batch_samples_number, new_loss_index_pointer);
    }

    virtual ~LossIndexBackPropagation();

    void set(const Index& new_batch_samples_number, LossIndex* new_loss_index_pointer)
    {
        batch_samples_number = new_batch_samples_number;

        loss_index_pointer = new_loss_index_pointer;

        // Neural network

        NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

        const Index parameters_number = neural_network_pointer->get_parameters_number();

        const Index outputs_number = neural_network_pointer->get_outputs_number();

        // First order loss

        neural_network.set(batch_samples_number, neural_network_pointer);

        error = 0;

        loss = 0;

        errors.resize(batch_samples_number, outputs_number);

        parameters = neural_network_pointer->get_parameters();

        gradient.resize(parameters_number);

        regularization_gradient.resize(parameters_number);
        regularization_gradient.setConstant(0);
    }


    void print()
    {
        cout << "Error:" << endl;
        cout << error << endl;

        cout << "Loss:" << endl;
        cout << loss << endl;

        cout << "Gradient:" << endl;
        cout << gradient << endl;
    }

    LossIndex* loss_index_pointer = nullptr;

    Index batch_samples_number = 0;

    NeuralNetworkBackPropagation neural_network;

    type error;

    type loss;

    Tensor<type, 2> errors;

    Tensor<type, 1> parameters;

    Tensor<type, 1> gradient;

    Tensor<type, 1> regularization_gradient;
};


/// A loss index composed of several terms, this structure represent the First Order for this function.

/// This structure contains second order information about the loss function (loss, gradient and Hessian).
/// Set of loss value, gradient vector and <i>Hessian</i> matrix of the loss index.
/// A method returning this structure might be implemented more efficiently than the loss,
/// gradient and <i>Hessian</i> methods separately.

struct LossIndexBackPropagationLM
{
    /// Default constructor.

    LossIndexBackPropagationLM() {}

    LossIndexBackPropagationLM(const Index& parameters_number, const Index& samples_number)
    {
        loss = 0;
        gradient.resize(parameters_number);
        squared_errors_Jacobian.resize(samples_number, parameters_number);
        hessian.resize(parameters_number, parameters_number);
        squared_errors.resize(samples_number);
    }

    void sum_hessian_diagonal(const type& value)
    {
        const Index parameters_number = hessian.dimension(0);

         #pragma omp parallel for

        for(Index i = 0; i < parameters_number; i++)
            hessian(i,i) += value;
    }

    type error;
    type loss;

    Tensor<type, 1> parameters;

    Tensor<type, 1> squared_errors;
    Tensor<type, 2> squared_errors_Jacobian;

    Tensor<type, 1> gradient;
    Tensor<type, 2> hessian;
};


}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2021 Artificial Intelligence Techniques, SL.
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
