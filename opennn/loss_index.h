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

namespace opennn
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

   /// Enumeration of the available regularization methods.

   enum class RegularizationMethod{L1, L2, NoRegularization};

   /// Returns a pointer to the neural network object associated with the error term.

   inline NeuralNetwork* get_neural_network_pointer() const 
   {
        #ifdef OPENNN_DEBUG

        if(!neural_network_pointer)
        {
             ostringstream buffer;

             buffer << "OpenNN Exception: LossIndex class.\n"
                    << "NeuralNetwork* get_neural_network_pointer() const method.\n"
                    << "Neural network pointer is nullptr.\n";

             throw invalid_argument(buffer.str());
        }

        #endif

      return neural_network_pointer;
   }

   /// Returns a pointer to the data set object associated with the error term.

   inline DataSet* get_data_set_pointer() const 
   {
        #ifdef OPENNN_DEBUG

        if(!data_set_pointer)
        {
             ostringstream buffer;

             buffer << "OpenNN Exception: LossIndex class.\n"
                    << "DataSet* get_data_set_pointer() const method.\n"
                    << "DataSet pointer is nullptr.\n";

             throw invalid_argument(buffer.str());
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

   virtual void set_normalization_coefficient() {}

   bool has_selection() const;

   // Numerical differentiation

   type calculate_eta() const;
   type calculate_h(const type&) const;

   Tensor<type, 1> calculate_gradient_numerical_differentiation();

   Tensor<type, 2> calculate_jacobian_numerical_differentiation();

   // Back propagation

   void calculate_errors(const DataSetBatch&,
                         const NeuralNetworkForwardPropagation&,
                         LossIndexBackPropagation&) const;

   virtual void calculate_error(const DataSetBatch&,
                                const NeuralNetworkForwardPropagation&,
                                LossIndexBackPropagation&) const = 0;

   virtual void calculate_output_delta(const DataSetBatch&,
                                       NeuralNetworkForwardPropagation&,
                                       LossIndexBackPropagation&) const = 0;

   void calculate_layers_delta(const DataSetBatch&,
                               NeuralNetworkForwardPropagation&,
                               LossIndexBackPropagation&) const;

   void calculate_layers_error_gradient(const DataSetBatch&,
                                 const NeuralNetworkForwardPropagation&,
                                 LossIndexBackPropagation&) const;

   void assemble_layers_error_gradient(LossIndexBackPropagation&) const;

   void back_propagate(const DataSetBatch&,
                       NeuralNetworkForwardPropagation&,
                       LossIndexBackPropagation&) const;

   // Back propagation LM

   void calculate_errors_lm(const DataSetBatch&,
                            const NeuralNetworkForwardPropagation&,
                            LossIndexBackPropagationLM&) const; // general

   virtual void calculate_squared_errors_lm(const DataSetBatch&,
                                            const NeuralNetworkForwardPropagation&,
                                            LossIndexBackPropagationLM&) const;

   virtual void calculate_error_lm(const DataSetBatch&,
                                   const NeuralNetworkForwardPropagation&,
                                   LossIndexBackPropagationLM&) const {}

   virtual void calculate_output_delta_lm(const DataSetBatch&,
                                          NeuralNetworkForwardPropagation&,
                                          LossIndexBackPropagationLM&) const {}

   void calculate_layers_delta_lm(const DataSetBatch&,
                                  NeuralNetworkForwardPropagation&,
                                  LossIndexBackPropagationLM&) const;

   virtual void calculate_error_gradient_lm(const DataSetBatch&,
                                      LossIndexBackPropagationLM&) const;

   void calculate_squared_errors_jacobian_lm(const DataSetBatch&,
                                             NeuralNetworkForwardPropagation&,
                                             LossIndexBackPropagationLM&) const;

   virtual void calculate_error_hessian_lm(const DataSetBatch&,
                                           LossIndexBackPropagationLM&) const {}

   void back_propagate_lm(const DataSetBatch&,
                          NeuralNetworkForwardPropagation&,
                          LossIndexBackPropagationLM&) const;

   // Regularization methods

   void add_regularization(LossIndexBackPropagation&) const;
   void add_regularization_gradient(LossIndexBackPropagation&) const;

   type calculate_regularization() const;

   type calculate_regularization(const Tensor<type, 1>&) const;

   void calculate_regularization_gradient(const Tensor<type, 1>&, Tensor<type, 1>&) const;
   void calculate_regularization_hessian(const Tensor<type, 1>&, Tensor<type, 2>&) const;

   // Serialization methods

   void from_XML(const tinyxml2::XMLDocument&);

   virtual void write_XML(tinyxml2::XMLPrinter&) const;

   void regularization_from_XML(const tinyxml2::XMLDocument&);
   void write_regularization_XML(tinyxml2::XMLPrinter&) const;

   virtual string get_error_type() const;
   virtual string get_error_type_text() const;

   string write_regularization_method() const;

   // Checking methods

   void check() const;

protected:

   ThreadPool* thread_pool = nullptr;
   ThreadPoolDevice* thread_pool_device = nullptr;

   /// Pointer to a neural network object.

   NeuralNetwork* neural_network_pointer = nullptr;

   /// Pointer to a data set object.

   DataSet* data_set_pointer = nullptr;

   /// Pointer to a regularization method object.

   RegularizationMethod regularization_method = RegularizationMethod::L2;

   /// Regularization weight value.

   type regularization_weight = static_cast<type>(0.01);

   /// Display messages to screen. 

   bool display = true;

   const Eigen::array<IndexPair<Index>, 1> AT_B = {IndexPair<Index>(0, 0)};
   const Eigen::array<IndexPair<Index>, 1> A_B = {IndexPair<Index>(1, 0)};

   const Eigen::array<IndexPair<Index>, 2> SSE = {IndexPair<Index>(0, 0), IndexPair<Index>(1, 1)};

   const Eigen::array<int, 1> rows_sum = {Eigen::array<int, 1>({1})};

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/loss_index_cuda.h"
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
        set(new_batch_samples_number, new_loss_index_pointer);
    }

    virtual ~LossIndexBackPropagation();

    void set(const Index& new_batch_samples_number, LossIndex* new_loss_index_pointer)
    {
        loss_index_pointer = new_loss_index_pointer;

        batch_samples_number = new_batch_samples_number;

        // Neural network

        NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

        const Index parameters_number = neural_network_pointer->get_parameters_number();

        const Index outputs_number = neural_network_pointer->get_outputs_number();

        // First order loss

        neural_network.set(batch_samples_number, neural_network_pointer);

        error = type(0);

        loss = type(0);

        errors.resize(batch_samples_number, outputs_number);

        if(assemble)
        {
            parameters = neural_network_pointer->get_parameters();

            gradient.resize(parameters_number);
        }
    }

    void print() const
    {
        cout << "Loss index back-propagation" << endl;

        cout << "Errors:" << endl;
        cout << errors << endl;

        cout << "Error:" << endl;
        cout << error << endl;

        cout << "Regularization:" << endl;
        cout << regularization << endl;

        cout << "Loss:" << endl;
        cout << loss << endl;

        cout << "Gradient:" << endl;
        cout << gradient << endl;

        neural_network.print();
    }

    Tensor< Tensor< TensorMap< Tensor<type, 1> >*, 1>, 1> get_layers_gradient()
    {
        NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

        const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();
        const Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

        Tensor< Tensor< TensorMap< Tensor<type, 1> >*, 1>, 1> layers_gradient(trainable_layers_number);

        for(Index i = 0; i < trainable_layers_number; i++)
        {
            layers_gradient(i) = neural_network.layers(i)->get_layer_gradient();
        }

        return layers_gradient;
    }

    Index batch_samples_number = 0;

    LossIndex* loss_index_pointer = nullptr;

    NeuralNetworkBackPropagation neural_network;

    type error = type(0);
    type regularization = type(0);
    type loss = type(0);

    Tensor<type, 2> errors;

    Tensor<type, 1> parameters;

    Tensor< Tensor< TensorMap<Tensor<type, 1> >*, 1>, 1> layers_gradient;

    Tensor<type, 1> gradient;

    bool assemble = true;
};


/// A loss index composed of several terms, this structure represent the First Order for this function.

/// This structure contains second-order information about the loss function (loss, gradient and Hessian).
/// Set of loss value, gradient vector and <i>Hessian</i> matrix of the loss index.
/// A method returning this structure might be implemented more efficiently than the loss,
/// gradient and <i>Hessian</i> methods separately.

struct LossIndexBackPropagationLM
{
    /// Default constructor.

    LossIndexBackPropagationLM() {}

    explicit LossIndexBackPropagationLM(const Index& new_batch_samples_number, LossIndex* new_loss_index_pointer)
    {
        set(new_batch_samples_number, new_loss_index_pointer);
    }

    void set(const Index& new_batch_samples_number, LossIndex* new_loss_index_pointer)
    {
        loss_index_pointer = new_loss_index_pointer;

        batch_samples_number = new_batch_samples_number;

        NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

        const Index parameters_number = neural_network_pointer->get_parameters_number();

        const Index outputs_number = neural_network_pointer->get_outputs_number();

        neural_network.set(batch_samples_number, neural_network_pointer);

        parameters = neural_network_pointer->get_parameters();

        error = type(0);

        loss = type(0);

        gradient.resize(parameters_number);

        regularization_gradient.resize(parameters_number);
        regularization_gradient.setZero();

        squared_errors_jacobian.resize(batch_samples_number, parameters_number);

        hessian.resize(parameters_number, parameters_number);

        regularization_hessian.resize(parameters_number, parameters_number);
        regularization_hessian.setZero();

        errors.resize(batch_samples_number, outputs_number);

        squared_errors.resize(batch_samples_number);
    }

    void print() const
    {
        cout << "Loss index back-propagation LM" << endl;

        cout << "Errors:" << endl;
        cout << errors << endl;

        cout << "Squared errors:" << endl;
        cout << squared_errors << endl;

        cout << "Squared errors Jacobian:" << endl;
        cout << squared_errors_jacobian << endl;

        cout << "Error:" << endl;
        cout << error << endl;

        cout << "Loss:" << endl;
        cout << loss << endl;

        cout << "Gradient:" << endl;
        cout << gradient << endl;

        cout << "Hessian:" << endl;
        cout << hessian << endl;
    }

    Index batch_samples_number = 0;

    LossIndex* loss_index_pointer = nullptr;

    type error = type(0);
    type regularization = type(0);
    type loss = type(0);

    Tensor<type, 1> parameters;

    NeuralNetworkBackPropagationLM neural_network;

    Tensor<type, 2> errors;
    Tensor<type, 1> squared_errors;
    Tensor<type, 2> squared_errors_jacobian;

    Tensor<type, 1> gradient;
    Tensor<type, 2> hessian;

    Tensor<type, 1> regularization_gradient;
    Tensor<type, 2> regularization_hessian;
};


}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
