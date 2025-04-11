//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   I N D E X   C L A S S   H E A D E R                         
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LOSSINDEX_H
#define LOSSINDEX_H

#include "data_set.h"
#include "neural_network.h"
#include "batch.h"
#include "neural_network_back_propagation_lm.h"

namespace opennn
{

struct BackPropagation;
struct BackPropagationLM;

#ifdef OPENNN_CUDA_test
struct BackPropagationCuda;
#endif


class LossIndex
{

public:

   LossIndex(NeuralNetwork* = nullptr, DataSet* = nullptr);

   enum class RegularizationMethod{L1, L2, NoRegularization};

   inline NeuralNetwork* get_neural_network() const 
   {
      return neural_network;
   }

   inline DataSet* get_data_set() const 
   {
      return data_set;
   }

   const type& get_regularization_weight() const;

   const bool& get_display() const;

   bool has_neural_network() const;

   bool has_data_set() const;

   RegularizationMethod get_regularization_method() const;

   void set(NeuralNetwork* = nullptr, DataSet* = nullptr);

   void set_threads_number(const int&);

   void set_neural_network(NeuralNetwork*);

   virtual void set_data_set(DataSet*);

   void set_regularization_method(const RegularizationMethod&);
   void set_regularization_method(const string&);
   void set_regularization_weight(const type&);

   void set_display(const bool&);

   virtual void set_normalization_coefficient() {}

   // Back propagation

   virtual void calculate_error(const Batch&,
                                const ForwardPropagation&,
                                BackPropagation&) const = 0;

   void add_regularization(BackPropagation&) const;
   void add_regularization_lm(BackPropagationLM&) const;

   virtual void calculate_output_delta(const Batch&,
                                       ForwardPropagation&,
                                       BackPropagation&) const = 0;

   void calculate_layers_error_gradient(const Batch&,
                                        ForwardPropagation&,
                                        BackPropagation&) const;

   void assemble_layers_error_gradient(BackPropagation&) const;

   void back_propagate(const Batch&,
                       ForwardPropagation&,
                       BackPropagation&) const;

   // Back propagation LM

   void calculate_errors_lm(const Batch&,
                            const ForwardPropagation&,
                            BackPropagationLM&) const; 

   virtual void calculate_squared_errors_lm(const Batch&,
                                            const ForwardPropagation&,
                                            BackPropagationLM&) const;

   virtual void calculate_error_lm(const Batch&,
                                   const ForwardPropagation&,
                                   BackPropagationLM&) const {}

   virtual void calculate_output_delta_lm(const Batch&,
                                          ForwardPropagation&,
                                          BackPropagationLM&) const {}

   void calculate_layers_squared_errors_jacobian_lm(const Batch&,
                                                    ForwardPropagation&,
                                                    BackPropagationLM&) const;

   virtual void calculate_error_gradient_lm(const Batch&,
                                      BackPropagationLM&) const;

   virtual void calculate_error_hessian_lm(const Batch&,
                                           BackPropagationLM&) const {}

   void back_propagate_lm(const Batch&,
                          ForwardPropagation&,
                          BackPropagationLM&) const;

   // Regularization

   type calculate_regularization(const Tensor<type, 1>&) const;

   void calculate_regularization_gradient(const Tensor<type, 1>&, Tensor<type, 1>&) const;
   void calculate_regularization_hessian(Tensor<type, 1>&, Tensor<type, 2>&) const;

   // Serialization

   void from_XML(const XMLDocument&);

   virtual void to_XML(XMLPrinter&) const;

   void regularization_from_XML(const XMLDocument&);
   void write_regularization_XML(XMLPrinter&) const;

   virtual string get_loss_method() const;
   virtual string get_error_type_text() const;

   string write_regularization_method() const;

   // Checking

//   void check() const;

   // Numerical differentiation

   static type calculate_h(const type&);

   Tensor<type, 1> calculate_numerical_gradient();
   Tensor<type, 1> calculate_numerical_gradient_lm();
   Tensor<type, 2> calculate_numerical_jacobian();
   Tensor<type, 1> calculate_numerical_inputs_derivatives();
   Tensor<type, 2> calculate_numerical_hessian();
   Tensor<type, 2> calculate_inverse_hessian();

#ifdef OPENNN_CUDA_test

public:

    void create_cuda();
    void destroy_cuda();

    cudnnHandle_t get_cudnn_handle();

    virtual void calculate_error_cuda(const BatchCuda&,
                                      const ForwardPropagationCuda&,
                                      BackPropagationCuda&) const {}

    virtual void calculate_output_delta_cuda(const BatchCuda&,
                                             ForwardPropagationCuda&,
                                             BackPropagationCuda&) const {}

    void calculate_layers_error_gradient_cuda(const BatchCuda&,
                                              ForwardPropagationCuda&,
                                              BackPropagationCuda&) const;

    void back_propagate_cuda(const BatchCuda&,
                             ForwardPropagationCuda&,
                             BackPropagationCuda&);

    void add_regularization_cuda(BackPropagationCuda&) const;

    void assemble_layers_error_gradient_cuda(BackPropagationCuda&) const;

    float calculate_regularization_cuda(Index, float*);

    void calculate_regularization_gradient_cuda(const Index parameters_number,
                                                float regularization,
                                                float* parameters,
                                                float* aux_vector,
                                                float* gradient);

    float l1_norm_cuda(Index, float*);

    float l2_norm_cuda(Index, float*);

    void l1_norm_gradient_cuda(const Index parameters_number,
                               float regularization,
                               float* parameters,
                               float* aux_vector,
                               float* gradient);

    void l2_norm_gradient_cuda(const Index parameters_number,
                               float regularization,
                               float* parameters,
                               float* aux_vector,
                               float* gradient);
    
protected:

    cublasHandle_t cublas_handle = nullptr;
    cudnnHandle_t cudnn_handle = nullptr;

#endif

protected:

    unique_ptr<ThreadPool> thread_pool;
    unique_ptr<ThreadPoolDevice> thread_pool_device;

   NeuralNetwork* neural_network = nullptr;

   DataSet* data_set = nullptr;

   RegularizationMethod regularization_method = RegularizationMethod::L2;

   type regularization_weight = type(0.01);

   bool display = true;

   const Eigen::array<IndexPair<Index>, 2> SSE = {IndexPair<Index>(0, 0), IndexPair<Index>(1, 1)};

   const Eigen::array<int, 1> rows_sum = {Eigen::array<int, 1>({1})};

};


struct BackPropagationLM
{
    BackPropagationLM(const Index& = 0, LossIndex* = nullptr);

    void set(const Index& = 0, LossIndex* = nullptr);

    void print() const;
    
    pair<type*, dimensions> get_output_deltas_pair() const;

    vector<vector<pair<type*, dimensions>>> get_layer_delta_pairs() const;

    Index samples_number = 0;

    Tensor<type, 1> output_deltas;
    dimensions output_deltas_dimensions;

    LossIndex* loss_index = nullptr;

    Tensor<type, 0> error;
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
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
