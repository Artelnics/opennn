//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E V E N B E R G - M A R Q U A R D T   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "batch.h"
#include "dense_layer.h"
#include "optimizer.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

class NeuralNetwork;
struct ForwardPropagation;
struct BackPropagationLM;

struct LayerBackPropagationLM
{
    virtual void initialize() = 0;

    virtual vector<TensorView*> get_gradient_views();

    virtual vector<TensorView*> get_workspace_views();

    vector<TensorView> get_input_deltas() const;

    Index batch_size = 0;

    Layer* layer = nullptr;

    vector<TensorView> input_deltas;
    vector<TensorView> output_deltas;
};

struct NeuralNetworkBackPropagationLM
{
    NeuralNetworkBackPropagationLM(NeuralNetwork* new_neural_network = nullptr);

    void set(const Index = 0, NeuralNetwork* = nullptr);

    const vector<unique_ptr<LayerBackPropagationLM>>& get_layers() const;

    const NeuralNetwork* get_neural_network() const;

    void print();

    Index batch_size = 0;

    NeuralNetwork* neural_network = nullptr;

    vector<unique_ptr<LayerBackPropagationLM>> layers;

    VectorR gradient;

    VectorR workspace;
};

struct BackPropagationLM
{
    BackPropagationLM(const Index = 0, Loss* = nullptr);
    virtual ~BackPropagationLM() = default;

    void set(const Index = 0, Loss* = nullptr);

    void print() const;

    TensorView get_output_deltas() const;

    vector<vector<TensorView>> get_layer_gradients() const;

    Index samples_number = 0;

    VectorR output_deltas;
    Shape output_delta_dimensions;

    Loss* loss = nullptr;

    type error;
    type regularization = type(0);
    type loss_value = type(0);

    NeuralNetworkBackPropagationLM neural_network;

    VectorR errors;
    VectorR squared_errors;
    MatrixR squared_errors_jacobian;

    VectorR gradient;
    MatrixR hessian;
};

class LevenbergMarquardtAlgorithm final : public Optimizer
{

public:

   LevenbergMarquardtAlgorithm(Loss* = nullptr);

   // Set

   void set_default();

   void set_damping_parameter(const type);

   void set_damping_parameter_factor(const type);

   void set_minimum_damping_parameter(const type);
   void set_maximum_damping_parameter(const type);

   // Stopping criteria

   void set_minimum_loss_decrease(const type);
   // Training

   void check() const override;

   TrainingResults train() override;

   enum DataSlot { ParameterUpdate };

   void update_parameters(
           const Batch&,
           ForwardPropagation&,
           BackPropagationLM&,
           OptimizerData&);

   // Serialization

   void from_XML(const XmlDocument&) override;

   void to_XML(XmlPrinter&) const override;
   
private:

    VectorR calculate_numerical_gradient();
    MatrixR calculate_numerical_jacobian();
    MatrixR calculate_numerical_hessian();

   void back_propagate(const Batch&, const ForwardPropagation&, BackPropagationLM&);

   void calculate_errors(const Batch&, const ForwardPropagation&, BackPropagationLM&) const;
   void calculate_squared_errors(const Batch&, const ForwardPropagation&, BackPropagationLM&) const;
   void calculate_error(const Batch&, const ForwardPropagation&, BackPropagationLM&) const;

   void compute_jacobian(const Batch& batch,
                         const ForwardPropagation& fp,
                         BackPropagationLM& bp_lm);

   void insert_dense_jacobian(const Dense<2>* layer,
                              const ForwardPropagation& fp,
                              Index layer_index,
                              Index parameter_offset,
                              MatrixR& jacobian);
   type damping_parameter = type(0);

   type minimum_damping_parameter = type(0);

   type maximum_damping_parameter = type(0);

   type damping_parameter_factor = type(0);

   // Stopping criteria 

   type minimum_loss_decrease = type(0);

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
