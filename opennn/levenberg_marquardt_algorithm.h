//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E V E N B E R G - M A R Q U A R D T   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "dataset.h"
#include "layer.h"
#include "optimizer.h"

namespace opennn
{

struct ForwardPropagation;
struct BackPropagationLM;
struct LevenbergMarquardtAlgorithmData;

struct LayerBackPropagationLM
{
    virtual void initialize() = 0;

    virtual vector<TensorView*> get_gradient_views();

    virtual vector<TensorView*> get_workspace_views();

    vector<TensorView> get_input_gradients() const;

    Index batch_size = 0;

    Layer* layer = nullptr;

    vector<TensorView> input_gradients;
    vector<TensorView> output_gradients;
};

struct NeuralNetworkBackPropagationLM
{
    NeuralNetworkBackPropagationLM(NeuralNetwork* new_neural_network = nullptr);

    void set(const Index = 0, NeuralNetwork* = nullptr);

    const vector<unique_ptr<LayerBackPropagationLM>>& get_layers() const;

    NeuralNetwork* get_neural_network() const;

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

    TensorView get_output_gradients() const;

    vector<vector<TensorView>> get_layer_gradients() const;

    Index samples_number = 0;

    VectorR output_gradients;
    Shape output_gradient_dimensions;

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

    //VectorR regularization_gradient;
    //MatrixR regularization_hessian;
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

   void update_parameters(
           const Batch&,
           ForwardPropagation&,
           BackPropagationLM&,
           LevenbergMarquardtAlgorithmData&);

   // Serialization

   void from_XML(const XMLDocument&) override;

   void to_XML(XMLPrinter&) const override;
   
private:

   VectorR calculate_numerical_gradient_lm();

   void compute_jacobian(const Batch& batch,
                         const ForwardPropagation& fp,
                         BackPropagationLM& bp_lm);

   // Specific logic for Dense layers
/*
   void insert_dense_jacobian(const Dense<2>* layer,
                              const ForwardPropagation& fp,
                              Index layer_index,
                              Index parameter_offset,
                              MatrixR& jacobian);
*/
   type damping_parameter = type(0);

   type minimum_damping_parameter = type(0);

   type maximum_damping_parameter = type(0);

   type damping_parameter_factor = type(0);

   // Stopping criteria 

   type minimum_loss_decrease = type(0);

};

struct LevenbergMarquardtAlgorithmData final : public OptimizerData
{

    LevenbergMarquardtAlgorithmData(LevenbergMarquardtAlgorithm* new_Levenberg_Marquardt_method = nullptr);

    void set(LevenbergMarquardtAlgorithm* = nullptr);

    LevenbergMarquardtAlgorithm* Levenberg_Marquardt_algorithm = nullptr;

    // Neural network data

    VectorR old_parameters;
    VectorR parameter_differences;

    VectorR parameter_updates;

    // Loss index data

    //type old_loss = type(0);

    // Optimization algorithm data

    Index epoch = 0;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
