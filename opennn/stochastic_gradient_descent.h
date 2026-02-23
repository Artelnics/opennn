//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T O C H A S T I C   G R A D I E N T   D E S C E N T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "optimization_algorithm.h"

namespace opennn
{

struct BackPropagation;
struct StochasticGradientDescentData;

#ifdef OPENNN_CUDA
struct SGDOptimizationDataCuda;
#endif


class StochasticGradientDescent final : public Optimizer
{

public:

    StochasticGradientDescent(const Loss* = nullptr);

    type get_initial_learning_rate() const;
    type get_initial_decay() const;
    type get_momentum() const;
    bool get_nesterov() const;

    type get_loss_goal() const;

    void set_default();

    void set_batch_size(const Index);

    Index get_samples_number() const;

    void set_initial_learning_rate(const type);
    void set_initial_decay(const type);
    void set_momentum(const type);
    void set_nesterov(bool);

    void set_maximum_epochs(const Index);

    void set_loss_goal(const type);
    void set_maximum_time(const type);

    void update_parameters(BackPropagation& , StochasticGradientDescentData&, type) const;

    TrainingResults train() override;

    Tensor<string, 2> to_string_matrix() const override;

    void from_XML(const XMLDocument&) override;

    void to_XML(XMLPrinter&) const override;

private:

    type initial_learning_rate;

    type initial_decay;

    type momentum = type(0);

    bool nesterov = false;

    Index batch_size = 1000;

    type training_loss_goal = type(0);

    Index maximum_validation_failures = numeric_limits<Index>::max();

#ifdef OPENNN_CUDA

public:

    TrainingResults train_cuda() override;

    void update_parameters(BackPropagationCuda&, SGDOptimizationDataCuda&) const;

#endif

};


struct StochasticGradientDescentData final : public OptimizerData
{
    StochasticGradientDescentData(StochasticGradientDescent* = nullptr);

    void set(StochasticGradientDescent* = nullptr);

    StochasticGradientDescent* stochastic_gradient_descent = nullptr;

    Index iteration = 0;

    VectorR parameter_updates;
    VectorR last_parameter_updates;
};


#ifdef OPENNN_CUDA

struct SGDOptimizationDataCuda final : public OptimizerData
{
    SGDOptimizationDataCuda(StochasticGradientDescent* = nullptr);

    //~SGDOptimizationDataCuda() { free(); }

    void set(StochasticGradientDescent* = nullptr);

    void print() const;

    StochasticGradientDescent* stochastic_gradient_descent = nullptr;

    Index iteration = 0;

    TensorCuda velocity;
};

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
