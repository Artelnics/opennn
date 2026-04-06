//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T O C H A S T I C   G R A D I E N T   D E S C E N T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "optimizer.h"

namespace opennn
{

struct BackPropagation;
struct StochasticGradientDescentData;

#ifdef CUDA
struct SGDOptimizationDataCuda;
#endif


class StochasticGradientDescent final : public Optimizer
{

public:

    StochasticGradientDescent(const Loss* = nullptr);

    void set_default();

    void set_batch_size(const Index);

    Index get_samples_number() const;

    void set_initial_learning_rate(const type);
    void set_initial_decay(const type);
    void set_momentum(const type);
    void set_nesterov(bool);

    void update_parameters(BackPropagation& , StochasticGradientDescentData&, type) const;

    TrainingResults train() override;


    void from_XML(const XMLDocument&) override;

    void to_XML(XMLPrinter&) const override;

private:

    type initial_learning_rate;

    type initial_decay;

    type momentum = type(0);

    bool nesterov = false;

    Index batch_size = 1000;

#ifdef CUDA

public:

    TrainingResults train_cuda() override;

    void update_parameters(BackPropagationCuda&, SGDOptimizationDataCuda&, type) const;

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


#ifdef CUDA

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
// Licensed under the GNU Lesser General Public License v2.1 or later.
