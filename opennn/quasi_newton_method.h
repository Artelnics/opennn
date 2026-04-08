//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Q U A S I - N E W T O N   M E T H O D    C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "loss.h"
#include "optimizer.h"

namespace opennn
{

struct QuasiNewtonMethodData;

class QuasiNewtonMethod final : public Optimizer
{

public:

    QuasiNewtonMethod(Loss* = nullptr);

    // Set

    void set_default();

    // Stopping criteria

    void set_minimum_loss_decrease(const type v) { minimum_loss_decrease = v; }

    // Training

    void update_parameters(const Batch& , ForwardPropagation& , BackPropagation& , QuasiNewtonMethodData&);

    TrainingResults train() override;

    // Serialization

    void from_XML(const XMLDocument&) override;

    void to_XML(XMLPrinter&) const override;

private:

    void calculate_inverse_hessian(QuasiNewtonMethodData&) const;

    pair<type, type> calculate_directional_point(const Batch&,
                                                 ForwardPropagation&,
                                                 BackPropagation&,
                                                 QuasiNewtonMethodData&,
                                                 type);

    type first_learning_rate = type(0.01);

    // Stopping criteria

    type minimum_loss_decrease = EPSILON;

};

struct QuasiNewtonMethodData final : public OptimizerData
{
    QuasiNewtonMethodData(QuasiNewtonMethod* new_quasi_newton_method = nullptr);

    void set(QuasiNewtonMethod* = nullptr);

    void print() const override;

    QuasiNewtonMethod* quasi_newton_method = nullptr;

    // Neural network data

    VectorR old_parameters;
    VectorR parameter_differences;

    VectorR parameter_updates;

    // Loss index data

    VectorR old_gradient;
    VectorR gradient_difference;

    MatrixR inverse_hessian;
    MatrixR old_inverse_hessian;

    VectorR old_inverse_hessian_dot_gradient_difference;

    // Optimization algorithm data

    VectorR BFGS;

    type training_slope = type(0);

    type learning_rate = type(0);
    type old_learning_rate = type(0);
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
