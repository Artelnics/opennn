//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Q U A S I - N E W T O N   M E T H O D    C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "loss_index.h"
#include "optimization_algorithm.h"

namespace opennn
{

struct QuasiNewtonMethodData;
struct Triplet;

class QuasiNewtonMethod final : public Optimizer
{

public:

    QuasiNewtonMethod(const Loss* = nullptr);

    // Stopping criteria

    type get_minimum_loss_decrease() const;
    type get_loss_goal() const;

    Index get_maximum_validation_failures() const;

    // Set

    void set_loss_index(Loss*) override;

    void set_display(bool) override;

    void set_default();

    // Stopping criteria

    void set_minimum_loss_decrease(const type);
    void set_loss_goal(const type);

    void set_maximum_validation_failures(const Index);

    void set_maximum_epochs(const Index);
    void set_maximum_time(const type);

    // Training

    void calculate_inverse_hessian(QuasiNewtonMethodData&) const;

    void update_parameters(const Batch& , ForwardPropagation& , BackPropagation& , QuasiNewtonMethodData&);

    TrainingResults train() override;

    // Serialization

    void from_XML(const XMLDocument&) override;

    void to_XML(XMLPrinter&) const override;

    Tensor<string, 2> to_string_matrix() const override;

    type calculate_learning_rate(const Triplet&) const;

    Triplet calculate_bracketing_triplet(const Batch&,
                                         ForwardPropagation&,
                                         BackPropagation&,
                                         QuasiNewtonMethodData&);

    pair<type, type> calculate_directional_point(const Batch&,
                                                 ForwardPropagation&,
                                                 BackPropagation&,
                                                 QuasiNewtonMethodData&,
                                                 type);

#ifdef OPENNN_CUDA

    TrainingResults train_cuda() override
    {
        throw runtime_error("CUDA train_cuda is not implemented for OptimizationMethod: QuasiNewtonMethod");
    }

#endif


private:

    type first_learning_rate = type(0.01);

    // Stopping criteria

    type minimum_loss_decrease = NUMERIC_LIMITS_MIN;

    type training_loss_goal;

    Index maximum_validation_failures;

    type learning_rate_tolerance;

    type loss_tolerance;

    const type golden_ratio = type(1.618);
};


struct Triplet
{
    Triplet();

    bool operator == (const Triplet& other_triplet) const;

    type get_length() const;

    pair<type, type> minimum() const;

    string struct_to_string() const;

    void print() const;

    void check() const;

    pair<type, type> A, U, B;
};


struct QuasiNewtonMethodData final : public OptimizerData
{
    QuasiNewtonMethodData(QuasiNewtonMethod* new_quasi_newton_method = nullptr);

    void set(QuasiNewtonMethod* = nullptr);

    void print() const;

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

    Index epoch = 0;

    Tensor0 training_slope;

    type learning_rate = type(0);
    type old_learning_rate = type(0);
};

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
