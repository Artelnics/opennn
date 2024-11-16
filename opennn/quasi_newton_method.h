//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Q U A S I - N E W T O N   M E T H O D    C L A S S   H E A D E R      
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef QUASINEWTONMETHOD_H
#define QUASINEWTONMETHOD_H

#include <string>
#include <iostream>

#include "config.h"
#include "loss_index.h"
#include "optimization_algorithm.h"
#include "learning_rate_algorithm.h"

namespace opennn
{

struct QuasiNewtonMehtodData;

class QuasiNewtonMethod : public OptimizationAlgorithm
{

public:

   enum class InverseHessianApproximationMethod{DFP, BFGS};

   explicit QuasiNewtonMethod(LossIndex* = nullptr);

   const LearningRateAlgorithm& get_learning_rate_algorithm() const;
   LearningRateAlgorithm* get_learning_rate_algorithm();

   const InverseHessianApproximationMethod& get_inverse_hessian_approximation_method() const;
   string write_inverse_hessian_approximation_method() const;

   const Index& get_epochs_number() const;

   // Stopping criteria

   const type& get_minimum_loss_decrease() const;
   const type& get_loss_goal() const;

   const Index& get_maximum_selection_failures() const;

   const Index& get_maximum_epochs_number() const;
   const type& get_maximum_time() const;

   // Set

   void set_loss_index(LossIndex*) override;

   void set_inverse_hessian_approximation_method(const InverseHessianApproximationMethod&);
   void set_inverse_hessian_approximation_method(const string&);

   void set_display(const bool&) final;

   void set_default() final;

   // Stopping criteria

   void set_minimum_loss_decrease(const type&);
   void set_loss_goal(const type&);

   void set_maximum_selection_failures(const Index&);

   void set_maximum_epochs_number(const Index&);
   void set_maximum_time(const type&);

   // Training

   void calculate_DFP_inverse_hessian(QuasiNewtonMehtodData&) const;

   void calculate_BFGS_inverse_hessian(QuasiNewtonMehtodData&) const;

   void calculate_inverse_hessian_approximation(QuasiNewtonMehtodData&) const;

   void update_parameters(const Batch& , ForwardPropagation& , BackPropagation& , QuasiNewtonMehtodData&) const;

   TrainingResults perform_training() final;

   string write_optimization_algorithm_type() const final;

   // Serialization
   
   void from_XML(const XMLDocument&) final;

   void to_XML(XMLPrinter&) const final;
   
   Tensor<string, 2> to_string_matrix() const final;

private: 

   LearningRateAlgorithm learning_rate_algorithm;

   InverseHessianApproximationMethod inverse_hessian_approximation_method;

   type first_learning_rate = type(0.01);

   // Stopping criteria

   type minimum_loss_decrease = NUMERIC_LIMITS_MIN;

   type training_loss_goal;

   Index maximum_selection_failures;

   Index maximum_epochs_number;

   type maximum_time;
};


struct QuasiNewtonMehtodData : public OptimizationAlgorithmData
{

    explicit QuasiNewtonMehtodData(QuasiNewtonMethod* new_quasi_newton_method = nullptr)
    {
        set(new_quasi_newton_method);
    }

    void set(QuasiNewtonMethod* = nullptr);

    virtual void print() const
    {
        cout << "Training Direction:" << endl
             << training_direction << endl
             << "Learning rate:" << endl
             << learning_rate << endl;
    }

    QuasiNewtonMethod* quasi_newton_method = nullptr;

    // Neural network data

    Tensor<type, 1> old_parameters;
    Tensor<type, 1> parameters_difference;

    Tensor<type, 1> parameters_increment;

    // Loss index data

    Tensor<type, 1> old_gradient;
    Tensor<type, 1> gradient_difference;

    Tensor<type, 2> inverse_hessian;
    Tensor<type, 2> old_inverse_hessian;

    Tensor<type, 1> old_inverse_hessian_dot_gradient_difference;

    // Optimization algorithm data

    Tensor<type, 1> BFGS;

    Index epoch = 0;

    Tensor<type, 0> training_slope;

    type learning_rate = type(0);
    type old_learning_rate = type(0);
};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
