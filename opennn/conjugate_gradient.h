//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N J U G A T E   G R A D I E N T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef CONJUGATEGRADIENT_H
#define CONJUGATEGRADIENT_H

#include "optimization_algorithm.h"
#include "learning_rate_algorithm.h"

namespace opennn
{

struct ConjugateGradientData;

class ConjugateGradient : public OptimizationAlgorithm
{

public:

   enum class TrainingDirectionMethod{PR, FR};

    ConjugateGradient(LossIndex* = nullptr);

   // Get

   const LearningRateAlgorithm& get_learning_rate_algorithm() const;
   LearningRateAlgorithm* get_learning_rate_algorithm();

   // Training operators

   const TrainingDirectionMethod& get_training_direction_method() const;
   string write_training_direction_method() const;

   // Stopping criteria
 
   const type& get_minimum_loss_decrease() const;
   const type& get_loss_goal() const;
   const Index& get_maximum_selection_failures() const;
   const Index& get_maximum_epochs_number() const;
   const type& get_maximum_time() const;

   // Set

   void set_default();

   void set_loss_index(LossIndex*) override;

   // Training operators

   void set_training_direction_method(const TrainingDirectionMethod&);
   void set_training_direction_method(const string&);

   // Stopping criteria 

   void set_loss_goal(const type&);
   void set_minimum_loss_decrease(const type&);
   void set_maximum_selection_failures(const Index&);

   void set_maximum_epochs_number(const Index&);
   void set_maximum_time(const type&);

   // Training direction

   type calculate_PR_parameter(const Tensor<type, 1>&, const Tensor<type, 1>&) const;
   type calculate_FR_parameter(const Tensor<type, 1>&, const Tensor<type, 1>&) const;

   void calculate_PR_training_direction(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&, Tensor<type, 1>&) const;
   void calculate_FR_training_direction(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&, Tensor<type, 1>&) const;

   void calculate_gradient_descent_training_direction(const Tensor<type, 1>&, Tensor<type, 1>&) const;

   void calculate_conjugate_gradient_training_direction(const Tensor<type, 1>&,
                                                        const Tensor<type, 1>&,
                                                        const Tensor<type, 1>&,
                                                        Tensor<type, 1>&) const;

   // Training

   TrainingResults perform_training() override;

   string write_optimization_algorithm_type() const override;

   // Serialization

   Tensor<string, 2> to_string_matrix() const override;

   void from_XML(const XMLDocument&) override;

   void to_XML(XMLPrinter&) const override;

   void update_parameters(
           const Batch&,
           ForwardPropagation&,
           BackPropagation&,
           ConjugateGradientData&) const;

private:

   type first_learning_rate = type(0.01);

   TrainingDirectionMethod training_direction_method = ConjugateGradient::TrainingDirectionMethod::FR;

   LearningRateAlgorithm learning_rate_algorithm;

   // Stopping criteria

   type minimum_loss_decrease = type(0);

   type training_loss_goal = type(0);

   Index maximum_selection_failures;

   Index maximum_epochs_number;

   type maximum_time;
};


struct ConjugateGradientData : public OptimizationAlgorithmData
{
    ConjugateGradientData(ConjugateGradient* = nullptr);

    void set(ConjugateGradient* = nullptr);

    virtual void print() const;

    ConjugateGradient* conjugate_gradient = nullptr;  

    Tensor<type, 1> parameters_increment;

    Tensor<type, 1> old_gradient;

    Tensor<type, 1> old_training_direction;

    Index epoch = 0;

    type learning_rate = type(0);
    type old_learning_rate = type(0);

    Tensor<type, 0> training_slope;
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
