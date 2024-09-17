//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E V E N B E R G - M A R Q U A R D T   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LEVENBERGMARQUARDTALGORITHM_H
#define LEVENBERGMARQUARDTALGORITHM_H

// System includes

#include <string>

// OpenNN includes

#include "config.h"
#include "optimization_algorithm.h"

// Eigen includes

#include "../eigen/Eigen/Dense"

namespace opennn
{

struct LevenbergMarquardtAlgorithmData;


class LevenbergMarquardtAlgorithm : public OptimizationAlgorithm
{

public:

   // Constructors

   explicit LevenbergMarquardtAlgorithm();

   explicit LevenbergMarquardtAlgorithm(LossIndex*);

   // Get

   // Stopping criteria

   const type& get_minimum_loss_decrease() const;
   const type& get_loss_goal() const;

   const Index& get_maximum_selection_failures() const;

   const Index& get_maximum_epochs_number() const;
   const type& get_maximum_time() const;

   // Utilities

   const type& get_damping_parameter() const;

   const type& get_damping_parameter_factor() const;

   const type& get_minimum_damping_parameter() const;
   const type& get_maximum_damping_parameter() const;

   // Set

   void set_default() final;

   void set_damping_parameter(const type&);

   void set_damping_parameter_factor(const type&);

   void set_minimum_damping_parameter(const type&);
   void set_maximum_damping_parameter(const type&);

   // Stopping criteria

   void set_minimum_loss_decrease(const type&);
   void set_loss_goal(const type&);

   void set_maximum_selection_failures(const Index&);

   void set_maximum_epochs_number(const Index&);
   void set_maximum_time(const type&);

   // Training

   void check() const final;

   TrainingResults perform_training() final;

   void update_parameters(
           const Batch&,
           ForwardPropagation&,
           BackPropagationLM&,
           LevenbergMarquardtAlgorithmData&);

   string write_optimization_algorithm_type() const final;

   // Serialization

   Tensor<string, 2> to_string_matrix() const final;
   
   void from_XML(const tinyxml2::XMLDocument&) final;

   void to_XML(tinyxml2::XMLPrinter&) const final;
   
private:

   // MEMBERS

   type damping_parameter = type(0);

   type minimum_damping_parameter = type(0);

   type maximum_damping_parameter = type(0);

   type damping_parameter_factor = type(0);

   // Stopping criteria 

   type minimum_loss_decrease = type(0);

   type training_loss_goal = type(0);

   Index maximum_selection_failures = 0;

   Index maximum_epochs_number = 0;

   type maximum_time = type(0);
};


struct LevenbergMarquardtAlgorithmData : public OptimizationAlgorithmData
{
    explicit LevenbergMarquardtAlgorithmData()
    {
    }

    explicit LevenbergMarquardtAlgorithmData(LevenbergMarquardtAlgorithm* new_Levenberg_Marquardt_method)
    {
        set(new_Levenberg_Marquardt_method);
    }

    virtual ~LevenbergMarquardtAlgorithmData() {}

    void set(LevenbergMarquardtAlgorithm* new_Levenberg_Marquardt_method);

    LevenbergMarquardtAlgorithm* Levenberg_Marquardt_algorithm = nullptr;

    // Neural network data

    Tensor<type, 1> old_parameters;
    Tensor<type, 1> parameters_difference;

    Tensor<type, 1> parameters_increment;

    // Loss index data

    type old_loss = type(0);

    // Optimization algorithm data

    Index epoch = 0;
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
