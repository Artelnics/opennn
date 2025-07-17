//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E V E N B E R G - M A R Q U A R D T   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LEVENBERGMARQUARDTALGORITHM_H
#define LEVENBERGMARQUARDTALGORITHM_H

#include "dataset.h"
#include "optimization_algorithm.h"

namespace opennn
{

struct ForwardPropagation;
struct BackPropagationLM;
struct LevenbergMarquardtAlgorithmData;

class LevenbergMarquardtAlgorithm : public OptimizationAlgorithm
{

public:

   LevenbergMarquardtAlgorithm(const LossIndex* = nullptr);

   const type& get_minimum_loss_decrease() const;
   const type& get_loss_goal() const;

   const Index& get_maximum_selection_failures() const;

   const type& get_damping_parameter() const;

   const type& get_damping_parameter_factor() const;

   const type& get_minimum_damping_parameter() const;
   const type& get_maximum_damping_parameter() const;

   // Set

   void set_default();

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

   void check() const override;

   TrainingResults perform_training() override;

   void update_parameters(
           const Batch&,
           ForwardPropagation&,
           BackPropagationLM&,
           LevenbergMarquardtAlgorithmData&);

   string get_name() const override;

   // Serialization

   Tensor<string, 2> to_string_matrix() const override;
   
   void from_XML(const XMLDocument&) override;

   void to_XML(XMLPrinter&) const override;

#ifdef OPENNN_CUDA

   TrainingResults perform_training_cuda() override 
   {
       throw runtime_error("CUDA perform_training_cuda is not implemented for OptimizationMethod: LevenbergMarquardtAlgorithm");
   }

#endif
   
private:

   type damping_parameter = type(0);

   type minimum_damping_parameter = type(0);

   type maximum_damping_parameter = type(0);

   type damping_parameter_factor = type(0);

   // Stopping criteria 

   type minimum_loss_decrease = type(0);

   type training_loss_goal = type(0);

   Index maximum_selection_failures = 0;
};


struct LevenbergMarquardtAlgorithmData : public OptimizationAlgorithmData
{

    LevenbergMarquardtAlgorithmData(LevenbergMarquardtAlgorithm* new_Levenberg_Marquardt_method = nullptr);

    void set(LevenbergMarquardtAlgorithm* = nullptr);

    LevenbergMarquardtAlgorithm* Levenberg_Marquardt_algorithm = nullptr;

    // Neural network data

//    Tensor<type, 1> parameters;
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
