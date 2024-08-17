//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I N P U T S   S E L E C T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef INPUTSSELECTIONALGORITHM_H
#define INPUTSSELECTIONALGORITHM_H

// System includes

#include <iostream>
#include <string>
#include <limits>

// OpenNN includes

#include "training_strategy.h"
#include "config.h"

namespace opennn
{

struct InputsSelectionResults;

class InputsSelection
{
public:

    // Constructors

    explicit InputsSelection();

    explicit InputsSelection(TrainingStrategy*);

    // Enumerations

    enum class StoppingCondition{MaximumTime, SelectionErrorGoal, MaximumInputs, MinimumInputs, MaximumEpochs,
                           MaximumSelectionFailures, CorrelationGoal};

    // Get methods

    TrainingStrategy* get_training_strategy() const;

    bool has_training_strategy() const;

    const Index& get_trials_number() const;

    const bool& get_display() const;

    const type& get_selection_error_goal() const;
    const Index& get_maximum_iterations_number() const;
    const type& get_maximum_time() const;
    const type& get_maximum_correlation() const;
    const type& get_minimum_correlation() const;
    const type& get_tolerance() const;

    // Set methods

    void set(TrainingStrategy*);

    virtual void set_default();

    void set_trials_number(const Index&);

    void set_display(const bool&);

    void set_selection_error_goal(const type&);
    void set_maximum_epochs_number(const Index&);
    void set_maximum_time(const type&);
    void set_maximum_correlation(const type&);
    void set_minimum_correlation(const type&);

    // Performances calculation methods

    string write_stopping_condition(const TrainingResults&) const;

    // inputs selection methods

    void check() const;

    // Utilities

    Index get_input_index(const Tensor<DataSet::VariableUse, 1>&, const Index&) const;

    virtual InputsSelectionResults perform_inputs_selection() = 0;

    string write_time(const type&) const;

protected:

    TrainingStrategy* training_strategy = nullptr;

    Tensor<Index, 1> original_input_raw_variables_indices;
    Tensor<Index, 1> original_target_raw_variables_indices;

    Index trials_number = 1;

    bool display = true;
   
    // Stopping criteria

    type selection_error_goal;

    Index maximum_epochs_number;

    type maximum_correlation;

    type minimum_correlation;

    type maximum_time;

    const Eigen::array<int, 1> rows_sum = {Eigen::array<int, 1>({1})};
};


struct InputsSelectionResults
{
    // Default constructor

    explicit InputsSelectionResults()
    {
    }

    // Maximum epochs number constructor

    explicit InputsSelectionResults(const Index& maximum_epochs_number)
    {
        set(maximum_epochs_number);
    }

    Index get_epochs_number() const;

    void set(const Index& maximum_epochs_number);

   virtual ~InputsSelectionResults() {}

   string write_stopping_condition() const;

   void resize_history(const Index& new_size);


   void print() const
   {
       cout << endl;
       cout << "Inputs Selection Results" << endl;

       cout << "Optimal inputs number: " << optimal_input_raw_variables_names.size() << endl;

       cout << "Inputs: " << endl;

       for(Index i = 0; i < optimal_input_raw_variables_names.size(); i++) cout << "   " << optimal_input_raw_variables_names(i) << endl;

       cout << "Optimum training error: " << optimum_training_error << endl;
       cout << "Optimum selection error: " << optimum_selection_error << endl;
   }


   // Neural network

   Tensor<type, 1> optimal_parameters;

   // Loss index

   Tensor<type, 1> training_error_history;

   Tensor<type, 1> selection_error_history;

    // Mean Selection Error of different neural networks

   Tensor < type, 1 >  mean_selection_error_history;

    // Mean Training Error of different neural networks

   Tensor < type, 1 >  mean_training_error_history;

   type optimum_training_error = numeric_limits<type>::max();

   type optimum_selection_error = numeric_limits<type>::max();

   Tensor<string, 1> optimal_input_raw_variables_names;

   Tensor<Index, 1> optimal_input_raw_variables_indices;

   Tensor<bool, 1> optimal_inputs;

   // Model selection

   InputsSelection::StoppingCondition stopping_condition = InputsSelection::StoppingCondition::MaximumTime;

   string elapsed_time;
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
