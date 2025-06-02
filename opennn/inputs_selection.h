//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I N P U T S   S E L E C T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef INPUTSSELECTIONALGORITHM_H
#define INPUTSSELECTIONALGORITHM_H

#include "training_strategy.h"

namespace opennn
{

struct InputsSelectionResults;

class InputsSelection
{
public:

    enum class StoppingCondition {
        MaximumTime,
        SelectionErrorGoal,
        MaximumInputs,
        MinimumInputs,
        MaximumEpochs,
        MaximumSelectionFailures,
        CorrelationGoal
    };


    InputsSelection(TrainingStrategy* = nullptr);

    TrainingStrategy* get_training_strategy() const;

    bool has_training_strategy() const;

    const Index& get_trials_number() const;

    const bool& get_display() const;

    const type& get_selection_error_goal() const;
    const Index& get_maximum_iterations_number() const;
    const type& get_maximum_time() const;
    const type& get_maximum_correlation() const;
    const type& get_minimum_correlation() const;

    void set(TrainingStrategy* = nullptr);

    void set_trials_number(const Index&);

    void set_display(const bool&);

    void set_selection_error_goal(const type&);
    void set_maximum_epochs_number(const Index&);
    void set_maximum_time(const type&);
    void set_maximum_correlation(const type&);
    void set_minimum_correlation(const type&);

    string write_stopping_condition(const TrainingResults&) const;

    void check() const;

    Index get_input_index(const Tensor<Dataset::VariableUse, 1>&, const Index&) const;

    virtual InputsSelectionResults perform_input_selection() = 0;

    string write_time(const type&) const;

protected:

    TrainingStrategy* training_strategy = nullptr;

    vector<Index> original_input_raw_variable_indices;
    vector<Index> original_target_raw_variable_indices;

    Index trials_number = 1;

    bool display = true;
   
    // Stopping criteria

    type selection_error_goal;

    Index maximum_epochs_number;

    type maximum_correlation;

    type minimum_correlation;

    type maximum_time;
};


struct InputsSelectionResults
{
    InputsSelectionResults(const Index& = 0);

    Index get_epochs_number() const;

    void set(const Index& = 0);

    string write_stopping_condition() const;

    void resize_history(const Index& new_size);

    void print() const;

    // Neural network

    Tensor<type, 1> optimal_parameters;

    // Loss index

    Tensor<type, 1> training_error_history;

    Tensor<type, 1> selection_error_history;

    // Mean Selection Error of different neural networks

    Tensor<type, 1>  mean_selection_error_history;

    // Mean Training Error of different neural networks

    Tensor<type, 1> mean_training_error_history;

    type optimum_training_error = numeric_limits<type>::max();

    type optimum_selection_error = numeric_limits<type>::max();

    vector<string> optimal_input_raw_variable_names;

    vector<Index> optimal_input_raw_variables_indices;

    Tensor<bool, 1> optimal_inputs;

    // Model selection

    InputsSelection::StoppingCondition stopping_condition = InputsSelection::StoppingCondition::MaximumTime;

    string elapsed_time;
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
