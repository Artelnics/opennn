//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R O N S   S E L E C T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef NEURONSSELECTION_H
#define NEURONSSELECTION_H

#include <iostream>
#include <string>

#include "config.h"
#include "training_strategy.h"

namespace opennn
{

struct NeuronsSelectionResults;

class NeuronsSelection
{
public:

    enum class StoppingCondition { MaximumTime, SelectionErrorGoal, MaximumEpochs, MaximumSelectionFailures, MaximumNeurons };

    explicit NeuronsSelection(TrainingStrategy* = nullptr);

    TrainingStrategy* get_training_strategy() const;

    bool has_training_strategy() const;

    const Index& get_maximum_neurons() const;
    const Index& get_minimum_neurons() const;
    const Index& get_trials_number() const;

    const bool& get_display() const;

    const type& get_selection_error_goal() const;
    const Index& get_maximum_epochs_number() const;
    const type& get_maximum_time() const;

    void set(TrainingStrategy* = nullptr);

    void set_training_strategy(TrainingStrategy*);

    void set_default();

    void set_maximum_neurons_number(const Index&);
    void set_minimum_neurons(const Index&);
    void set_trials_number(const Index&);

    void set_display(const bool&);

    void set_selection_error_goal(const type&);
    void set_maximum_epochs_number(const Index&);
    void set_maximum_time(const type&);

    string write_stopping_condition(const TrainingResults&) const;

    void delete_selection_history();
    void delete_training_error_history();
    void check() const;

    virtual NeuronsSelectionResults perform_neurons_selection() = 0;

    string write_time(const type&) const;

protected:

    TrainingStrategy* training_strategy = nullptr;

    Tensor<Index, 1> neurons_history;

    Tensor<type, 1> selection_error_history;

    Tensor<type, 1> training_error_history;

    Index minimum_neurons;

    Index maximum_neurons;

    Index trials_number = 1;

    bool display = true;

    type selection_error_goal;

    Index maximum_epochs_number;

    type maximum_time;
};


struct NeuronsSelectionResults
{
    

    explicit NeuronsSelectionResults()
    {
    }

    // Epochs constructor

   explicit NeuronsSelectionResults(const Index& maximum_epochs_number);

   void resize_history(const Index& new_size);

   string write_stopping_condition() const;

   void print() const
   {
       cout << endl
            << "Neurons Selection Results" << endl
            << "Optimal neurons number: " << optimal_neurons_number << endl
            << "Optimum training error: " << optimum_training_error << endl
            << "Optimum selection error: " << optimum_selection_error << endl;
   }

   // Neural network

   Tensor<Index, 1> neurons_number_history;

   Index optimal_neurons_number = 1;

   Tensor<type, 1> optimal_parameters;

   // Loss index

   Tensor<type, 1> training_error_history;

   Tensor<type, 1> selection_error_history;

   type optimum_training_error = type(10);

   type optimum_selection_error = type(10);

   // Model selection

   NeuronsSelection::StoppingCondition stopping_condition = NeuronsSelection::StoppingCondition::MaximumTime;

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
