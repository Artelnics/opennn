//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R O N   S E L E C T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

namespace opennn
{

class TrainingStrategy;
struct TrainingResults;
struct NeuronsSelectionResults;

class NeuronSelection
{
public:

    enum class StoppingCondition { MaximumTime, SelectionErrorGoal, MaximumEpochs, MaximumSelectionFailures, MaximumNeurons };

    NeuronSelection(const TrainingStrategy* = nullptr);
    virtual ~NeuronSelection() = default;

    TrainingStrategy* get_training_strategy() const;

    bool has_training_strategy() const;

    Index get_maximum_neurons() const;
    Index get_minimum_neurons() const;
    Index get_trials_number() const;

    bool get_display() const;

    type get_validation_error_goal() const;
    Index get_maximum_epochs_number() const;
    type get_maximum_time() const;

    void set(const TrainingStrategy* = nullptr);

    void set_training_strategy(TrainingStrategy*);

    void set_default();

    void set_maximum_neurons(const Index);
    void set_minimum_neurons(const Index);
    void set_trials_number(const Index);

    void set_display(bool);

    void set_validation_error_goal(const type);
    void set_maximum_epochs(const Index);
    void set_maximum_time(const type);

    string write_stopping_condition(const TrainingResults&) const;

    void delete_selection_history();
    void delete_training_error_history();
    void check() const;

    virtual NeuronsSelectionResults perform_neurons_selection() = 0;

    string write_time(const type) const;

    string get_name() const
    {
        return name;
    }

    virtual Tensor<string, 2> to_string_matrix() const { return {}; }

    virtual void from_XML(const XMLDocument&) = 0;

    virtual void to_XML(XMLPrinter&) const = 0;

    virtual void print(){}

protected:

    TrainingStrategy* training_strategy = nullptr;

    VectorI neurons_history;

    VectorR validation_error_history;

    VectorR training_error_history;

    Index minimum_neurons = 0;

    Index maximum_neurons = 0;

    Index trials_number = 1;

    type validation_error_goal = 0;

    Index maximum_epochs = 10;

    type maximum_time = 0;

    bool display = true;

    string name;
};


struct NeuronsSelectionResults
{
   NeuronsSelectionResults(const Index maximum_epochs = 0);

   void resize_history(const Index new_size);

   string write_stopping_condition() const;

   void print() const;

   // Neural network

   VectorI neurons_number_history;

   Index optimal_neurons_number = 1;

   VectorR optimal_parameters;

   // Loss index

   VectorR training_error_history;

   VectorR validation_error_history;

   type optimum_training_error = type(10);

   type optimum_validation_error = type(10);

   // Model selection

   NeuronSelection::StoppingCondition stopping_condition = NeuronSelection::StoppingCondition::MaximumTime;

   string elapsed_time;
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
