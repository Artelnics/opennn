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

    NeuronSelection(TrainingStrategy* = nullptr);
    virtual ~NeuronSelection() = default;

    const TrainingStrategy* get_training_strategy() const { return training_strategy; }

    bool has_training_strategy() const { return training_strategy; }

    bool get_display() const { return display; }

    void set(TrainingStrategy*);

    void set_training_strategy(TrainingStrategy* ts) { training_strategy = ts; }

    void set_default();

    void set_maximum_neurons(const Index n) { maximum_neurons = n; }
    void set_minimum_neurons(const Index n) { minimum_neurons = n; }
    void set_trials_number(const Index n) { trials_number = n; }

    void set_display(bool d) { display = d; }

    void set_validation_error_goal(const type v) { validation_error_goal = v; }
    void set_maximum_epochs(const Index n) { maximum_epochs = n; }
    void set_maximum_validation_failures(const Index n) { maximum_validation_failures = n; }
    void set_maximum_time(const type t) { maximum_time = t; }

void check() const;

    virtual NeuronsSelectionResults perform_neurons_selection() = 0;

    string get_name() const
    {
        return name;
    }

    virtual void from_XML(const XmlDocument&) = 0;

    virtual void to_XML(XmlPrinter&) const = 0;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

    virtual void print(){}

protected:

    TrainingStrategy* training_strategy = nullptr;

    VectorR validation_error_history;

    VectorR training_error_history;

    Index minimum_neurons = 0;

    Index maximum_neurons = 0;

    Index trials_number = 1;

    type validation_error_goal = 0;

    Index maximum_epochs = 10;

    Index maximum_validation_failures = 100;

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
// Licensed under the GNU Lesser General Public License v2.1 or later.
