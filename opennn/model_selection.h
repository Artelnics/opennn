//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M O D E L   S E L E C T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "inputs_selection.h"
#include "neuron_selection.h"

namespace opennn
{

class TrainingStrategy;

class ModelSelection
{

public:


    explicit ModelSelection(TrainingStrategy* = nullptr);
    const TrainingStrategy* get_training_strategy() const noexcept { return training_strategy; }
    bool has_training_strategy() const noexcept { return training_strategy; }
    void set(TrainingStrategy* new_training_strategy) { training_strategy = new_training_strategy; }

    void set_default();

    NeuronsSelectionResult perform_neurons_selection();

    InputsSelectionResult perform_input_selection();
    void from_JSON(const JsonDocument&);

    void to_JSON(JsonWriter&) const;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

private:

    NeuronSelection* get_neurons_selection() const { return neurons_selection.get(); }
    InputsSelection* get_inputs_selection() const { return inputs_selection.get(); }
    void set_neurons_selection(const string&);
    void set_inputs_selection(const string&);

    TrainingStrategy* training_strategy = nullptr;

    unique_ptr<NeuronSelection> neurons_selection;

    unique_ptr<InputsSelection> inputs_selection;
};

}
