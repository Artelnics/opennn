//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M O D E L   S E L E C T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "inputs_selection.h"
#include "growing_neurons.h"

namespace opennn
{

class TrainingStrategy;

class ModelSelection
{

public:

    explicit ModelSelection(TrainingStrategy* = nullptr);
    const TrainingStrategy* get_training_strategy() const noexcept { return training_strategy; }
    void set(TrainingStrategy*);

    void set_default();

    NeuronsSelectionResult perform_neurons_selection();

    InputsSelectionResult perform_input_selection();

    string get_inputs_selection_name() const { return inputs_selection ? inputs_selection->get_name() : string(); }

    void from_JSON(const JsonDocument&);

    void to_JSON(JsonWriter&) const;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

private:

    void set_inputs_selection(const string&);

    TrainingStrategy* training_strategy = nullptr;

    GrowingNeurons neurons_selection;

    unique_ptr<InputsSelection> inputs_selection;
};

}
