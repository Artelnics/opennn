//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M O D E L   S E L E C T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/model_selection/input_selection.h"
#include "opennn/model_selection/growing_neurons.h"

namespace opennn
{

class Training;

class ModelSelection
{

public:

    explicit ModelSelection(Training* = nullptr);
    const Training* get_training() const noexcept { return training; }
    void set(Training*);

    void set_default();

    NeuronsSelectionResult perform_neurons_selection() { return neurons_selection.perform_neurons_selection(); }

    InputSelectionResult perform_input_selection() { return input_selection->perform_input_selection(); }

    string get_input_selection_name() const { return input_selection ? input_selection->get_name() : string(); }

    void from_JSON(const JsonDocument&);

    void to_JSON(JsonWriter&) const;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

private:

    void set_input_selection(const string&);

    Training* training = nullptr;

    GrowingNeurons neurons_selection;

    unique_ptr<InputSelection> input_selection;
};

}
