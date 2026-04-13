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

    // Constructors

    ModelSelection(TrainingStrategy* = nullptr);

    // Get

    const TrainingStrategy* get_training_strategy() const { return training_strategy; }
    bool has_training_strategy() const { return training_strategy; }

    // Set

    void set(TrainingStrategy* ts) { training_strategy = ts; }

    void set_default();

    // Model selection

    void check() const;

    NeuronsSelectionResults perform_neurons_selection();

    InputsSelectionResults perform_input_selection();

    // Serialization
    
    void from_XML(const XmlDocument&);

    void to_XML(XmlPrinter&) const;

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
