//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M O D E L   S E L E C T I O N   C L A S S   H E A D E R               
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef MODELSELECTION_H
#define MODELSELECTION_H

#include "inputs_selection.h"
#include "neurons_selection.h"

namespace opennn
{

class TrainingStrategy;

class ModelSelection
{

public: 

    // Constructors

    ModelSelection(const TrainingStrategy* = nullptr);

    // Get

    TrainingStrategy* get_training_strategy() const;
    bool has_training_strategy() const;

    NeuronsSelection* get_neurons_selection() const;
    InputsSelection* get_inputs_selection() const;

    // Set

    void set(const TrainingStrategy* = nullptr);

    void set_default();

    void set_neurons_selection(const string&);
    void set_inputs_selection(const string&);

    // Model selection

    void check() const;

    NeuronsSelectionResults perform_neurons_selection();

    InputsSelectionResults perform_input_selection();

    // Serialization
    
    void from_XML(const XMLDocument&);

    void to_XML(XMLPrinter&) const;

    void print() const;
    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

private: 

    TrainingStrategy* training_strategy = nullptr;

    unique_ptr<NeuronsSelection> neurons_selection;

    unique_ptr<InputsSelection> inputs_selection;
};

}

#endif
