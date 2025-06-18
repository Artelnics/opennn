//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M O D E L   S E L E C T I O N   C L A S S   H E A D E R               
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef MODELSELECTION_H
#define MODELSELECTION_H

#include "growing_neurons.h"
#include "growing_inputs.h"
#include "genetic_algorithm.h"

namespace opennn
{

class TrainingStrategy;

class ModelSelection
{

public: 

    // Constructors

    ModelSelection(TrainingStrategy* = nullptr);

    enum class NeuronsSelectionMethod{GROWING_NEURONS};

    enum class InputsSelectionMethod{GROWING_INPUTS, GENETIC_ALGORITHM};

    // Get

    TrainingStrategy* get_training_strategy() const;
    bool has_training_strategy() const;

    const NeuronsSelectionMethod& get_neurons_selection_method() const;
    const InputsSelectionMethod& get_inputs_selection_method() const;

    GrowingNeurons* get_growing_neurons();

    GrowingInputs* get_growing_inputs();
    GeneticAlgorithm* get_genetic_algorithm();

    // Set

    void set(TrainingStrategy* = nullptr);

    void set_default();

    void set_display(const bool&);

    void set_neurons_selection_method(const NeuronsSelectionMethod&);
    void set_neurons_selection_method(const string&);

    void set_inputs_selection_method(const InputsSelectionMethod&);
    void set_inputs_selection_method(const string&);

    // Model selection

    void check() const;

    NeuronsSelectionResults perform_neurons_selection();

    InputsSelectionResults perform_input_selection();

    // Serialization
    
    void from_XML(const XMLDocument&);

    void to_XML(XMLPrinter&) const;

    string write_neurons_selection_method() const;
    string write_inputs_selection_method() const;

    void print() const;
    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

private: 

    TrainingStrategy* training_strategy = nullptr;

    GrowingNeurons growing_neurons;

    GrowingInputs growing_inputs;

    GeneticAlgorithm genetic_algorithm;

    NeuronsSelectionMethod neurons_selection_method;

    InputsSelectionMethod inputs_selection_method;

    bool display = true;
};

}

#endif
