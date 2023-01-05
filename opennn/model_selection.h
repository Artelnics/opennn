//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M O D E L   S E L E C T I O N   C L A S S   H E A D E R               
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef MODELSELECTION_H
#define MODELSELECTION_H

// System includes

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <ctime>

// OpenNN includes

#include "config.h"
#include "training_strategy.h"
#include "growing_neurons.h"
#include "growing_inputs.h"
#include "genetic_algorithm.h"

namespace opennn
{

/// This class represents the concept of model selection[1] algorithm in OpenNN.

///
/// It is used for finding a network architecture with maximum generalization capabilities.
///
/// [1] Neural Designer "Model Selection Algorithms in Predictive Analytics." \ref https://www.neuraldesigner.com/blog/model-selection

class ModelSelection
{

public:  

    // Constructors

    explicit ModelSelection();

    explicit ModelSelection(TrainingStrategy*);

    /// Enumeration of all the available neurons selection algorithms.

    enum class NeuronsSelectionMethod{GROWING_NEURONS};

    /// Enumeration of all the available inputs selection algorithms.

    enum class InputsSelectionMethod{GROWING_INPUTS, GENETIC_ALGORITHM};

    // Get methods

    TrainingStrategy* get_training_strategy_pointer() const;
    bool has_training_strategy() const;

    const NeuronsSelectionMethod& get_neurons_selection_method() const;
    const InputsSelectionMethod& get_inputs_selection_method() const;

    GrowingNeurons* get_growing_neurons_pointer();

    GrowingInputs* get_growing_inputs_pointer();
    GeneticAlgorithm* get_genetic_algorithm_pointer();

    // Set methods

    void set(TrainingStrategy*);

    void set_default();

    void set_display(const bool&);

    void set_neurons_selection_method(const NeuronsSelectionMethod&);
    void set_neurons_selection_method(const string&);

    void set_inputs_selection_method(const InputsSelectionMethod&);
    void set_inputs_selection_method(const string&);

    // Model selection methods

    void check() const;

    NeuronsSelectionResults perform_neurons_selection();

    InputsSelectionResults perform_inputs_selection();

    // Serialization methods
    
    void from_XML(const tinyxml2::XMLDocument&);

    void write_XML(tinyxml2::XMLPrinter&) const;

    string write_neurons_selection_method() const;
    string write_inputs_selection_method() const;

    void print() const;
    void save(const string&) const;
    void load(const string&);

private: 

    /// Pointer to a training strategy object.

    TrainingStrategy* training_strategy_pointer = nullptr;

    /// Growing neurons object to be used for neurons selection.

    GrowingNeurons growing_neurons;

    /// Growing inputs object to be used for inputs selection.

    GrowingInputs growing_inputs;

    /// Genetic algorithm object to be used for inputs selection.

    GeneticAlgorithm genetic_algorithm;

    /// Type of neurons selection algorithm.

    NeuronsSelectionMethod neurons_selection_method;

    /// Type of inputs selection algorithm.

    InputsSelectionMethod inputs_selection_method;

    /// Display messages to screen.

    bool display = true;
};

}

#endif
