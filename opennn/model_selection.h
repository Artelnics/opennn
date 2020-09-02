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
#include "pruning_inputs.h"
#include "genetic_algorithm.h"

namespace OpenNN
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

    // Destructor

    virtual ~ModelSelection();

    /// Enumeration of all the available order selection algorithms.

    enum NeuronsSelectionMethod{NO_NEURONS_SELECTION, GROWING_NEURONS};

    /// Enumeration of all the available inputs selection algorithms.

    enum InputsSelectionMethod{NO_INPUTS_SELECTION, GROWING_INPUTS, PRUNING_INPUTS, GENETIC_ALGORITHM};

    /// This structure contains the results from the model selection process.

    struct Results
    {
        /// Default constructor.

        explicit Results();

        /// Pointer to a structure with the results from the growing neurons selection algorithm.

        GrowingNeurons::GrowingNeuronsResults* growing_neurons_results_pointer = nullptr;


        /// Pointer to a structure with the results from the growing inputs selection algorithm.

        GrowingInputs::GrowingInputsResults* growing_inputs_results_pointer = nullptr;


        /// Pointer to a structure with the results from the pruning inputs selection algorithm.

        PruningInputs::PruningInputsResults* pruning_inputs_results_pointer = nullptr;


        /// Pointer to a structure with the results from the genetic inputs selection algorithm.

        GeneticAlgorithm::GeneticAlgorithmResults* genetic_algorithm_results_pointer = nullptr;
    };

    
    // Get methods

    TrainingStrategy* get_training_strategy_pointer() const;
    bool has_training_strategy() const;

    const NeuronsSelectionMethod& get_neurons_selection_method() const;
    const InputsSelectionMethod& get_inputs_selection_method() const;

    GrowingNeurons* get_growing_neurons_pointer();

    GrowingInputs* get_growing_inputs_pointer();
    PruningInputs* get_pruning_inputs_pointer();
    GeneticAlgorithm* get_genetic_algorithm_pointer();

    // Set methods

    void set_default();

    void set_display(const bool&);

    void set_training_strategy_pointer(TrainingStrategy*);

    void set_neurons_selection_method(const NeuronsSelectionMethod&);
    void set_neurons_selection_method(const string&);

    void set_inputs_selection_method(const InputsSelectionMethod&);
    void set_inputs_selection_method(const string&);

    void set_approximation(const bool&);

    // Model selection methods

    void check() const;

    Results perform_neurons_selection();

    Results perform_inputs_selection();

    Results perform_model_selection();

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

    /// Growing order object to be used for order selection.

    GrowingNeurons growing_neurons;

    /// Growing inputs object to be used for inputs selection.

    GrowingInputs growing_inputs;

    /// Pruning inputs object to be used for inputs selection.

    PruningInputs pruning_inputs;

    /// Genetic algorithm object to be used for inputs selection.

    GeneticAlgorithm genetic_algorithm;

    /// Type of order selection algorithm.

    NeuronsSelectionMethod neurons_selection_method;

    /// Type of inputs selection algorithm.

    InputsSelectionMethod inputs_selection_method;

    /// Display messages to screen.

    bool display = true;
};

}

#endif
