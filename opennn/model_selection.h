/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M O D E L   S E L E C T I O N   C L A S S   H E A D E R                                                    */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __MODELSELECTION_H__
#define __MODELSELECTION_H__

// System includes

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <ctime>

// OpenNN includes

#include "training_strategy.h"

#include "incremental_order.h"
#include "golden_section_order.h"
#include "simulated_annealing_order.h"

#include "growing_inputs.h"
#include "pruning_inputs.h"
#include "genetic_algorithm.h"

// TinyXml includes

#include "tinyxml2.h"

namespace OpenNN
{

/// This class represents the concept of model selection algorithm.
/// It is used for finding a network architecture with maximum selection capabilities.

class ModelSelection
{

public:  

    // DEFAULT CONSTRUCTOR

    explicit ModelSelection();

    // TRAINING STRATEGY CONSTRUCTOR

    explicit ModelSelection(TrainingStrategy*);

    // FILE CONSTRUCTOR

    explicit ModelSelection(const string&);

    // XML CONSTRUCTOR

    explicit ModelSelection(const tinyxml2::XMLDocument&);


    // DESTRUCTOR

    virtual ~ModelSelection();

    /// Enumeration of all the available types of inputs selection algorithms.

    enum InputsSelectionMethod
    {
        NO_INPUTS_SELECTION,
        GROWING_INPUTS,
        PRUNING_INPUTS,
        GENETIC_ALGORITHM
    };

    /// Enumeration of all the available types of order selection algorithms.

    enum OrderSelectionMethod
    {
        NO_ORDER_SELECTION,
        INCREMENTAL_ORDER,
        GOLDEN_SECTION,
        SIMULATED_ANNEALING
    };

    // STRUCTURES

    ///
    /// This structure contains the results from the model selection process.
    ///

    struct Results
    {

        void save(const string&) const;

        // Order selection results

        /// Pointer to a structure with the results from the incremental order selection algorithm.

        IncrementalOrder::IncrementalOrderResults* incremental_order_results_pointer;

        /// Pointer to a structure with the results from the golden section order selection algorithm.

        GoldenSectionOrder::GoldenSectionOrderResults* golden_section_order_results_pointer;

        /// Pointer to a structure with the results from the simulated annealing order selection algorithm.

        SimulatedAnnealingOrder::SimulatedAnnealingOrderResults* simulated_annealing_order_results_pointer;

        /// Pointer to a structure with the results from the growing inputs selection algorithm.

        GrowingInputs::GrowingInputsResults* growing_inputs_results_pointer;

        /// Pointer to a structure with the results from the pruning inputs selection algorithm.

        PruningInputs::PruningInputsResults* pruning_inputs_results_pointer;

        /// Pointer to a structure with the results from the genetic inputs selection algorithm.

        GeneticAlgorithm::GeneticAlgorithmResults* genetic_algorithm_results_pointer;
    };

    // METHODS

    // Get methods

    TrainingStrategy* get_training_strategy_pointer() const;
    bool has_training_strategy() const;

    const OrderSelectionMethod& get_order_selection_method() const;
    const InputsSelectionMethod& get_inputs_selection_method() const;

    IncrementalOrder* get_incremental_order_pointer() const;
    GoldenSectionOrder* get_golden_section_order_pointer() const;
    SimulatedAnnealingOrder* get_simulated_annealing_order_pointer() const;

    GrowingInputs* get_growing_inputs_pointer() const;
    PruningInputs* get_pruning_inputs_pointer() const;
    GeneticAlgorithm* get_genetic_algorithm_pointer() const;

    // Set methods

    void set_default();

    void set_display(const bool&);

    void set_training_strategy_pointer(TrainingStrategy*);

#ifdef __OPENNN_MPI__
    void set_MPI(TrainingStrategy*, const ModelSelection*);

    void set_inputs_selection_MPI(const ModelSelection*);
    void set_order_selection_MPI(const ModelSelection*);
#endif

    void set_order_selection_method(const OrderSelectionMethod&);
    void set_order_selection_method(const string&);

    void set_inputs_selection_method(const InputsSelectionMethod&);
    void set_inputs_selection_method(const string&);

    void set_approximation(const bool&);

    // Pointer methods

    void destruct_order_selection();

    void destruct_inputs_selection();

    // Cross validation methods

    Vector<NeuralNetwork> perform_k_fold_cross_validation(const size_t& = 4);
    Vector<NeuralNetwork> perform_random_cross_validation(const size_t& = 4, const double& = 0.25);
    Vector<NeuralNetwork> perform_positives_cross_validation();

    // Model selection methods

    void check() const;

    Vector<double> calculate_inputs_importance() const;

    Results perform_order_selection() const;

    Results perform_inputs_selection() const;

    Results perform_model_selection() const;

    // Serialization methods

    tinyxml2::XMLDocument* to_XML() const;
    void from_XML(const tinyxml2::XMLDocument&);

    void write_XML(tinyxml2::XMLPrinter&) const;

    void print() const;
    void save(const string&) const;
    void load(const string&);

private: 

    // MEMBERS

    /// Pointer to a training strategy object.

    TrainingStrategy* training_strategy_pointer;

    /// Pointer to a incremental order object to be used in the order selection.

    IncrementalOrder* incremental_order_pointer;

    /// Pointer to a golden section order object to be used in the order selection.

    GoldenSectionOrder* golden_section_order_pointer;

    /// Pointer to a simulated annealing order object to be used in the order selection.

    SimulatedAnnealingOrder* simulated_annelaing_order_pointer;

    /// Pointer to a growing inputs object to be used in the inputs selection.

    GrowingInputs* growing_inputs_pointer;

    /// Pointer to a pruning inputs object to be used in the inputs selection.

    PruningInputs* pruning_inputs_pointer;

    /// Pointer to a genetic algorithm object to be used in the inputs selection.

    GeneticAlgorithm* genetic_algorithm_pointer;

    /// Type of order selection algorithm.

    OrderSelectionMethod order_selection_method;

    /// Type of inputs selection algorithm.

    InputsSelectionMethod inputs_selection_method;

    /// Display messages to screen.

    bool display;
};

}

#endif
