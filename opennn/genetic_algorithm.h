//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G E N E T I C   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef GENETICALGORITHM_H
#define GENETICALGORITHM_H

// System includes

#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdio.h>
#include <random>

// OpenNN includes

#include "training_strategy.h"
#include "tensor_utilities.h"
#include "inputs_selection.h"
#include "config.h"
#include "testing_analysis.h"
namespace opennn
{
/// This concrete class represents a genetic algorithm, inspired by the process of natural selection[1] such as mutation,
/// crossover and selection.

///
/// This algorithm are commonly used in optimization and search problems. if the data_set has many inputs,
/// but we do not know how they affect the target,
/// then this algorithm provides the best possible combination of variables to optimize the problem.
///
/// \cite 1 Neural Designer "Genetic Algorithms for Feature Selection."
/// \ref https://www.neuraldesigner.com/blog/genetic_algorithms_for_feature_selection

class GeneticAlgorithm : public InputsSelection
{

public:

    // Constructors

    explicit GeneticAlgorithm();

    explicit GeneticAlgorithm(TrainingStrategy*);

    enum class InitializationMethod{Random,Correlations};

    // Get methods

    const Tensor <bool, 2>& get_population() const;


    const Tensor <type, 1>& get_training_errors() const;

    const Tensor <type, 1>& get_selection_errors() const;


    const Tensor <type, 1>& get_fitness() const;

    const Tensor <bool, 1>& get_selection() const;


    Index get_individuals_number() const;

    Index get_genes_number() const;


    const type& get_mutation_rate() const;

    const Index& get_elitism_size() const;

    const InitializationMethod& get_initialization_method() const;

    // Set methods

    virtual void set_default() final;

    void set_population(const Tensor<bool, 2>&);

    void set_individuals_number(const Index& new_individuals_number=4);

    void set_genes_number(const Index&);  

    void set_initialization_method(const GeneticAlgorithm::InitializationMethod&);

    void set_training_errors(const Tensor<type, 1>&);

    void set_selection_errors(const Tensor<type, 1>&);


    void set_fitness(const Tensor<type, 1>&);

    void set_mutation_rate(const type&);


    void set_elitism_size(const Index&);

    void set_maximum_epochs_number(const Index&);

    void set_initial_columns_indices(const Tensor<Index ,1>&);

    // GENETIC OPERATORS METHODS

    // Population methods

    void initialize_population();

    void initialize_population_random();

    void calculate_inputs_activation_probabilities();
    
    void initialize_population_correlations();

    type generate_random_between_0_and_1();

    void evaluate_population();

    void perform_fitness_assignment();

    Tensor<type, 1> calculate_selection_probabilities();

    // Selection methods

    void perform_selection();

    Index weighted_random(const Tensor<type, 1>&);

    // Crossover methods

    void perform_crossover();

    // Mutation methods

    void perform_mutation();

    // Check  methods

    void check_categorical_columns();

    Tensor <bool, 1> get_individual_variables_to_indexes (Tensor <bool, 1>&);

    Tensor <bool, 1> get_individual_columns(Tensor <bool, 1>&);

    Tensor <bool, 1> get_individual_variables(Tensor <bool,1>&);

    Tensor <Index, 1> get_selected_individuals_indices ();

    Tensor <Index, 1> get_individual_as_columns_indexes_from_variables( Tensor <bool, 1>&);

    void set_unused_columns(Tensor<Index, 1>&);

    Tensor<Index, 1> get_original_unused_columns();

    InputsSelectionResults perform_inputs_selection ()  final;

    // Serialization method

    Tensor <string, 2> to_string_matrix() const;

    bool calculate_random_bool();

    void from_XML(const tinyxml2::XMLDocument&);

    void write_XML(tinyxml2::XMLPrinter&) const;

    void print() const;
    
    void save(const string&) const;

    void load(const string&);

    Tensor <Tensor<type, 1>, 1> parameters;

private:
    
    /// Initial inputs columns

    Tensor <Index, 1> initial_columns_indices;
    Tensor <bool, 1> original_input_columns;

    /// Initial columns unused

    Tensor <Index, 1> original_unused_columns_indices;
    Tensor<bool, 1> original_unused_columns;

    /// Activation probabilities.
    
    Tensor <type, 1> inputs_activation_probabilities;

    /// Population matrix.

    Tensor <bool, 2> population;

    Tensor <type, 1> training_errors;

    Tensor <type, 1> selection_errors;


    /// Fitness of population.

    Tensor<type, 1> fitness;

    Tensor<bool, 1> selection;


    /// Performance of population.

    type mean_training_error;

    type mean_selection_error;

    type mean_inputs_number;
    
    Tensor <bool, 2> optimal_individuals_history;

    Tensor <Index, 1> original_input_columns_indices;

    Tensor <Index, 1> original_target_columns_indices;

    Index genes_number;

    /// Mutation rate.
    /// The mutation rate value must be between 0 and 1.
    /// This is a parameter of the mutation operator.

    type mutation_rate;

    /// Elitism size.
    /// It represents the number of individuals which will always be selected for recombination.
    /// This is a parameter of the selection operator.

    Index elitism_size;

    InitializationMethod initialization_method;

};

}

#endif
